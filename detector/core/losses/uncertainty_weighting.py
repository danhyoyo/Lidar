"""Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) for multi-task loss balancing."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemperatureSoftmaxUncertainty"]


class TemperatureSoftmaxUncertainty(nn.Module):
    """Normalized multi-task weighting with conserved gradient budget and soft bound limits.

    Formula:
        s_clamped = clamp(s, -clamp_bound, clamp_bound)
        w_i = M * exp(s_i / tau) / sum(exp(s_j / tau))
        L_total = sum(w_i * L_i)

    Guarantees:
        1. sum(w_i) == M (strictly conserved total gradient scale)
        2. w_i > 0 for all tasks (no task starvation)
        3. Completely immune to numerical explosion under BF16 / FP32.
    """

    def __init__(
        self,
        task_names=None,
        num_tasks=None,
        temperature=2.0,
        clamp_bound=3.0,
    ):
        super().__init__()
        if task_names is not None:
            self.task_names = list(task_names)
            self.num_tasks = len(self.task_names)
        elif num_tasks is not None:
            self.num_tasks = int(num_tasks)
            self.task_names = [f"task_{i}" for i in range(self.num_tasks)]
        else:
            self.task_names = ["cls", "offset", "size", "yaw", "geo"]
            self.num_tasks = len(self.task_names)

        self.temperature = float(temperature)
        self.clamp_bound = float(clamp_bound)

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.clamp_bound <= 0.0:
            raise ValueError(f"clamp_bound must be positive, got {self.clamp_bound}")

        # Learnable log-scale logits initialized to 0 (all tasks start with equal weight = 1.0)
        self.log_scales = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

    def get_task_weights(self):
        """Compute the normalized task weights as a dict {task_name: float_weight}."""
        clamped_scales = self.log_scales.clamp(-self.clamp_bound, self.clamp_bound)
        normalized = F.softmax(clamped_scales / self.temperature, dim=0) * self.num_tasks
        return {name: float(normalized[i].item()) for i, name in enumerate(self.task_names)}

    def forward(self, task_losses):
        """Compute total weighted loss and return weights telemetry.

        Args:
            task_losses: dict mapping task_name -> scalar Tensor loss.
        """
        clamped_scales = self.log_scales.clamp(-self.clamp_bound, self.clamp_bound)
        normalized_weights = F.softmax(clamped_scales / self.temperature, dim=0) * self.num_tasks

        total_loss = torch.zeros((), device=self.log_scales.device)
        weights_dict = {}

        # If keys of task_losses match self.task_names
        if all(name in task_losses for name in self.task_names):
            for i, name in enumerate(self.task_names):
                w = normalized_weights[i]
                total_loss = total_loss + w * task_losses[name]
                weights_dict[name] = float(w.detach().item())
        else:
            # Map sequentially to the dict items
            for i, (name, loss_val) in enumerate(task_losses.items()):
                if i >= self.num_tasks:
                    break
                w = normalized_weights[i]
                total_loss = total_loss + w * loss_val
                weights_dict[name] = float(w.detach().item())

        return total_loss, weights_dict
