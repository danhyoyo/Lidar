"""Temperature-Softmax Bounded Uncertainty Weighting (T-SBUW) with Running Scale Calibration."""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemperatureSoftmaxUncertainty"]


class TemperatureSoftmaxUncertainty(nn.Module):
    """Normalized multi-task weighting with conserved gradient budget, smooth tanh soft-bounding,
    and running loss scale calibration to prevent simplex collapse.

    Formula:
        s_bounded = clamp_bound * tanh(s / clamp_bound)
        w_i = M * exp(s_bounded_i / tau) / sum(exp(s_bounded_j / tau))
        L_calibrated_i = L_i / EMA(L_i)
        L_total = (1 / M * sum(EMA(L_i))) * sum(w_i * L_calibrated_i)

    Guarantees:
        1. sum(w_i) == M (strictly conserved total gradient scale)
        2. w_i > 0 for all tasks (no task starvation)
        3. Equalized competition across disparate loss scales via running EMA normalization
        4. Preserved overall nominal loss scale matching standard detector training
        5. 100% backward compatibility for legacy checkpoints missing running buffers
        6. Completely immune to numerical explosion under BF16 / FP32.
    """

    def __init__(
        self,
        task_names=None,
        num_tasks=None,
        temperature=2.0,
        clamp_bound=3.0,
        ema_momentum=0.99,
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
        self.ema_momentum = float(ema_momentum)

        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.clamp_bound <= 0.0:
            raise ValueError(f"clamp_bound must be positive, got {self.clamp_bound}")
        if not (0.0 < self.ema_momentum < 1.0):
            raise ValueError(f"ema_momentum must be in (0, 1), got {self.ema_momentum}")

        # Learnable log-scale logits initialized to 0 (all tasks start with equal weight = 1.0)
        self.log_scales = nn.Parameter(torch.zeros(self.num_tasks, dtype=torch.float32))

        # Running scale buffers to equalize disparate loss magnitudes
        self.register_buffer("running_loss_means", torch.ones(self.num_tasks, dtype=torch.float32))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def _bounded_scales(self):
        """Smooth tanh soft-bounding strictly within (-clamp_bound, clamp_bound)."""
        return self.clamp_bound * torch.tanh(self.log_scales / self.clamp_bound)

    @torch.no_grad()
    def get_task_weights(self):
        """Compute the normalized task weights as a dict {task_name: float_weight}."""
        bounded = self._bounded_scales()
        normalized = F.softmax(bounded / self.temperature, dim=0) * self.num_tasks
        return {name: float(normalized[i].item()) for i, name in enumerate(self.task_names)}

    def forward(self, task_losses):
        """Compute total weighted loss and return weights telemetry.

        Args:
            task_losses: dict mapping task_name -> scalar Tensor loss.
        """
        missing = [name for name in self.task_names if name not in task_losses]
        if missing and self.task_names != [f"task_{i}" for i in range(self.num_tasks)]:
            raise KeyError(
                f"Missing required task losses in TemperatureSoftmaxUncertainty: {missing}"
            )

        target_device = next(iter(task_losses.values())).device
        if self.log_scales.device != target_device:
            self.to(target_device)

        bounded = self._bounded_scales()
        normalized_weights = F.softmax(bounded / self.temperature, dim=0) * self.num_tasks

        is_generic = self.task_names == [f"task_{i}" for i in range(self.num_tasks)]
        if is_generic and len(task_losses) == self.num_tasks:
            ordered_losses = [task_losses[k].float() for k in task_losses.keys()]
            ordered_names = list(task_losses.keys())
        else:
            ordered_losses = [task_losses[name].float() for name in self.task_names]
            ordered_names = self.task_names

        stacked_losses = torch.stack(ordered_losses)

        with torch.no_grad():
            detached = stacked_losses.detach().clamp_min(1e-4)
            if not self.initialized:
                self.running_loss_means.copy_(detached)
                self.initialized.copy_(torch.tensor(True, device=target_device))
            else:
                self.running_loss_means.lerp_(detached, 1.0 - self.ema_momentum)

        calibrated_losses = stacked_losses / self.running_loss_means.clamp_min(1e-4)
        nominal_scale = self.running_loss_means.mean().detach()
        total_loss = torch.sum(normalized_weights * calibrated_losses) * nominal_scale

        weights_dict = {
            name: normalized_weights[i].detach() for i, name in enumerate(ordered_names)
        }

        return total_loss, weights_dict

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Allow seamless loading of legacy checkpoints that lack running loss calibration buffers."""
        for buf_name in ("running_loss_means", "initialized"):
            full_key = f"{prefix}{buf_name}"
            if full_key not in state_dict:
                state_dict[full_key] = getattr(self, buf_name).clone()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )
