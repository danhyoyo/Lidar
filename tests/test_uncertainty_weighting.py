import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))

import torch
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty


class TestTemperatureSoftmaxUncertainty(unittest.TestCase):
    def test_initial_weights_are_strictly_equal_to_one(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=5, temperature=2.0)
        task_losses = {
            "cls": torch.tensor(1.0),
            "offset": torch.tensor(0.5),
            "size": torch.tensor(0.8),
            "yaw": torch.tensor(0.2),
            "geo": torch.tensor(0.4),
        }
        total, weights = weighting(task_losses)
        # Sum of nominal weights must be 5.0
        self.assertAlmostEqual(sum(weights.values()), 5.0, places=5)
        for name, w in weights.items():
            self.assertAlmostEqual(w, 1.0, places=5)

    def test_gradient_conservation_and_updates(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=3, temperature=1.5, clamp_bound=3.0)
        task_losses = {
            "a": torch.tensor(5.0, requires_grad=True),
            "b": torch.tensor(0.2, requires_grad=True),
            "c": torch.tensor(0.1, requires_grad=True),
        }
        total, weights = weighting(task_losses)
        total.backward()

        self.assertTrue(torch.isfinite(weighting.log_scales.grad).all())
        self.assertAlmostEqual(sum(weights.values()), 3.0, places=5)

    def test_bound_clamping_prevents_extreme_starvation(self):
        weighting = TemperatureSoftmaxUncertainty(num_tasks=2, temperature=1.0, clamp_bound=2.0)
        with torch.no_grad():
            weighting.log_scales.copy_(torch.tensor([-100.0, 100.0]))
        weights = weighting.get_task_weights()
        min_weight = min(weights.values())
        self.assertGreater(min_weight, 0.01)
        self.assertAlmostEqual(sum(weights.values()), 2.0, places=5)

    def test_smooth_tanh_gradient_non_zero_at_saturation(self):
        weighting = TemperatureSoftmaxUncertainty(task_names=("t1", "t2"), temperature=1.0, clamp_bound=3.0)
        # Establish baseline scales first so step 2 observes a deviation
        weighting({"t1": torch.tensor(1.0), "t2": torch.tensor(1.0)})
        # Initialize at extreme boundary past clamp_bound
        with torch.no_grad():
            weighting.log_scales.copy_(torch.tensor([5.0, -5.0]))
        task_losses = {"t1": torch.tensor(1.0), "t2": torch.tensor(2.0)}
        total, _ = weighting(task_losses)
        total.backward()

        # Gradient must remain strictly non-zero (eliminates hard clamp saturation dead zone)
        self.assertTrue(torch.isfinite(weighting.log_scales.grad).all())
        self.assertTrue((weighting.log_scales.grad != 0).all())

    def test_missing_task_key_raises_key_error(self):
        weighting = TemperatureSoftmaxUncertainty(task_names=("cls", "offset", "geo"))
        incomplete_losses = {"cls": torch.tensor(1.0), "offset": torch.tensor(2.0)}
        with self.assertRaises(KeyError):
            weighting(incomplete_losses)

    def test_scale_calibration_prevents_simplex_collapse(self):
        # Disparate loss scales: cls=0.03, geo=0.60
        weighting = TemperatureSoftmaxUncertainty(
            task_names=["cls", "offset", "size", "yaw", "geo"],
            temperature=2.0,
            clamp_bound=3.0,
            ema_momentum=0.9,
        )
        opt = torch.optim.Adam(weighting.parameters(), lr=0.05)

        losses = {
            "cls": torch.tensor(0.03),
            "offset": torch.tensor(0.20),
            "size": torch.tensor(0.25),
            "yaw": torch.tensor(0.30),
            "geo": torch.tensor(0.60),
        }

        for _ in range(100):
            opt.zero_grad()
            total_loss, weights = weighting(losses)
            total_loss.backward()
            opt.step()

        final_weights = weighting.get_task_weights()
        for name, w in final_weights.items():
            self.assertGreater(w, 0.4, f"Task {name} suffered simplex collapse: weight {w} < 0.4")
            self.assertLess(w, 2.0, f"Task {name} monopolized weight budget: weight {w} > 2.0")
        self.assertAlmostEqual(sum(final_weights.values()), 5.0, places=4)


if __name__ == "__main__":
    unittest.main()
