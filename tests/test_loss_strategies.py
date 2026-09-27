import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

import torch
import torch.nn as nn
from core.losses.strategies import (
    BaseLossStrategy,
    BaselineLossStrategy,
    UwagLossStrategy,
    OgaLossStrategy,
    register_loss_strategy,
    build_loss_strategy,
    get_available_loss_strategies,
)
from core.losses.loss_fn import LossFunction


class TestLossStrategies(unittest.TestCase):
    def setUp(self):
        self.B, self.H, self.W = 2, 8, 8
        self.pred = {
            "cls": torch.randn(self.B, 3, self.H, self.W, requires_grad=True),
            "offset": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "size": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
            "yaw": torch.randn(self.B, 2, self.H, self.W, requires_grad=True),
        }
        self.target = {
            "cls": torch.sigmoid(torch.randn(self.B, 3, self.H, self.W)),
            "offset": torch.zeros(self.B, 2, self.H, self.W),
            "size": torch.zeros(self.B, 2, self.H, self.W),
            "yaw": torch.tensor([1.0, 0.0]).view(1, 2, 1, 1).expand(self.B, 2, self.H, self.W),
            "reg_mask": torch.ones(self.B, self.H, self.W),
        }

    def test_default_strategies_registered(self):
        available = get_available_loss_strategies()
        for name in ("baseline", "uwag", "oga"):
            self.assertIn(name, available)

    def test_unsupported_strategy_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            build_loss_strategy("nonexistent_loss", "gaussian")
        self.assertIn("Unsupported loss name", str(ctx.exception))
        self.assertIn("Available strategies", str(ctx.exception))

    def test_baseline_strategy(self):
        strategy = BaselineLossStrategy("gaussian")
        loss_dict = strategy(self.pred, self.target)
        self.assertIn("loss", loss_dict)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        loss_dict["loss"].backward()
        self.assertTrue(self.pred["offset"].grad is not None)

    def test_uwag_strategy(self):
        strategy = UwagLossStrategy("gaussian", {"geometric_weight": 0.2})
        pred = {k: v.clone().detach().requires_grad_(True) for k, v in self.pred.items()}
        loss_dict = strategy(pred, self.target)
        self.assertIn("loss", loss_dict)
        self.assertIn("weight_cls", loss_dict)
        loss_dict["loss"].backward()
        self.assertTrue(pred["offset"].grad is not None)
        self.assertTrue(strategy.log_scales.grad is not None)

    def test_oga_strategy(self):
        strategy = OgaLossStrategy("gaussian", {"temperature": 2.0, "clamp_bound": 3.0})
        pred = {k: v.clone().detach().requires_grad_(True) for k, v in self.pred.items()}
        loss_dict = strategy(pred, self.target)
        self.assertIn("loss", loss_dict)
        self.assertIn("corner_dist", loss_dict)
        self.assertIn("proj_giou", loss_dict)
        self.assertIn("weight_geo", loss_dict)
        loss_dict["loss"].backward()
        self.assertTrue(pred["offset"].grad is not None)

    def test_legacy_checkpoint_state_dict_compatibility(self):
        # 1. UWAG legacy state dict without 'strategy.' prefix
        crit_uwag = LossFunction("gaussian", {"name": "uwag"})
        legacy_uwag_sd = {"log_scales": torch.tensor([0.5, 0.5, 0.5, 0.5])}
        crit_uwag.load_state_dict(legacy_uwag_sd, strict=True)
        self.assertTrue(torch.allclose(crit_uwag.log_scales, torch.tensor([0.5, 0.5, 0.5, 0.5])))

        # 2. OGA legacy state dict without 'strategy.' prefix
        crit_oga = LossFunction("gaussian", {"name": "oga"})
        legacy_oga_sd = {"weighting.log_scales": torch.ones(5) * 0.2}
        crit_oga.load_state_dict(legacy_oga_sd, strict=True)
        self.assertTrue(torch.allclose(crit_oga.weighting.log_scales, torch.ones(5) * 0.2))

        # 3. Modern state dict with 'strategy.' prefix
        current_sd = crit_oga.state_dict()
        crit_oga_new = LossFunction("gaussian", {"name": "oga"})
        crit_oga_new.load_state_dict(current_sd, strict=True)
        self.assertTrue(torch.allclose(crit_oga_new.weighting.log_scales, torch.ones(5) * 0.2))

    def test_custom_strategy_registration(self):
        @register_loss_strategy("custom_dummy_loss")
        class DummyStrategy(BaseLossStrategy):
            def forward(self, pred, target):
                return {"loss": (pred["offset"].sum() + pred["cls"].sum()) * 0.0 + 1.0}

        self.assertIn("custom_dummy_loss", get_available_loss_strategies())

        criterion = LossFunction("gaussian", {"name": "custom_dummy_loss"})
        loss_dict = criterion(self.pred, self.target)
        self.assertEqual(loss_dict["loss"].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
