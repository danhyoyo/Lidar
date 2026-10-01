import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from train import build_optimizer, build_scheduler, build_parser


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean(dim=[-2, -1]))


class DummyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))


class TestOptimizerBuilder(unittest.TestCase):
    def test_build_optimizer_adamw_groups(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.002,
                "weight_decay": 0.0001,
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 3)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]
        crit_group = optimizer.param_groups[2]

        self.assertEqual(decay_group["weight_decay"], 0.0001)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        self.assertEqual(crit_group["weight_decay"], 0.0)
        self.assertEqual(decay_group["lr"], 0.002)

    def test_build_optimizer_adam_fallback(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adam",
                "learning_rate": 0.005,
                "weight_decay": 0.0005,
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.005)


class TestSchedulerBuilder(unittest.TestCase):
    def test_build_scheduler_cosine_warmup(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.002,
                "scheduler": "cosine",
                "warmup_epochs": 5,
                "min_lr": 1e-6,
            }
        }
        epochs = 100
        optimizer = build_optimizer(model, criterion, config)
        scheduler = build_scheduler(optimizer, config, epochs)

        # Initial LR before any step (first warmup epoch start)
        self.assertLess(optimizer.param_groups[0]["lr"], 0.002)

        # Step through 5 warmup epochs
        for ep in range(5):
            optimizer.step()
            scheduler.step()

        # At end of warmup (milestone epoch 5), LR reaches peak 0.002
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.002, places=5)

        # Step remaining 95 epochs
        for ep in range(95):
            optimizer.step()
            scheduler.step()

        # At end of 100 epochs, LR reaches min_lr
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6, places=5)

    def test_build_scheduler_multistep_compatibility(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "learning_rate": 0.005,
                "scheduler": "multistep",
                "lr_decay_at": [65, 85],
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        scheduler = build_scheduler(optimizer, config, 100)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)

    def test_build_scheduler_state_dict_save_load(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.002,
                "scheduler": "cosine",
                "warmup_epochs": 5,
                "min_lr": 1e-6,
            }
        }
        optimizer1 = build_optimizer(model, criterion, config)
        scheduler1 = build_scheduler(optimizer1, config, 100)
        for _ in range(10):
            optimizer1.step()
            scheduler1.step()
        sd = scheduler1.state_dict()
        opt_sd = optimizer1.state_dict()

        optimizer2 = build_optimizer(model, criterion, config)
        scheduler2 = build_scheduler(optimizer2, config, 100)
        optimizer2.load_state_dict(opt_sd)
        scheduler2.load_state_dict(sd)
        self.assertAlmostEqual(
            optimizer1.param_groups[0]["lr"],
            optimizer2.param_groups[0]["lr"],
            places=6,
        )
        self.assertEqual(scheduler1.get_last_lr(), scheduler2.get_last_lr())

        scheduler1.step()
        scheduler2.step()
        self.assertAlmostEqual(
            optimizer1.param_groups[0]["lr"],
            optimizer2.param_groups[0]["lr"],
            places=6,
        )


class TestGradClippingAndParser(unittest.TestCase):
    def test_parser_grad_clip_norm(self):
        parser = build_parser()
        args = parser.parse_args([
            "--config", "configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_oga.json",
            "--detector-root", "detector",
            "--output-root", "artifacts/kitti",
            "--grad-clip-norm", "5.0",
        ])
        self.assertEqual(args.grad_clip_norm, 5.0)

    def test_gradient_clipping_bounds_norm(self):
        model = DummyModel()
        criterion = DummyCriterion()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        x = torch.randn(2, 8, 16, 16)
        out = model(x)
        loss = out.sum() * 10000.0
        loss.backward()

        params = [p for p in model.parameters() if p.requires_grad] + [
            p for p in criterion.parameters() if p.requires_grad
        ]
        pre_clip_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params if p.grad is not None))
        self.assertGreater(pre_clip_norm.item(), 10.0)

        torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
        post_clip_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in params if p.grad is not None))
        self.assertAlmostEqual(post_clip_norm.item(), 10.0, places=3)
        optimizer.step()

    def test_build_optimizer_weight_decay_0001(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.0007,
                "weight_decay": 0.001,
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.001)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)


if __name__ == "__main__":
    unittest.main()

