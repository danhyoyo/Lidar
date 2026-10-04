import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from train import build_optimizer, build_scheduler, resolve_training_epochs, clip_optimizer_gradients


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
    def test_screening_keeps_the_full_schedule_and_supports_resume(self):
        cfg = {'epochs': 100, 'stop_after_epoch': 50}
        self.assertEqual(resolve_training_epochs(cfg), (100, 50))
        self.assertEqual(resolve_training_epochs(cfg, stop_override=100), (100, 100))
        self.assertEqual(resolve_training_epochs(cfg, epochs_override=1), (1, 1))
        for stop in [0, -1, 101]:
            with self.assertRaises(ValueError):
                resolve_training_epochs(cfg, stop_override=stop)

    def test_clipping_covers_model_and_criterion_optimizer_parameters(self):
        model, criterion = DummyModel(), DummyCriterion()
        optimizer = build_optimizer(model, criterion, {'train': {'learning_rate': .0007}})
        for group in optimizer.param_groups:
            for param in group['params']:
                param.grad = torch.full_like(param, 100.)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        clip_optimizer_gradients(optimizer, scaler, 10.)
        gradients = [p.grad.flatten() for group in optimizer.param_groups for p in group['params']]
        self.assertLessEqual(torch.cat(gradients).norm().item(), 10.00001)

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


if __name__ == "__main__":
    unittest.main()
