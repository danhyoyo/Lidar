import sys
import unittest
from pathlib import Path
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from train import build_optimizer


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


if __name__ == "__main__":
    unittest.main()
