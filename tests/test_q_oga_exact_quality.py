"""Quality-only ablation, optional curriculum, and genuine positive peaks."""

import copy
import json
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "detector/core/datasets"))
sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
from common import build_model, generate_run_name
from core.losses.loss_fn import LossFunction
from core.losses.iou_targets import compute_iou_targets
from core.losses.quality_focal_loss import quality_focal_loss
import train
from train import build_optimizer, set_loss_epoch


def peak_batch(disjoint=True):
    pred = {"cls": torch.zeros(1, 3, 1, 2, requires_grad=True),
            "offset": torch.tensor([[[[1.5 if disjoint else 0., 0.]], [[0., 0.]]]], requires_grad=True),
            "size": torch.zeros(1, 2, 1, 2, requires_grad=True),
            "yaw": torch.tensor([[[[1., 1.]], [[0., 0.]]]], requires_grad=True)}
    target = {"cls": torch.tensor([[[[0., 0.]], [[1., .5]], [[0., 0.]]]]),
              "offset": torch.zeros(1, 2, 1, 2), "size": torch.zeros(1, 2, 1, 2),
              "yaw": pred["yaw"].detach().clone(), "reg_mask": torch.ones(1, 1, 2)}
    return pred, target


class TestQOgaExactQuality(unittest.TestCase):
    def criterion(self, **options):
        return LossFunction("gaussian", {"name": "q_oga", "quality_target": "rotated_iou", **options})

    def test_exact_target_changes_cls_but_preserves_regression(self):
        pred, target = peak_batch()
        legacy = LossFunction("gaussian", {"name": "q_oga"}).eval()
        exact = self.criterion().eval()
        actual, old = exact(pred, target), legacy(pred, target)
        expected_q = torch.zeros(1, 1, 2)
        torch.testing.assert_close(actual["cls"], quality_focal_loss(pred["cls"], target["cls"], expected_q))
        self.assertNotAlmostEqual(actual["cls"].item(), old["cls"].item(), places=6)
        for name in ("offset", "size", "yaw", "geo", "corner_dist", "proj_giou", "rda_mean_weight"):
            torch.testing.assert_close(actual[name], old[name], rtol=0, atol=0)
        self.assertEqual(actual["quality_iou_mean"].item(), 0.)
        self.assertEqual(actual["quality_iou_zero_fraction"].item(), 1.)
        self.assertEqual(actual["quality_peak_count"].item(), 1.)

    def test_curriculum_endpoints_and_gaussian_neighbor_mask(self):
        pred, target = peak_batch()
        criterion = self.criterion(quality_warmup_epochs=8).eval()
        self.assertTrue(hasattr(criterion, "set_epoch"), "Q-OGA curriculum epoch hook is missing")
        for epoch, expected_target in ((0, 1.), (4, .5), (8, 0.), (12, 0.)):
            criterion.set_epoch(epoch)
            actual = criterion(pred, target)
            q = torch.tensor([[[expected_target, 0.]]])
            torch.testing.assert_close(actual["cls"], quality_focal_loss(pred["cls"], target["cls"], q))
            self.assertEqual(actual["quality_target_mean"].item(), expected_target)
            self.assertEqual(actual["quality_iou_mean"].item(), 0.)
        # Changing quality outside the peak must leave Gaussian-negative loss unchanged.
        q1, q2 = torch.tensor([[[0., 0.]]]), torch.tensor([[[0., 1.]]])
        torch.testing.assert_close(quality_focal_loss(pred["cls"], target["cls"], q1),
                                   quality_focal_loss(pred["cls"], target["cls"], q2))

    def test_quality_is_detached_from_classification(self):
        pred, target = peak_batch(disjoint=False)
        q = compute_iou_targets(pred, target, method="rotated_iou")
        self.assertFalse(q.requires_grad)
        cls = quality_focal_loss(pred["cls"], target["cls"], q)
        gradients = torch.autograd.grad(cls, tuple(pred.values()), allow_unused=True)
        self.assertIsNotNone(gradients[0])
        self.assertTrue(all(g is None for g in gradients[1:]))

    def test_fp32_bf16_backward_and_empty_objects(self):
        for dtype in (torch.float32, torch.bfloat16):
            for empty in (False, True):
                pred, target = peak_batch()
                if empty:
                    target["cls"].zero_()
                    target["reg_mask"].zero_()
                criterion = self.criterion()
                with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
                    actual = criterion(pred, target)
                self.assertIn("quality_iou_mean", actual)
                self.assertTrue(all(torch.isfinite(v).all() for v in actual.values()))
                actual["loss"].backward()
                for head in pred.values():
                    self.assertIsNotNone(head.grad)
                    self.assertTrue(torch.isfinite(head.grad).all())

    def test_epoch_state_resume_and_training_hook(self):
        criterion = self.criterion(quality_warmup_epochs=8)
        self.assertTrue(hasattr(train, "set_loss_epoch"), "Trainer does not set the loss curriculum epoch")
        train.set_loss_epoch(criterion, 4)
        resumed = self.criterion(quality_warmup_epochs=8)
        resumed.load_state_dict(copy.deepcopy(criterion.state_dict()))
        pred, target = peak_batch()
        self.assertEqual(resumed(pred, target)["quality_target_mean"].item(), .5)
        train.set_loss_epoch(resumed, 8)
        self.assertEqual(resumed(pred, target)["quality_target_mean"].item(), 0.)
        legacy = LossFunction("gaussian", {"name": "q_oga"})
        exact = self.criterion()
        self.assertEqual(set(legacy.state_dict()), set(exact.state_dict()))
        exact.load_state_dict(legacy.state_dict(), strict=True)

    def test_training_quality_aggregation_uses_peak_counts(self):
        self.assertTrue(hasattr(train, "QualityMetrics"), "Training quality statistics are missing")
        stats = train.QualityMetrics()
        criterion = self.criterion().eval()
        pred, target = peak_batch()
        stats.update(criterion(pred, target))  # One non-overlapping peak.
        pred, target = peak_batch(disjoint=False)
        target["cls"][:, 2, 0, 1] = 1.  # Two exact-overlap peaks.
        stats.update(criterion(pred, target))
        pred, target = peak_batch()
        target["cls"].zero_()
        target["reg_mask"].zero_()
        stats.update(criterion(pred, target))  # Empty batch does not bias means.
        actual = stats.summarize()
        self.assertAlmostEqual(actual["quality_iou_mean"], 2 / 3, places=6)
        self.assertAlmostEqual(actual["quality_iou_zero_fraction"], 1 / 3, places=6)
        self.assertEqual(actual["quality_peak_count"], 3)
        stats = train.QualityMetrics()
        stats.update(LossFunction("gaussian", {"name": "q_oga"})(pred, target))
        self.assertEqual(stats.summarize(), {})

    def test_invalid_modes_and_unsupervised_iqa_are_rejected(self):
        for options in ({"quality_target": "wrong"}, {"quality_warmup_epochs": -1},
                        {"quality_warmup_epochs": 1.5}, {"use_iou": True},
                        {"quality_target": "mgiou", "quality_warmup_epochs": 8}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.criterion(**options)
        with self.assertRaises(ValueError):
            LossFunction("binary", {"name": "q_oga", "quality_target": "rotated_iou"})
        criterion = self.criterion(quality_warmup_epochs=8)
        pred, target = peak_batch()
        with self.assertRaisesRegex(ValueError, "set_epoch"):
            criterion(pred, target)
        for epoch in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                criterion.set_epoch(epoch)
        pred["iou"] = torch.zeros_like(pred["offset"][:, :1])
        with self.assertRaisesRegex(ValueError, "IQA"):
            self.criterion()(pred, target)

    def test_unassigned_peak_fails_before_silent_zero_quality(self):
        pred, target = peak_batch()
        target["reg_mask"].zero_()
        with self.assertRaisesRegex(ValueError, "assigned regression"):
            self.criterion()(pred, target)

    def test_quality_candidates_preserve_the_reference_recipe(self):
        def load_config(name):
            path = ROOT / "configs/experiments/qoga_quality" / name
            self.assertTrue(path.is_file(), f"Missing single-seed experiment: {path}")
            return json.loads(path.read_text())

        configs = [load_config(name) for name in
                   ("legacy_s42.json", "exact_iou_s42.json", "exact_iou_warmup8_s42.json")]
        run_names = []
        normalized = []
        for config in configs:
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["train"]["epochs"], 100)
            self.assertEqual(config["train"]["warmup_epochs"], 8)
            self.assertEqual(config["train"]["precision"], "bf16")
            self.assertTrue(config["train"]["compile_model"])
            self.assertFalse(config["model"]["header_use_iou"])
            self.assertEqual(config["loss"]["name"], "q_oga")
            run_names.append(generate_run_name(config, seed=42))
            recipe = copy.deepcopy(config)
            recipe.pop("experiment", None)
            recipe.pop("note", None)
            recipe["loss"].pop("quality_target", None)
            recipe["loss"].pop("quality_warmup_epochs", None)
            normalized.append(recipe)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])
        self.assertEqual(len(set(run_names)), 3)

    def test_selected_model_and_optimizer_support_bf16_training_step(self):
        def load_config(name):
            return json.loads((ROOT / "configs/experiments/qoga_quality" / name).read_text())

        for name in ("exact_iou_s42.json", "exact_iou_warmup8_s42.json"):
            with self.subTest(config_name=name):
                config = load_config(name)
                torch.manual_seed(42)
                model = build_model(config)
                criterion = LossFunction("gaussian", config["loss"])
                set_loss_epoch(criterion, 0)
                optimizer = build_optimizer(model, criterion, config)
                voxel = torch.randn(2, 8, 64, 64)
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    pred = model(voxel)
                    B, _, H, W = pred["offset"].shape
                    target = {
                        "cls": torch.zeros_like(pred["cls"], dtype=torch.float32),
                        "offset": torch.zeros_like(pred["offset"], dtype=torch.float32),
                        "size": torch.zeros_like(pred["size"], dtype=torch.float32),
                        "yaw": torch.zeros_like(pred["yaw"], dtype=torch.float32),
                        "reg_mask": torch.zeros(B, H, W),
                    }
                    target["yaw"][:, 0] = 1
                    target["cls"][:, 1, H // 2, W // 2] = 1
                    target["reg_mask"][:, H // 2, W // 2] = 1
                    losses = criterion(pred, target)
                self.assertTrue(torch.isfinite(losses["loss"]))
                losses["loss"].backward()
                model_grads = [p.grad for p in model.parameters() if p.grad is not None]
                self.assertTrue(all(torch.isfinite(g).all() for g in model_grads))
                optimizer.step()
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
