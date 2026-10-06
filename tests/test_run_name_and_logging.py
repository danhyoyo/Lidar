import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from common import generate_run_name, create_experiment_config


class TestRunNameAndLogging(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(REPO_ROOT / "configs" / "config.json", "r", encoding="utf-8") as f:
            cls.base_cfg = json.load(f)

    def test_run_name_format_cumulative(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "scale_gated_fpn": True,
                "c4_attention_scales": [3, 5],
                "use_reparam": True,
                "header_use_iou": True,
            },
            "loss": {"name": "oga", "use_iou": True},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-ms_litemla-reparam-s42",
        )

    def test_run_name_format_iqa(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "scale_gated_fpn": True,
                "c4_attention_scales": [],
                "use_reparam": True,
                "header_use_iou": True,
            },
            "loss": {"name": "oga", "use_iou": True},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-reparam-s42",
        )

    def test_run_name_format_baseline_legacy35(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "backbone": "mobilepixor",
                "scale_gated_fpn": False,
                "c4_attention": "none",
                "c4_attention_scales": [],
                "header_use_iou": False,
                "use_reparam": False,
            },
            "loss": {"name": "baseline", "use_iou": False},
            "data": {"bev_encoding": {"name": "binary_slices"}},
            "augmentation": {"p": 0.0},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name, "mobilepixor-noaug-baseline_loss-legacy35-baseline_iou-s42"
        )

    def test_run_name_format_baseline_a4(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "backbone": "mobilepixor_coordatt",
                "scale_gated_fpn": True,
                "c4_attention": "none",
                "c4_attention_scales": [],
                "header_use_iou": False,
                "use_reparam": False,
            },
            "loss": {"name": "uwag", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixor_coordatt-standard_aug-uwag_loss-rich8-baseline_iou-sgfpn-s42",
        )

    def test_run_name_format_mobilepixor_ablation(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "backbone": "mobilepixor",
                "scale_gated_fpn": False,
                "c4_attention": "none",
                "c4_attention_scales": [],
                "header_use_iou": False,
                "use_reparam": False,
            },
            "loss": {"name": "baseline", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixor-standard_aug-baseline_loss-rich8-baseline_iou-s42",
        )

    def test_run_name_format_mobilepixornext_pure_baseline(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "backbone": "mobilepixornext",
                "scale_gated_fpn": False,
                "c4_attention": "none",
                "c4_attention_scales": [],
                "header_use_iou": False,
                "use_reparam": False,
            },
            "loss": {"name": "baseline", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixornext-standard_aug-baseline_loss-rich8-baseline_iou-s42",
        )

    def test_run_name_format_ms_litemla(self):
        cfg = create_experiment_config(self.base_cfg, {
            "model": {
                "backbone": "mobilepixornext",
                "scale_gated_fpn": True,
                "c4_attention": "litemla",
                "c4_attention_scales": [3, 5],
                "header_use_iou": False,
                "use_reparam": False,
            },
            "loss": {"name": "baseline", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(
            run_name,
            "mobilepixornext-standard_aug-baseline_loss-rich8-baseline_iou-sgfpn-ms_litemla-s42",
        )

    def test_run_name_distinguishes_truc_b_variants(self):
        cfg1 = create_experiment_config(self.base_cfg, {
            "model": {
                "scale_gated_fpn": True,
                "c4_attention_scales": [],
                "use_reparam": True,
                "header_use_iou": False,
            },
            "loss": {"name": "q_oga", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        cfg2 = create_experiment_config(self.base_cfg, {
            "model": {
                "scale_gated_fpn": True,
                "c4_attention_scales": [3, 5],
                "use_reparam": True,
                "header_use_iou": False,
            },
            "loss": {"name": "q_oga", "use_iou": False},
            "data": {"bev_encoding": {"name": "rich8"}},
            "augmentation": {"p": 0.5},
        })
        name1 = generate_run_name(cfg1, seed=42)
        name2 = generate_run_name(cfg2, seed=42)
        self.assertNotEqual(name1, name2)
        self.assertNotIn("ms_litemla", name1)
        self.assertIn("ms_litemla", name2)
        self.assertEqual(
            name1,
            "mobilepixornext-standard_aug-q_oga_loss-rich8-baseline_iou-sgfpn-reparam-s42",
        )
        self.assertEqual(
            name2,
            "mobilepixornext-standard_aug-q_oga_loss-rich8-baseline_iou-sgfpn-ms_litemla-reparam-s42",
        )

    def test_all_configs_generate_valid_run_name_format(self):
        configs = list((REPO_ROOT / "configs").rglob("*.json"))
        self.assertGreaterEqual(len(configs), 1)
        for cfg_path in configs:
            with open(cfg_path) as f:
                cfg = json.load(f)
            run_name = generate_run_name(cfg, seed=42)
            parts = run_name.split("-")
            self.assertGreaterEqual(
                len(parts),
                5,
                f"Run name {run_name} must have at least 5 hyphen-separated tokens",
            )
            self.assertIn(parts[1], {"pcu", "standard_aug", "compose_aug", "noaug",
                                     "openpcdet_aug", "openpcdet_gt_aug", "hybrid_gt_aug"})
            self.assertTrue(parts[2].endswith("_loss"))
            self.assertIn(parts[3], {"rich8", "rich10", "rich11", "rich12", "legacy35"})
            self.assertIn(parts[4], {"baseline_iou", "iqa"})

    def test_trainer_has_no_tqdm_overhead(self):
        trainer_code = (
            REPO_ROOT / "tools" / "kitti_training_pipeline" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("tqdm", trainer_code)
        self.assertNotIn("--log-interval", trainer_code)

    def test_trainer_writes_clean_epoch_summary_to_train_log(self):
        trainer_code = (
            REPO_ROOT / "tools" / "kitti_training_pipeline" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertIn('train_log_path = run_dir / "train.log"', trainer_code)
        self.assertIn("log_line(epoch_summary)", trainer_code)
        self.assertIn("Train Loss: {train_objective:.4f}", trainer_code)
        self.assertIn("Time: train={training_seconds:.1f}s, val=", trainer_code)
        self.assertNotIn("objective.detach().item()", trainer_code)


if __name__ == "__main__":
    unittest.main()
