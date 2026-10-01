import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from common import generate_run_name


class TestRunNameAndLogging(unittest.TestCase):
    def test_run_name_format_pcu(self):
        cfg_path = REPO_ROOT / "configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixornext-pcu-oga_loss-rich8-baseline_iou-sgfpn-s42")

    def test_run_name_format_cumulative(self):
        cfg_path = REPO_ROOT / "configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixornext-pcu-oga_loss-rich8-iqa-sgfpn-reparam-s42")

    def test_run_name_format_iqa(self):
        cfg_path = REPO_ROOT / "configs/kitti/iou_aware_header/kitti_mobilepixornext_litemla_oga_reparam_iqa.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixornext-standard_aug-oga_loss-rich8-iqa-sgfpn-reparam-s42")

    def test_run_name_format_baseline_legacy35(self):
        cfg_path = REPO_ROOT / "configs/kitti/baselines/kitti_mobilepixor_baseline.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixor-noaug-baseline_loss-legacy35-baseline_iou-s42")

    def test_run_name_format_baseline_a4(self):
        cfg_path = REPO_ROOT / "configs/kitti/mobilebev/a4_rich8_sgfpn_bev.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixor_coordatt-standard_aug-uwag_loss-rich8-baseline_iou-sgfpn-s42")

    def test_run_name_format_mobilepixor_ablation(self):
        cfg_path = REPO_ROOT / "configs/kitti/mobilepixor_ablation/00_mobilepixor_baseline.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixor-standard_aug-baseline_loss-rich8-baseline_iou-s42")

    def test_run_name_format_mobilepixor_pcu(self):
        cfg_path = REPO_ROOT / "configs/kitti/mobilepixor_ablation/05_mobilepixor_full_pcu.json"
        with open(cfg_path) as f:
            cfg = json.load(f)
        run_name = generate_run_name(cfg, seed=42)
        self.assertEqual(run_name, "mobilepixor-pcu-oga_loss-rich8-iqa-sgfpn-s42")

    def test_all_configs_generate_valid_run_name_format(self):
        configs = list((REPO_ROOT / "configs").rglob("*.json"))
        self.assertGreater(len(configs), 5)
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
            self.assertIn(parts[1], {"pcu", "standard_aug", "noaug"})
            self.assertTrue(parts[2].endswith("_loss"))
            self.assertIn(parts[3], {"rich8", "legacy35"})
            self.assertIn(parts[4], {"baseline_iou", "iqa"})

    def test_trainer_contains_tqdm(self):
        trainer_code = (
            REPO_ROOT / "tools" / "kitti_training_pipeline" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tqdm", trainer_code)

    def test_trainer_has_log_interval_argument(self):
        trainer_code = (
            REPO_ROOT / "tools" / "kitti_training_pipeline" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertIn("--log-interval", trainer_code)


if __name__ == "__main__":
    unittest.main()
