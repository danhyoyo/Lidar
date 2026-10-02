import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "detector"))
sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

import torch
from common import build_model, generate_run_name, input_shape
from core.losses.loss_fn import LossFunction


class TestMobilePixorAblationConfigs(unittest.TestCase):
    CONFIG_DIR = REPO_ROOT / "configs" / "kitti" / "mobilepixor_ablation"
    EXPECTED_CONFIGS = [
        "00_mobilepixor_baseline.json",
        "01_mobilepixor_oga.json",
        "01_mobilepixor_q_oga.json",
        "01_mobilepixor_gw_qal.json",
        "02_mobilepixor_oga_sgfpn.json",
        "03_mobilepixor_oga_sgfpn_modernhead.json",
        "03_mobilepixor_oga_sgfpn_iqa.json",
        "04_mobilepixor_oga_sgfpn_iqa.json",
        "04_mobilepixor_oga_sgfpn_iqa_rich8.json",
    ]

    def test_all_expected_configs_exist(self):
        for name in self.EXPECTED_CONFIGS:
            path = self.CONFIG_DIR / name
            self.assertTrue(path.is_file(), f"Missing config: {path}")

    def test_configs_instantiate_model_and_criterion(self):
        for name in self.EXPECTED_CONFIGS:
            path = self.CONFIG_DIR / name
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            model = build_model(cfg)
            criterion = LossFunction(
                cfg["model"]["cls_encoding"], cfg.get("loss")
            )
            shape = input_shape(cfg)
            dummy_in = torch.randn(shape)
            out = model(dummy_in)
            self.assertIn("cls", out)

            # Verify run name format is valid
            run_name = generate_run_name(cfg, seed=42)
            self.assertTrue(run_name.startswith("mobilepixor-"))


if __name__ == "__main__":
    unittest.main()
