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
from common import build_model, generate_run_name, input_shape, create_experiment_config
from core.losses.loss_fn import LossFunction


class TestMobilePixorAblationConfigs(unittest.TestCase):
    MASTER_CONFIG_PATH = REPO_ROOT / "configs" / "config.json"
    ABLATION_OVERRIDES = {
        "00_mobilepixor_baseline": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": False, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
            "loss": {"name": "baseline"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "01_mobilepixor_oga": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": False, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
            "loss": {"name": "oga"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "01_mobilepixor_q_oga": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": False, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
            "loss": {"name": "q_oga"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "01_mobilepixor_gw_qal": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": False, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
            "loss": {"name": "gw_qal"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "02_mobilepixor_oga_sgfpn": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": True, "c4_attention": "none", "header_use_bn": False, "header_act": "relu"},
            "loss": {"name": "oga"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "03_mobilepixor_oga_sgfpn_modernhead": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": True, "c4_attention": "none", "header_use_bn": True, "header_act": "silu"},
            "loss": {"name": "oga"},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "03_mobilepixor_oga_sgfpn_iqa": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": True, "c4_attention": "none", "header_use_bn": True, "header_act": "silu", "header_use_iou": True},
            "loss": {"name": "oga", "use_iou": True},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
        "04_mobilepixor_oga_sgfpn_iqa_rich8": {
            "model": {"backbone": "mobilepixor", "scale_gated_fpn": True, "c4_attention": "none", "header_use_bn": True, "header_act": "silu", "header_use_iou": True},
            "loss": {"name": "oga", "use_iou": True},
            "data": {"bev_encoding": {"name": "rich8"}},
        },
    }

    def test_master_config_exists(self):
        self.assertTrue(self.MASTER_CONFIG_PATH.is_file(), f"Missing config: {self.MASTER_CONFIG_PATH}")

    def test_configs_instantiate_model_and_criterion(self):
        with open(self.MASTER_CONFIG_PATH, "r", encoding="utf-8") as f:
            base_cfg = json.load(f)

        for name, overrides in self.ABLATION_OVERRIDES.items():
            cfg = create_experiment_config(base_cfg, overrides)
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
            self.assertTrue(run_name.startswith("mobilepixor-"), f"Unexpected run_name: {run_name}")


if __name__ == "__main__":
    unittest.main()
