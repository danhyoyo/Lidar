import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from train import build_parser


class TestOverrideJsonCLI(unittest.TestCase):
    def test_parser_accepts_override_json(self):
        parser = build_parser()
        args = parser.parse_args([
            "--config", "configs/kitti/oga_loss/kitti_mobilepixornext_litemla_oga.json",
            "--detector-root", "detector",
            "--output-root", "/tmp",
            "--override-json", '{"loss": {"name": "gw_qal", "beta_q": 1.0}}',
        ])
        self.assertEqual(args.override_json, '{"loss": {"name": "gw_qal", "beta_q": 1.0}}')
        parsed = json.loads(args.override_json)
        self.assertEqual(parsed["loss"]["name"], "gw_qal")
        self.assertEqual(parsed["loss"]["beta_q"], 1.0)


if __name__ == "__main__":
    unittest.main()
