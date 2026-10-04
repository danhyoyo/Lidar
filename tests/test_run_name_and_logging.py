import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools" / "kitti_training_pipeline"))

from common import generate_run_name


class TestRunNameAndLogging(unittest.TestCase):
    def test_augmentation_trials_have_unique_stable_names(self):
        directory = REPO_ROOT / "configs/kitti/augmentation"
        names = set()
        for path in sorted(directory.glob("*.json")):
            cfg = json.loads(path.read_text())
            name = generate_run_name(cfg, seed=42)
            augmentation = "standard_aug" if path.stem == "a_standard" else "pcu"
            self.assertEqual(name, f"mobilepixornext-{augmentation}-baseline_loss-rich8-baseline_iou-sgfpn-{path.stem}-s42")
            names.add(name)
        self.assertEqual(len(names), 5)

    def test_legacy_generic_name_is_preserved_without_experiment_tag(self):
        cfg = {"model": {"backbone": "mobilepixor", "header_use_iou": True, "use_reparam": True},
               "loss": {"name": "oga"}, "augmentation": {"use_pcu_aug": True},
               "data": {"bev_encoding": {"name": "rich8"}}}
        self.assertEqual(generate_run_name(cfg, seed=42), "mobilepixor-pcu-oga_loss-rich8-iqa-reparam-s42")

    def test_experiment_tag_cannot_escape_the_output_directory(self):
        with self.assertRaises(ValueError):
            generate_run_name({"experiment": {"name": "../overwrite"}}, seed=42)

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
