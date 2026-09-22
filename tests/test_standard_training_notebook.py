"""Contract checks for the shared Colab training notebook."""

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"


class StandardTrainingNotebookTests(unittest.TestCase):
    def test_repository_keeps_only_the_standard_notebook(self):
        expected = {"3D_Lidar_Object_Detection_Notebook_standard.ipynb"}
        notebooks = {
            path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*.ipynb")
        }
        self.assertEqual(notebooks, expected)

    def test_supports_clean_backbone_variants_and_safe_resume(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('BRANCH = "main"', source)
        self.assertIn("refs/remotes/origin/{BRANCH}", source)
        self.assertIn('CONFIG_OVERRIDE = None', source)
        config_files = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs").rglob("*.json")
        }
        for variant in ("B0", "B1_C2PSA"):
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertIn(match.group(1), config_files)
        self.assertIn('VARIANT = "B0"', source)
        self.assertIn('PRECISION = "auto"', source)
        self.assertIn("--clean-backbone", source)
        self.assertNotIn("kitti_uwag_coordatt_aug.json", source)
        self.assertNotIn("configs/kitti/mobilebev/", source)
        self.assertIn("run.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertIn("!set -o pipefail", source)

    def test_c2psa_config_changes_only_the_backbone_attention(self):
        baseline = json.loads(
            (ROOT / "configs/kitti/b0_b1/kitti_mobilepixor_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        c2psa = json.loads(
            (ROOT / "configs/kitti/b0_b1/kitti_mobilepixor_c2psa.json").read_text(
                encoding="utf-8"
            )
        )
        for key in set(baseline) - {"model", "note"}:
            self.assertEqual(c2psa[key], baseline[key], key)
        self.assertEqual(baseline["model"]["c5_attention"], "none")
        self.assertEqual(c2psa["model"]["c5_attention"], "c2psa")
        self.assertEqual(c2psa["model"]["backbone"], "mobilepixor")
        self.assertEqual(c2psa["loss"]["name"], "baseline")
        self.assertEqual(c2psa["augmentation"]["p"], 0.5)
        for transform in ("rotation", "scaling", "translation"):
            self.assertTrue(c2psa["augmentation"][transform]["use"])

    def test_architecture_comparison_runs_before_training(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        architecture_index = next(
            index
            for index, cell in enumerate(cells)
            if "Selected architecture:" in "".join(cell.get("source", []))
        )
        smoke_index = next(
            index
            for index, cell in enumerate(cells)
            if "## Smoke test" in "".join(cell.get("source", []))
        )
        source = "".join(cells[architecture_index]["source"])

        self.assertLess(architecture_index, smoke_index)
        self.assertIn("print(selected_model)", source)
        self.assertIn("Differences from baseline:", source)
        self.assertIn(
            "configs/kitti/b0_b1/kitti_mobilepixor_baseline.json", source
        )
        self.assertIn("trainable_parameter_count", source)
        self.assertIn("del baseline_model, selected_model", source)
        compile(source, str(NOTEBOOK), "exec")

    def test_trainer_writes_a_last_checkpoint_every_epoch(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('atomic_torch_save(payload, checkpoints_dir / "last.pt")', source)

    def test_fp16_grad_scaler_can_recover_from_a_scaled_gradient_overflow(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(
            "if not gradients_are_finite(model_parameters + criterion_parameters):",
            source,
        )

    def test_trainer_records_each_epoch_in_csv_and_tensorboard(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('csv_path = run_dir / "metrics.csv"', source)
        self.assertIn('tensorboard_dir = run_dir / "tensorboard"', source)
        self.assertIn("append_csv_row(csv_path, csv_row)", source)
        self.assertIn('writer.add_scalar(f"train/{name}"', source)
        self.assertIn('writer.add_scalar(f"validation/{name}"', source)


if __name__ == "__main__":
    unittest.main()
