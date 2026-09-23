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

    def test_supports_all_registered_variants_and_safe_resume(self):
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
        for variant in (f"A{i}" for i in range(5)):
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertIn(match.group(1), config_files)
        self.assertIn("run.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertIn("!set -o pipefail", source)

    def test_acceleration_options_are_explicit_and_resume_safe(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('TARGET_BACKEND = "numba"', source)
        self.assertIn("COMPILE_MODEL = False", source)
        self.assertIn('COMPILE_MODEL_ARGUMENT = "--compile-model"', source)
        self.assertGreaterEqual(source.count('--target-backend "{TARGET_BACKEND}"'), 2)
        self.assertGreaterEqual(source.count("{COMPILE_MODEL_ARGUMENT}"), 2)
        self.assertIn('"target_backend": TARGET_BACKEND', source)
        self.assertIn('"compile_model": COMPILE_MODEL', source)
        self.assertIn('"num_workers": NUM_WORKERS', source)

    def test_smoke_test_validates_the_selected_runtime_profile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("RUNTIME_PROFILE =", source)
        self.assertIn("SMOKE_RUN_NAME =", source)
        self.assertIn("SMOKE_METRICS =", source)
        self.assertIn("SMOKE_CHECKPOINT =", source)
        self.assertIn('math.isfinite(smoke_row["train_objective"])', source)
        self.assertIn('math.isfinite(smoke_row["validation"]["loss"])', source)
        self.assertIn('smoke_row["optimizer_updates"] < 1', source)

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

    def test_training_keeps_reproducible_cudnn_and_one_loss_finite_guard(self):
        trainer = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        loss = (ROOT / "detector/core/losses/loss_fn.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("torch.backends.cudnn.benchmark = False", trainer)
        self.assertEqual(
            trainer.count("if not torch.isfinite")
            + loss.count("if not torch.isfinite"),
            1,
        )


if __name__ == "__main__":
    unittest.main()
