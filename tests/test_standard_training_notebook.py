"""Contract checks for the shared Colab training notebook."""

import json
import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"


class StandardTrainingNotebookTests(unittest.TestCase):
    def test_all_training_branches_keep_only_the_standard_notebook(self):
        expected = {"3D_Lidar_Object_Detection_Notebook_standard.ipynb"}
        for branch in (
            "main", "mobileBEV-architecture", "gaussian-lidar-detection",
            "Proposal2-Loss-Function", "A56_proposal_2.5",
        ):
            notebooks = set(subprocess.check_output(
                ["git", "ls-tree", "-r", "--name-only", branch],
                cwd=ROOT,
                text=True,
            ).splitlines())
            self.assertEqual(
                {path for path in notebooks if path.endswith(".ipynb")}, expected,
                branch,
            )

    def test_supports_all_registered_variants_and_safe_resume(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('BRANCH = "integrated_P12"', source)
        self.assertIn('PHYSICAL_BATCH_SIZE = 16', source)
        self.assertIn('ACCUMULATION_STEPS = 1', source)
        self.assertIn("refs/remotes/origin/{BRANCH}", source)
        self.assertIn('CONFIG_OVERRIDE = None', source)
        for variant in (
            *[f"A{i}" for i in range(5)], *[f"B{i}" for i in range(4)],
            *[f"C{i}" for i in range(3)],
        ):
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertTrue((ROOT / match.group(1)).is_file(), match.group(1))
        self.assertIn("run.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertIn("!set -o pipefail", source)
        self.assertIn('selection_split = REPO_DIR / CONFIG_DATA["val"]["data"]', source)
        self.assertIn('predict_log_variance', source)
        self.assertNotIn('VARIANT.startswith("B")', source)
        self.assertNotIn('VARIANT in {"B2", "B3"}', source)
        self.assertIn('"epochs": EPOCHS', source)
        self.assertIn('RESOLVED_CONFIG = RUN_DIR / "config.resolved.json"', source)
        self.assertIn('--config "{RESOLVED_CONFIG}"', source)

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


if __name__ == "__main__":
    unittest.main()
