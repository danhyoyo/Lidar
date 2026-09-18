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

        self.assertIn('BRANCH = "A56_proposal_2.5"', source)
        self.assertIn("refs/remotes/origin/{BRANCH}", source)
        self.assertIn('CONFIG_OVERRIDE = None', source)
        branch_files = set(subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "A56_proposal_2.5", "--", "configs"],
            cwd=ROOT,
            text=True,
        ).splitlines())
        for variant in (*[f"A{i}" for i in range(7)], *[f"B{i}" for i in range(4)]):
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertIn(match.group(1), branch_files)
        self.assertIn("run.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertIn("!set -o pipefail", source)

    def test_trainer_writes_a_last_checkpoint_every_epoch(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('atomic_torch_save(payload, checkpoints_dir / "last.pt")', source)


if __name__ == "__main__":
    unittest.main()
