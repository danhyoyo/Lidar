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

    def test_supports_b_series_variants_and_trainer_safe_resume(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('BRANCH = "feature/loss-function"', source)
        self.assertIn("refs/remotes/origin/{BRANCH}", source)
        self.assertIn('VARIANT = "B5"', source)
        self.assertIn('CONFIG_OVERRIDE = None', source)
        config_files = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs").rglob("*.json")
        }
        expected_configs = {
            f"B{i}": f"configs/kitti/loss_ablation/{name}.json"
            for i, name in enumerate(
                (
                    "b0_a4_legacy_uwag",
                    "b1_l1_fixed",
                    "b2_kfiou_fixed",
                    "b3_probiou_fixed",
                    "b4_kld_fixed",
                    "b5_mgiou_fixed",
                )
            )
        }
        for variant, expected_path in expected_configs.items():
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertEqual(match.group(1), expected_path)
            self.assertIn(match.group(1), config_files)
        self.assertIn("CONFIG_JSON", source)
        self.assertIn("selection_policy", source)
        self.assertIn('"best": "best.pt"', source)
        self.assertIn('"final_epoch": "final.pt"', source)
        self.assertIn("SELECTED_CHECKPOINT_FILENAME", source)
        self.assertIn('CHECKPOINT = RUN_DIR / "selected" / SELECTED_CHECKPOINT_FILENAME', source)
        self.assertIn("run.manifest.json", source)
        self.assertIn("config.resolved.json", source)
        self.assertIn(".sha256.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertNotIn("run.json", source)
        self.assertNotIn("RUN_METADATA", source)
        self.assertNotIn("ALLOW_LEGACY_RESUME", source)
        self.assertNotIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertNotIn("test_mobile_bev.py", source)
        self.assertIn("test_loss_ablation.py", source)
        self.assertIn("test_standard_training_notebook.py", source)
        self.assertIn("!set -o pipefail", source)

    def test_verification_uses_repo_root_discover_commands_with_fail_fast_chain(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        expected = (
            "MPLCONFIGDIR=/tmp/lidar-mpl PYTHONPATH=. python3 -m unittest "
            "discover -s tests -p 'test_loss_ablation.py' && "
            "MPLCONFIGDIR=/tmp/lidar-mpl PYTHONPATH=. python3 -m unittest "
            "discover -s tests -p 'test_standard_training_notebook.py'"
        )
        self.assertIn(expected, source)
        self.assertNotIn(
            "python3 -m unittest tests/test_loss_ablation.py "
            "tests/test_standard_training_notebook.py",
            source,
        )

    def test_notebook_defaults_are_sensible_for_a_colab_l4_b5_smoke(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('PRECISION = "bf16"', source)
        self.assertIn("PHYSICAL_BATCH_SIZE = 2", source)
        self.assertIn("ACCUMULATION_STEPS = 2", source)
        self.assertIn("EPOCHS = 100", source)
        self.assertIn("must remain unchanged when resuming", source)

    def test_smoke_resume_reuses_trainer_provenance_and_runtime_controls(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn("SMOKE_RUN_DIR", source)
        self.assertIn("SMOKE_RESOLVED_CONFIG_PATH", source)
        self.assertIn("SMOKE_MANIFEST_PATH", source)
        self.assertIn("SMOKE_RESUME_ARGUMENT", source)
        self.assertIn("smoke_last_checkpoint.with_suffix", source)
        self.assertIn("--max-train-batches 2 --max-val-batches 1 --num-workers 0", source)
        self.assertNotIn("SMOKE_RUN_DIR.mkdir", source)

    def test_notebook_json_has_no_stale_colab_outputs(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            with self.subTest(cell=index):
                self.assertEqual(cell.get("outputs", []), [])
                if cell.get("cell_type") == "code":
                    self.assertIsNone(cell.get("execution_count"))

    def test_trainer_writes_a_last_checkpoint_every_epoch(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'save_checkpoint_with_sha256(payload, checkpoints_dir / "last.pt")',
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
        self.assertIn("writer.close()", source)

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
