"""Contract checks for the shared Colab training notebook."""

import json
import contextlib
import io
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"


class StandardTrainingNotebookTests(unittest.TestCase):
    def test_repository_keeps_only_the_standard_notebook(self):
        expected = {"3D_Lidar_Object_Detection_Notebook_standard.ipynb"}
        notebooks = set(subprocess.check_output(
            ["git", "ls-files", "*.ipynb"], cwd=ROOT, text=True,
        ).splitlines())
        self.assertEqual(notebooks, expected)

    def test_supports_clean_backbone_variants_and_safe_resume(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

        self.assertIn('BRANCH = "decoupled-c4-c5-routing"', source)
        self.assertIn("refs/remotes/origin/{BRANCH}", source)
        self.assertIn('CONFIG_OVERRIDE = None', source)
        self.assertIn("model.scale_gated_fpn", source)
        config_files = {
            path.relative_to(ROOT).as_posix()
            for path in (ROOT / "configs").rglob("*.json")
        }
        variants = (
            "B0", "B1_C2PSA", "C4_LSK", "C4_LITEMLA",
            "RICH8_SGFPN_CONTROL", "C4_LITEMLA_LATERAL_ONLY",
            "C4_DAT_LATERAL_ONLY", "C4_BRA_LATERAL_ONLY",
            "C4_LITEMLA_C5_C2PSA_SHARED",
            "C4_LITEMLA_C5_C2PSA_DECOUPLED",
        )
        for variant in variants:
            match = re.search(rf'"{variant}": "([^"]+\.json)"', source)
            self.assertIsNotNone(match, variant)
            self.assertIn(match.group(1), config_files)
        selected_variant = re.search(r'^VARIANT = "([A-Z0-9_]+)"', source, flags=re.MULTILINE)
        self.assertIsNotNone(selected_variant)
        self.assertIn(selected_variant.group(1), variants)
        self.assertIn('PRECISION = "auto"', source)
        self.assertIn("--clean-backbone", source)
        self.assertIn("tests/test_c4_attention.py", source)
        self.assertIn("resolve_ablation_config(", source)
        self.assertIn("write_json(CONFIG, resolved_config)", source)
        self.assertIn('resolved_config["train"]["physical_batch_size"]', source)
        self.assertIn('resolved_config["train"]["accumulation_steps"]', source)
        self.assertIn("BEV_ENCODING_OVERRIDE", source)
        self.assertIn("SCALE_GATED_FPN_OVERRIDE", source)
        self.assertIn("C5_ATTENTION_OVERRIDE", source)
        self.assertIn("C4_ATTENTION_ROUTE_OVERRIDE", source)
        self.assertIn("header_use_bn", source)
        self.assertIn("header_act", source)
        self.assertNotIn("kitti_uwag_coordatt_aug.json", source)
        self.assertNotIn("configs/kitti/mobilebev/", source)
        self.assertIn("run.json", source)
        self.assertIn("--resume", source)
        self.assertIn("last.pt", source)
        self.assertIn("checkpoint_epoch", source)
        self.assertNotIn("import subprocess", source)
        self.assertNotIn("subprocess.", source)
        self.assertIn("!set -o pipefail", source)

    def test_acceleration_options_are_explicit_and_smoke_validated(self):
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
        self.assertIn("SMOKE_METRICS", source)
        self.assertIn('math.isfinite(smoke_row["train_objective"])', source)
        self.assertIn('math.isfinite(smoke_row["validation"]["loss"])', source)
        self.assertIn('smoke_row["optimizer_updates"] < 1', source)

    def test_committed_b0_and_c2psa_profiles_are_explicit(self):
        baseline = json.loads(
            (
                ROOT
                / "configs/kitti/backbone_branch/kitti_mobilepixor_baseline.json"
            ).read_text(encoding="utf-8")
        )
        c2psa = json.loads(
            (
                ROOT
                / "configs/kitti/backbone_branch/kitti_mobilepixor_c2psa.json"
            ).read_text(encoding="utf-8")
        )
        for key in set(baseline) - {"data", "model", "note"}:
            self.assertEqual(c2psa[key], baseline[key], key)
        for key in set(baseline["data"]) - {"bev_encoding"}:
            self.assertEqual(c2psa["data"][key], baseline["data"][key], key)
        self.assertEqual(baseline["model"]["c5_attention"], "none")
        self.assertEqual(c2psa["model"]["c5_attention"], "c2psa")
        self.assertEqual(baseline["model"]["c4_attention"], "none")
        self.assertEqual(c2psa["model"]["c4_attention"], "none")
        self.assertIs(baseline["model"]["scale_gated_fpn"], False)
        self.assertIs(c2psa["model"]["scale_gated_fpn"], True)
        self.assertEqual(c2psa["model"]["backbone"], "mobilepixor")
        self.assertEqual(c2psa["loss"]["name"], "baseline")
        self.assertEqual(c2psa["augmentation"]["p"], 0.5)
        expected_baseline_encoding = {
            "density_norm": 32,
            "intensity_scale": 1,
            "name": "binary_slices",
        }
        expected_c2psa_encoding = {
            "density_norm": 32,
            "intensity_scale": 1,
            "name": "rich8",
        }
        self.assertEqual(baseline["data"]["bev_encoding"], expected_baseline_encoding)
        self.assertEqual(c2psa["data"]["bev_encoding"], expected_c2psa_encoding)
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
            "configs/kitti/backbone_branch/kitti_mobilepixor_baseline.json",
            source,
        )
        self.assertIn("trainable_parameter_count", source)
        self.assertIn("del baseline_model, selected_model", source)
        compile(source, str(NOTEBOOK), "exec")

    def test_trainer_writes_a_last_checkpoint_every_epoch(self):
        source = (ROOT / "tools/kitti_training_pipeline/train.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('atomic_torch_save(payload, checkpoints_dir / "last.pt")', source)

    def test_selector_and_architecture_cells_execute_for_all_presets(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
        selector = next(source for source in sources if "write_json(CONFIG, resolved_config)" in source)
        # Colab-only directory/install magics are not part of selector logic.
        selector = "\n".join(line for line in selector.splitlines() if not line.startswith("%"))
        architecture = next(source for source in sources if "Selected architecture:" in source)
        sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
        profiles = {
            "B0": (598073, "rich8", True, -7776 + 480),
            "B1_C2PSA": (626825, "binary_slices", False, 7776 - 480),
            "C4_LSK": (610623, "binary_slices", False, 7776 - 480),
            "C4_LITEMLA": (619321, "binary_slices", False, 7776 - 480),
            "RICH8_SGFPN_CONTROL": (590777, "binary_slices", False, 7776 - 480),
            "C4_LITEMLA_LATERAL_ONLY": (619321, "binary_slices", False, 7776 - 480),
            "C4_DAT_LATERAL_ONLY": (608713, "binary_slices", False, 7776 - 480),
            "C4_BRA_LATERAL_ONLY": (608249, "binary_slices", False, 7776 - 480),
            "C4_LITEMLA_C5_C2PSA_SHARED": (655369, "binary_slices", False, 7776 - 480),
            "C4_LITEMLA_C5_C2PSA_DECOUPLED": (655369, "binary_slices", False, 7776 - 480),
        }
        run_names = set()
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory).resolve()
            shutil.copytree(ROOT / "configs/kitti/backbone_branch", temporary_root / "configs/kitti/backbone_branch")
            for variant, (count, alternate_encoding, alternate_gated, alternate_delta) in profiles.items():
                scenarios = (
                    (None, None, 0),
                    (alternate_encoding, alternate_gated, alternate_delta),
                )
                for encoding, gated, delta in scenarios:
                    with self.subTest(variant=variant, encoding=encoding, gated=gated):
                        scope = {
                            "Path": Path, "REPO_DIR": temporary_root,
                            "CONFIG_OVERRIDE": None, "VARIANT": variant,
                            "BRANCH": "C2PSA_c5block", "BEV_ENCODING_OVERRIDE": encoding,
                            "SCALE_GATED_FPN_OVERRIDE": gated, "C5_ATTENTION_OVERRIDE": None,
                            "C4_ATTENTION_ROUTE_OVERRIDE": None,
                            "PHYSICAL_BATCH_SIZE": 2, "ACCUMULATION_STEPS": 4,
                            "EPOCHS": 100, "PRECISION": "bf16",
                            "SEED": 42, "RUN_NAME": "",
                            "RUNTIME_PROFILE": "numba_eager",
                            "ARTIFACT_ROOT": temporary_root / "artifacts",
                        }
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            exec(compile(selector, "notebook-selector", "exec"), scope)
                            self.assertTrue(scope["CONFIG"].is_file())
                            self.assertEqual(json.loads(scope["CONFIG"].read_text()), scope["resolved_config"])
                            # Use real detector sources and the temporary resolved config.
                            scope["REPO_DIR"] = ROOT
                            exec(compile(architecture, "notebook-architecture", "exec"), scope)
                        self.assertEqual(scope["baseline_parameters"], 598073)
                        self.assertEqual(scope["selected_parameters"], count + delta)
                        self.assertIn("Differences from baseline:", output.getvalue())
                        run_names.add(scope["RUN_NAME"])
        self.assertEqual(len(run_names), 20)

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
