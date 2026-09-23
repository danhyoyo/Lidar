#!/usr/bin/env python3
"""Config invariants for the independent B-series loss ablation."""

import copy
import importlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
# The detector package's runtime code imports ``core.*`` because training is
# launched from ``detector/``.  Make the repository-root unittest command use
# the same import root without requiring an external PYTHONPATH override.
sys.path.insert(0, str(ROOT / "detector"))
# ``focal_loss.py`` follows the training layout and imports the dataset helper
# package as a top-level ``utils_1`` module when launched from ``detector/``.
sys.path.insert(0, str(ROOT / "detector" / "core" / "datasets"))

from detector.core.losses import gaussian_geometry_loss
from detector.core.losses.gaussian_geometry_loss import box_covariance
from detector.core.losses.loss_fn import LossFunction
from core.losses import mgiou_loss as runtime_mgiou_module

try:
    from detector.core.losses import mgiou_loss as mgiou_module
except (ImportError, ModuleNotFoundError):
    mgiou_module = None

CONFIG_DIR = ROOT / "configs" / "kitti" / "loss_ablation"
A4_CONFIG = ROOT / "configs" / "kitti" / "mobilebev" / "a4_rich8_sgfpn_bev.json"
TRAIN_PIPELINE = ROOT / "tools" / "kitti_training_pipeline"


def load_training_module():
    """Load the training helpers using the same import root as the CLI."""
    path = str(TRAIN_PIPELINE)
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("train")


def require_training_helper(name):
    """Make a missing Task 7 helper fail with an actionable RED assertion."""
    helper = getattr(load_training_module(), name, None)
    if not callable(helper):
        raise AssertionError(
            f"train.py must expose the Task 7 helper {name}()"
        )
    return helper


EXPECTED = {
    "b0_a4_legacy_uwag": {
        "codename": "B0",
        "objective": "A4 snapshot + legacy UWAG + auxiliary 0.2",
        "note": "loss_ablation_B0_a4_legacy_uwag",
        "loss": {
            "epsilon": 0.0001,
            "geometric_weight": 0.2,
            "initial_log_scales": [0, 0, 0, 0],
            "max_abs_log_size": 10,
            "name": "uwag",
        },
    },
    "b1_l1_fixed": {
        "codename": "B1",
        "objective": "Masked-L1 fixed",
        "note": "loss_ablation_B1_l1_fixed",
        "loss": {
            "regression": "l1",
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 0.000001,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        },
    },
    "b2_kfiou_fixed": {
        "codename": "B2",
        "objective": "Full KFIoU fixed",
        "note": "loss_ablation_B2_kfiou_fixed",
        "loss": {
            "regression": "kfiou",
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 0.000001,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        },
    },
    "b3_probiou_fixed": {
        "codename": "B3",
        "objective": "ProbIoU BD/Hellinger fixed",
        "note": "loss_ablation_B3_probiou_fixed",
        "loss": {
            "regression": "probiou",
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 0.000001,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        },
    },
    "b4_kld_fixed": {
        "codename": "B4",
        "objective": "KLD-log fixed",
        "note": "loss_ablation_B4_kld_fixed",
        "loss": {
            "regression": "kld",
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 0.000001,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        },
    },
    "b5_mgiou_fixed": {
        "codename": "B5",
        "objective": "MGIoU fixed",
        "note": "loss_ablation_B5_mgiou_fixed_candidate",
        "loss": {
            "regression": "mgiou",
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 0.000001,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        },
    },
}


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def without_loss_and_note(config):
    snapshot = copy.deepcopy(config)
    snapshot.pop("loss", None)
    snapshot.pop("note", None)
    return snapshot


def without_selection_policy(config):
    snapshot = copy.deepcopy(config)
    snapshot.get("train", {}).pop("selection_policy", None)
    return snapshot


class LossAblationConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = {
            stem: CONFIG_DIR / f"{stem}.json"
            for stem in EXPECTED
        }
        cls.configs = (
            {stem: read_json(path) for stem, path in cls.paths.items()}
            if all(path.is_file() for path in cls.paths.values())
            else None
        )

    def _require_configs(self):
        if self.configs is None:
            self.skipTest("B-series config snapshots have not been created")

    def test_all_expected_config_files_exist(self):
        missing = [str(path) for path in self.paths.values() if not path.is_file()]
        self.assertEqual(missing, [], f"missing B-series configs: {missing}")

    def test_configs_have_expected_codenames_objectives_and_loss_schema(self):
        self._require_configs()
        objectives = {
            "uwag": "A4 snapshot + legacy UWAG + auxiliary 0.2",
            "l1": "Masked-L1 fixed",
            "kfiou": "Full KFIoU fixed",
            "probiou": "ProbIoU BD/Hellinger fixed",
            "kld": "KLD-log fixed",
            "mgiou": "MGIoU fixed",
        }
        for stem, expected in EXPECTED.items():
            with self.subTest(stem=stem):
                config = self.configs[stem]
                self.assertEqual(stem.split("_", 1)[0].upper(), expected["codename"])
                self.assertEqual(config["note"], expected["note"])
                self.assertEqual(config["loss"], expected["loss"])
                self.assertIn(expected["codename"], config["note"])
                objective_key = config["loss"].get(
                    "name", config["loss"].get("regression")
                )
                self.assertEqual(objectives[objective_key], expected["objective"])

    def test_b0_b5_share_one_snapshot_outside_loss_and_note(self):
        self._require_configs()
        snapshots = [without_loss_and_note(config) for config in self.configs.values()]
        for snapshot in snapshots[1:]:
            self.assertEqual(snapshot, snapshots[0])

    def test_snapshot_matches_a4_reference(self):
        self._require_configs()
        expected = without_selection_policy(without_loss_and_note(read_json(A4_CONFIG)))
        for stem, config in self.configs.items():
            with self.subTest(stem=stem):
                actual = without_selection_policy(without_loss_and_note(config))
                self.assertEqual(actual, expected)

    def test_all_b_configs_use_final_epoch_selection(self):
        self._require_configs()
        for stem, config in self.configs.items():
            with self.subTest(stem=stem):
                self.assertEqual(config["train"]["selection_policy"], "final_epoch")


class LossFunctionIntegrationTests(unittest.TestCase):
    """Compatibility contract for legacy and fixed B-series objectives."""

    @staticmethod
    def _maps(offset=(0.25, -0.1)):
        target = {
            "cls": torch.zeros(1, 3, 2, 2),
            "offset": torch.zeros(1, 2, 2, 2),
            "size": torch.zeros(1, 2, 2, 2),
            "yaw": torch.zeros(1, 2, 2, 2),
            "reg_mask": torch.ones(1, 2, 2),
        }
        target["cls"][0, 1, 0, 0] = 1.0
        target["yaw"][:, 0] = 1.0
        pred = {key: value.clone() for key, value in target.items() if key != "reg_mask"}
        pred["offset"][:, 0, 0, 0] = offset[0]
        pred["offset"][:, 1, 0, 0] = offset[1]
        return pred, target

    @staticmethod
    def _fixed_config(regression, **overrides):
        config = {
            "regression": regression,
            "weighting": "fixed",
            "regression_weight": 1.0,
            "covariance_epsilon": 1e-6,
            "max_abs_log_size": 10.0,
            "probiou_switch_fraction": 0.5,
            "kld_tau": 1.0,
        }
        config.update(overrides)
        return config

    @staticmethod
    def _scalar(value):
        return float(value.detach().cpu()) if torch.is_tensor(value) else float(value)

    def test_legacy_baseline_is_cls_plus_three_l1_terms(self):
        pred, target = self._maps()
        result = LossFunction("gaussian", {"name": "baseline"})(pred, target)

        self.assertEqual(
            set(result), {"loss", "cls", "offset", "size", "yaw", "geo"}
        )
        expected = result["cls"] + result["offset"] + result["size"] + result["yaw"]
        self.assertAlmostEqual(self._scalar(result["loss"]), expected, places=6)
        self.assertEqual(result["geo"], 0.0)

    def test_legacy_uwag_keeps_four_scales_auxiliary_and_strict_state_roundtrip(self):
        pred, target = self._maps()
        config = {
            "name": "uwag",
            "epsilon": 1e-4,
            "geometric_weight": 0.2,
            "initial_log_scales": [0.0, 0.0, 0.0, 0.0],
            "max_abs_log_size": 10.0,
        }
        criterion = LossFunction("gaussian", config)
        result = criterion(pred, target)
        state = criterion.state_dict()
        restored = LossFunction("gaussian", config)
        restored.load_state_dict(state, strict=True)

        self.assertEqual(tuple(criterion.log_scales.shape), (4,))
        self.assertEqual(set(state), set(restored.state_dict()))
        self.assertTrue(torch.equal(criterion.log_scales, restored.log_scales))
        self.assertEqual(criterion.geometric_weight, 0.2)
        self.assertGreater(result["geo"], 0.0)
        self.assertEqual(
            {f"weight_{task}" for task in criterion.TASKS},
            set(result).intersection({f"weight_{task}" for task in criterion.TASKS}),
        )

    def test_b1_fixed_applies_regression_weight_to_three_l1_terms(self):
        pred, target = self._maps()
        result = LossFunction(
            "gaussian", self._fixed_config("l1", regression_weight=2.0)
        )(pred, target)

        self.assertEqual(
            set(result), {"loss", "cls", "offset", "size", "yaw", "geo"}
        )
        expected = result["cls"] + 2.0 * (
            result["offset"] + result["size"] + result["yaw"]
        )
        self.assertAlmostEqual(self._scalar(result["loss"]), expected, places=6)
        self.assertEqual(result["geo"], 0.0)

    def test_b2_to_b4_return_cls_plus_geometry_and_stable_output_keys(self):
        pred, target = self._maps(offset=(2.0, 0.0))
        required = {"loss", "cls", "offset", "size", "yaw", "geo"}
        for regression in ("kfiou", "probiou", "kld"):
            with self.subTest(regression=regression):
                criterion = LossFunction(
                    "gaussian", self._fixed_config(regression)
                )
                result = criterion(pred, target, progress=0.25)

                self.assertTrue(required.issubset(result))
                self.assertAlmostEqual(
                    self._scalar(result["loss"]),
                    result["cls"] + result["geo"],
                    places=6,
                )
                self.assertTrue(torch.isfinite(torch.as_tensor(result["loss"])).item())
                if "clamp_count" in result:
                    self.assertGreaterEqual(int(result["clamp_count"]), 0)

    def test_b3_progress_is_forwarded_and_default_is_zero(self):
        pred, target = self._maps(offset=(2.0, 0.0))
        criterion = LossFunction("gaussian", self._fixed_config("probiou"))

        default = criterion(pred, target)
        explicit_zero = criterion(pred, target, progress=0.0)
        explicit_late = criterion(pred, target, progress=1.0)

        self.assertAlmostEqual(default["geo"], explicit_zero["geo"], places=6)
        self.assertNotAlmostEqual(default["geo"], explicit_late["geo"], places=6)

    def test_invalid_regression_weighting_and_scalars_raise(self):
        invalid_configs = {
            "regression": self._fixed_config("not-a-loss"),
            "weighting": self._fixed_config("l1", weighting="uwag"),
            "regression_weight": self._fixed_config("l1", regression_weight=-1.0),
            "covariance_epsilon": self._fixed_config("kfiou", covariance_epsilon=0.0),
            "max_abs_log_size": self._fixed_config("kfiou", max_abs_log_size=-1.0),
            "probiou_switch_fraction": self._fixed_config(
                "probiou", probiou_switch_fraction=1.1
            ),
            "kld_tau": self._fixed_config("kld", kld_tau=0.5),
        }
        for label, config in invalid_configs.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    LossFunction("gaussian", config)

    def test_weighting_only_cannot_be_misread_as_legacy_schema(self):
        """A fixed-schema discriminator must require its regression selector."""
        with self.assertRaises(ValueError):
            LossFunction("gaussian", {"weighting": "fixed"})

    def test_b5_owns_mgiou_and_reports_finite_candidate_diagnostics(self):
        pred, target = self._maps(offset=(2.0, 0.0))
        criterion = LossFunction("gaussian", self._fixed_config("mgiou"))
        result = criterion(pred, target)

        self.assertIsNotNone(mgiou_module)
        self.assertIsInstance(
            criterion.geometry_criterion, runtime_mgiou_module.MGIoULoss
        )
        self.assertEqual(
            set(result),
            {
                "loss", "cls", "offset", "size", "yaw", "geo",
                "clamp_count", "yaw_fallback_count",
            },
        )
        self.assertAlmostEqual(
            self._scalar(result["loss"]), result["cls"] + result["geo"], places=6
        )
        self.assertTrue(torch.isfinite(torch.as_tensor(result["loss"])).item())
        self.assertEqual(result["yaw_fallback_count"], 0)

    def test_b5_empty_mask_and_zero_yaw_have_stable_diagnostics(self):
        pred, target = self._maps()
        target["reg_mask"].zero_()
        empty = LossFunction("gaussian", self._fixed_config("mgiou"))(pred, target)
        self.assertTrue(torch.isfinite(torch.as_tensor(empty["loss"])).item())
        self.assertEqual(empty["geo"], 0.0)
        self.assertEqual(empty["yaw_fallback_count"], 0)

        pred, target = self._maps()
        target["reg_mask"].zero_()
        target["reg_mask"][0, 0, 0] = 1.0
        pred["yaw"].zero_()
        fallback = LossFunction("gaussian", self._fixed_config("mgiou"))(pred, target)
        self.assertTrue(torch.isfinite(torch.as_tensor(fallback["loss"])).item())
        self.assertEqual(fallback["yaw_fallback_count"], 1)


class TrainingGuardContractTests(unittest.TestCase):
    """Small pure contracts for the Task 7 trainer/resume guard.

    These tests intentionally do not construct a dataset or run a training
    epoch.  They define the state-machine boundaries that the trainer must
    implement, leaving the CLI loop free to use its existing data pipeline.
    """

    def test_planned_attempts_include_a_partial_accumulation_group(self):
        planned = require_training_helper("planned_optimizer_update_attempts")

        self.assertEqual(planned(epochs=3, batches_per_epoch=5, accumulation_steps=2), 9)
        self.assertEqual(planned(epochs=1, batches_per_epoch=4, accumulation_steps=2), 2)
        self.assertEqual(planned(epochs=1, batches_per_epoch=1, accumulation_steps=8), 1)
        with self.assertRaises(ValueError):
            planned(epochs=0, batches_per_epoch=5, accumulation_steps=2)

    def test_progress_is_normalized_from_attempts_not_successes(self):
        progress = require_training_helper("normalized_training_progress")

        self.assertEqual(progress(global_update_attempts=0, total_planned_attempts=10), 0.0)
        self.assertAlmostEqual(
            progress(global_update_attempts=5, total_planned_attempts=10),
            0.5,
        )
        self.assertEqual(progress(global_update_attempts=10, total_planned_attempts=10), 1.0)
        # A skipped accumulation group still consumes an attempt and therefore
        # must not postpone the B3 BD -> Hellinger switch.
        self.assertAlmostEqual(
            progress(global_update_attempts=5, total_planned_attempts=10),
            0.5,
        )
        with self.assertRaises(ValueError):
            progress(global_update_attempts=1, total_planned_attempts=0)

    def test_gradient_finite_predicate_and_l2_norm_are_explicit(self):
        finite = require_training_helper("all_gradients_finite")
        norm = require_training_helper("gradient_l2_norm")

        parameter = nn.Parameter(torch.tensor([3.0, 4.0]))
        (parameter.square().sum()).backward()

        self.assertTrue(finite([parameter]))
        self.assertAlmostEqual(float(norm([parameter])), 10.0, places=6)

        parameter.grad[0] = float("inf")
        self.assertFalse(finite([parameter]))
        # Norm reporting must never silently turn an invalid gradient into a
        # finite zero; the guard decides whether this group may be stepped.
        self.assertFalse(math.isfinite(float(norm([parameter]))))

    def test_selection_policy_defaults_to_legacy_best_and_supports_final_epoch(self):
        filename = require_training_helper("selection_checkpoint_filename")

        self.assertEqual(filename({"train": {}}), "best.pt")
        self.assertEqual(
            filename({"train": {"selection_policy": "final_epoch"}}),
            "final.pt",
        )
        with self.assertRaises(ValueError):
            filename({"train": {"selection_policy": "unknown"}})

    def test_checkpoint_payload_round_trips_update_counters(self):
        training = load_training_module()

        class Stateful:
            def state_dict(self):
                return {}

        payload = training.checkpoint_payload(
            Stateful(), Stateful(), Stateful(), Stateful(), Stateful(),
            epoch=7,
            validation={"loss": 1.25},
            best_val=1.0,
            config={"train": {"selection_policy": "final_epoch"}},
            global_update_attempts=19,
            successful_updates=16,
            skipped_updates=3,
        )

        self.assertEqual(payload["global_update_attempts"], 19)
        self.assertEqual(payload["successful_updates"], 16)
        self.assertEqual(payload["skipped_updates"], 3)
        restored = copy.deepcopy(payload)
        self.assertEqual(restored["global_update_attempts"], 19)
        self.assertEqual(restored["successful_updates"], 16)
        self.assertEqual(restored["skipped_updates"], 3)

    def test_run_directory_policy_is_safe_for_new_runs_and_resume(self):
        validate = require_training_helper("validate_run_directory_policy")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            absent_run = root / "new-run"
            # A fresh run may create its selected output directory later.
            validate(absent_run, None)

            existing_run = root / "existing-run"
            existing_run.mkdir()
            with self.assertRaises(ValueError):
                validate(existing_run, None)

            checkpoint = existing_run / "checkpoints" / "last.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            validate(existing_run, checkpoint)

            outside_checkpoint = root / "outside.pt"
            outside_checkpoint.touch()
            with self.assertRaises(ValueError):
                validate(existing_run, outside_checkpoint)

    def test_epoch_success_guard_rejects_zero_successful_updates(self):
        require_successful_epoch = require_training_helper("require_successful_epoch")

        for epoch in (0, 1, 7):
            with self.subTest(epoch=epoch):
                with self.assertRaises(ValueError):
                    require_successful_epoch(epoch, 0)

        # A nonzero successful-update count is valid for both an initial and
        # a later epoch; the helper is a safety gate, not a scheduling policy.
        require_successful_epoch(0, 1)
        require_successful_epoch(7, 16)

    def test_checkpoint_sha256_sidecar_detects_tampering(self):
        training = load_training_module()
        verify = require_training_helper("verify_checkpoint_sha256")

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoints" / "last.pt"
            digest = training.save_checkpoint_with_sha256(
                {"epoch": 3, "weights": torch.tensor([1.0, 2.0])},
                checkpoint,
            )

            self.assertEqual(verify(checkpoint), digest)
            checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
            with self.assertRaises(ValueError):
                verify(checkpoint)

            sidecar = checkpoint.with_suffix(checkpoint.suffix + ".sha256.json")
            sidecar.unlink()
            with self.assertRaises(FileNotFoundError):
                verify(checkpoint)

    def test_runtime_controls_are_json_serializable_and_identity_defining(self):
        controls = require_training_helper("resolved_runtime_controls")

        base = controls(torch.device("cpu"), 2, 3, 4, "python", False)
        self.assertEqual(
            set(base),
            {
                "device", "num_workers", "max_train_batches", "max_val_batches",
                "target_backend", "compile_model",
            },
        )
        # The resolved object is persisted in provenance/resume state, so it
        # must not contain a torch.device or another non-JSON value.
        json.dumps(base)

        self.assertNotEqual(base, controls(torch.device("cpu"), 2, 4, 4, "python", False))
        self.assertNotEqual(base, controls(torch.device("cpu"), 2, 3, 5, "python", False))
        self.assertNotEqual(base, controls(torch.device("cuda"), 2, 3, 4, "python", False))
        self.assertNotEqual(base, controls(torch.device("cpu"), 3, 3, 4, "python", False))
        self.assertNotEqual(base, controls(torch.device("cpu"), 2, 3, 4, "numba", False))
        self.assertNotEqual(base, controls(torch.device("cpu"), 2, 3, 4, "python", True))

    def test_run_manifest_input_hashes_detect_changed_or_malformed_inputs(self):
        training = load_training_module()
        verify = require_training_helper("verify_run_manifest_inputs")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_config = root / "config.json"
            train_split = root / "train.txt"
            val_split = root / "val.txt"
            source_config.write_text('{"seed": 42}\n', encoding="utf-8")
            train_split.write_text("train-v1\n", encoding="utf-8")
            val_split.write_text("val-v1\n", encoding="utf-8")

            manifest = {
                "source_config": {
                    "path": str(source_config.resolve()),
                    "sha256": training.sha256(source_config),
                },
                "splits": {
                    "train": {
                        "path": str(train_split.resolve()),
                        "sha256": training.sha256(train_split),
                    },
                    "val": {
                        "path": str(val_split.resolve()),
                        "sha256": training.sha256(val_split),
                    },
                },
            }

            verify(manifest, source_config, train_split, val_split)

            train_split.write_text("train-v2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(manifest, source_config, train_split, val_split)
            train_split.write_text("train-v1\n", encoding="utf-8")

            val_split.write_text("val-v2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(manifest, source_config, train_split, val_split)
            val_split.write_text("val-v1\n", encoding="utf-8")

            source_config.write_text('{"seed": 43}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(manifest, source_config, train_split, val_split)
            source_config.write_text('{"seed": 42}\n', encoding="utf-8")

            malformed = copy.deepcopy(manifest)
            del malformed["splits"]["train"]["sha256"]
            with self.assertRaises(ValueError):
                verify(malformed, source_config, train_split, val_split)

    def test_prob_iou_switch_boundary_uses_attempt_progress_after_skips(self):
        progress = require_training_helper("normalized_training_progress")
        criterion = gaussian_geometry_loss.ProbIoULoss(
            epsilon=1e-6,
            max_abs_log_size=10.0,
            switch_fraction=0.5,
        )
        target_yaw = torch.zeros(1, 2, 1, 1)
        target_yaw[:, 0] = 1.0
        prediction_yaw = target_yaw.clone()
        target_size = torch.tensor(
            [math.log(2.0), math.log(4.0)]
        ).view(1, 2, 1, 1)
        prediction_size = target_size.clone()
        target_offset = torch.zeros(1, 2, 1, 1)
        prediction_offset = torch.tensor([[[[2.0]], [[0.0]]]])
        mask = torch.ones(1, 1, 1)

        early, _ = criterion(
            prediction_offset, prediction_size, prediction_yaw,
            target_offset, target_size, target_yaw, mask,
            progress=progress(4, 10),
        )
        late, _ = criterion(
            prediction_offset, prediction_size, prediction_yaw,
            target_offset, target_size, target_yaw, mask,
            progress=progress(5, 10),
        )
        self.assertNotAlmostEqual(float(early), float(late), places=6)

    def test_loss_diagnostic_counts_are_aggregated_without_sample_weighting(self):
        accumulate = require_training_helper("accumulate_loss_outputs")
        components = {}
        diagnostics = {}

        accumulate(
            components,
            diagnostics,
            {"loss": 2.0, "geo": 1.5, "yaw_fallback_count": 3, "clamp_count": 2},
            batch_size=4,
        )
        accumulate(
            components,
            diagnostics,
            {"loss": 5.0, "yaw_fallback_count": 1, "clamp_count": 7},
            batch_size=2,
        )

        self.assertEqual(components, {"loss": 18.0, "geo": 6.0})
        self.assertEqual(diagnostics, {"yaw_fallback_count": 4, "clamp_count": 9})


class GaussianCovarianceTests(unittest.TestCase):
    """RED contract for the shared Gaussian box covariance helper."""

    DIVISOR = 4.0
    EPSILON = 1e-6
    MAX_ABS_LOG_SIZE = 10.0

    def _covariance(self, log_size, doubled_yaw):
        return box_covariance(
            log_size,
            doubled_yaw,
            divisor=self.DIVISOR,
            epsilon=self.EPSILON,
            max_abs_log_size=self.MAX_ABS_LOG_SIZE,
        )

    def test_theta_zero_uses_length_on_x_axis(self):
        log_size = torch.tensor([[math.log(2.0), math.log(4.0)]])
        doubled_yaw = torch.tensor([[1.0, 0.0]])

        covariance, clamp_count = self._covariance(log_size, doubled_yaw)

        expected = torch.tensor([[[4.0, 0.0], [0.0, 1.0]]])
        self.assertTrue(torch.allclose(covariance, expected, atol=1e-5, rtol=1e-5))
        self.assertEqual(int(clamp_count), 0)

    def test_theta_and_theta_plus_pi_have_same_covariance(self):
        theta = 0.37
        doubled_yaw = torch.tensor(
            [[math.cos(2.0 * theta), math.sin(2.0 * theta)]]
        )
        doubled_yaw_plus_pi = torch.tensor(
            [[
                math.cos(2.0 * (theta + math.pi)),
                math.sin(2.0 * (theta + math.pi)),
            ]]
        )
        log_size = torch.tensor([[math.log(1.5), math.log(5.0)]])

        covariance, _ = self._covariance(log_size, doubled_yaw)
        covariance_plus_pi, _ = self._covariance(log_size, doubled_yaw_plus_pi)

        self.assertTrue(torch.allclose(covariance, covariance_plus_pi, atol=1e-6, rtol=1e-6))

    def test_near_square_and_thin_boxes_are_finite_positive_definite(self):
        log_size = torch.log(
            torch.tensor(
                [[2.0, 2.000001], [0.01, 12.0]],
                dtype=torch.float32,
            )
        )
        doubled_yaw = torch.tensor([[0.6, 0.8], [0.0, 1.0]], dtype=torch.float32)

        covariance, _ = self._covariance(log_size, doubled_yaw)
        eigenvalues = torch.linalg.eigvalsh(covariance)

        self.assertTrue(torch.isfinite(covariance).all().item())
        self.assertTrue((eigenvalues > 0).all().item())

    def test_bfloat16_inputs_are_promoted_to_float32_for_covariance(self):
        log_size = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
        doubled_yaw = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)

        covariance, _ = self._covariance(log_size, doubled_yaw)

        self.assertEqual(covariance.dtype, torch.float32)
        self.assertTrue(torch.isfinite(covariance).all().item())
        self.assertTrue(torch.isfinite(torch.linalg.eigvalsh(covariance)).all().item())

    def test_extreme_aspect_ratios_are_strictly_well_conditioned(self):
        """Both Gaussian divisors must survive thin/long FP32 boxes."""
        doubled_yaw = torch.tensor([[0.6, 0.8]], dtype=torch.float32)
        for divisor in (4.0, 12.0):
            for narrow_log_size in (-5.0, -10.0):
                with self.subTest(divisor=divisor, narrow_log_size=narrow_log_size):
                    log_size = torch.tensor(
                        [[narrow_log_size, -narrow_log_size]],
                        dtype=torch.float32,
                    )
                    covariance, _ = box_covariance(
                        log_size,
                        doubled_yaw,
                        divisor=divisor,
                        epsilon=self.EPSILON,
                        max_abs_log_size=self.MAX_ABS_LOG_SIZE,
                    )
                    sign, logdet = torch.linalg.slogdet(covariance)
                    eigenvalues = torch.linalg.eigvalsh(covariance)

                    self.assertEqual(covariance.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(covariance).all().item())
                    self.assertTrue(torch.isfinite(logdet).all().item())
                    self.assertTrue((sign > 0).all().item())
                    self.assertTrue((eigenvalues > 0).all().item())


class ExtremeGaussianLossStabilityTests(unittest.TestCase):
    """Forward/backward finite-value contracts at difficult aspect ratios."""

    CRITERIA = (
        ("kfiou", gaussian_geometry_loss.KFIoULoss),
        ("probiou", gaussian_geometry_loss.ProbIoULoss),
        ("kld", gaussian_geometry_loss.KLDLoss),
    )

    @staticmethod
    def _maps(narrow_log_size):
        size = (float(narrow_log_size), float(-narrow_log_size))
        pred_offset = torch.tensor(
            [[[[0.05]], [[-0.03]]]], dtype=torch.float32, requires_grad=True
        )
        pred_size = torch.tensor(
            [[[[size[0]]], [[size[1]]]]],
            dtype=torch.float32,
            requires_grad=True,
        )
        pred_yaw = torch.tensor(
            [[[[0.6]], [[0.8]]]], dtype=torch.float32, requires_grad=True
        )
        target_offset = torch.zeros(1, 2, 1, 1, dtype=torch.float32)
        target_size = torch.tensor(
            [[[[size[0]]], [[size[1]]]]], dtype=torch.float32
        )
        target_yaw = torch.tensor(
            [[[[0.6]], [[0.8]]]], dtype=torch.float32
        )
        reg_mask = torch.ones(1, 1, 1, dtype=torch.float32)
        return (
            pred_offset,
            pred_size,
            pred_yaw,
            target_offset,
            target_size,
            target_yaw,
            reg_mask,
        )

    def test_extreme_boxes_have_finite_forward_and_head_gradients(self):
        for criterion_name, criterion_type in self.CRITERIA:
            progress_values = (0.0, 1.0) if criterion_name == "probiou" else (None,)
            for narrow_log_size in (-5.0, -10.0):
                for progress in progress_values:
                    with self.subTest(
                        criterion=criterion_name,
                        narrow_log_size=narrow_log_size,
                        progress=progress,
                    ):
                        maps = self._maps(narrow_log_size)
                        criterion = criterion_type(
                            epsilon=1e-6,
                            max_abs_log_size=10.0,
                        )
                        if progress is None:
                            loss, _ = criterion(*maps)
                        else:
                            loss, _ = criterion(*maps, progress=progress)

                        self.assertTrue(torch.isfinite(loss).item())
                        loss.backward()
                        for prediction in maps[:3]:
                            self.assertIsNotNone(prediction.grad)
                            self.assertTrue(
                                torch.isfinite(prediction.grad).all().item()
                            )


class KFIoUTests(unittest.TestCase):
    """Contract for the class-based full masked KFIoU geometry loss."""

    EPSILON = 1e-6
    MAX_ABS_LOG_SIZE = 10.0

    @staticmethod
    def _maps(height=1, width=1, offset=(0.0, 0.0)):
        target_offset = torch.zeros(1, 2, height, width)
        target_size = torch.tensor(
            [math.log(2.0), math.log(4.0)], dtype=torch.float32
        ).view(1, 2, 1, 1).repeat(1, 1, height, width)
        target_yaw = torch.zeros(1, 2, height, width)
        target_yaw[:, 0] = 1.0

        pred_offset = target_offset.clone()
        pred_offset[:, 0] = offset[0]
        pred_offset[:, 1] = offset[1]
        pred_size = target_size.clone()
        pred_yaw = target_yaw.clone()
        reg_mask = torch.ones(1, height, width)
        return (
            pred_offset,
            pred_size,
            pred_yaw,
            target_offset,
            target_size,
            target_yaw,
            reg_mask,
        )

    def _call_kfiou(self, *args):
        criterion_type = getattr(gaussian_geometry_loss, "KFIoULoss", None)
        self.assertIsNotNone(
            criterion_type,
            "gaussian_geometry_loss.KFIoULoss is not implemented",
        )
        criterion = criterion_type(
            epsilon=self.EPSILON,
            max_abs_log_size=self.MAX_ABS_LOG_SIZE,
        )
        self.assertIsInstance(criterion, nn.Module)
        return criterion(*args)

    def test_public_kfiou_class_is_nn_module(self):
        criterion_type = getattr(gaussian_geometry_loss, "KFIoULoss", None)
        self.assertIsNotNone(criterion_type)
        self.assertTrue(issubclass(criterion_type, nn.Module))

    def test_identical_boxes_return_kfiou_shape_minimum(self):
        maps = self._maps()

        loss, clamp_count = self._call_kfiou(*maps)

        self.assertAlmostEqual(
            float(loss.detach()), math.exp(2.0 / 3.0) - 1.0, places=5
        )
        self.assertEqual(int(clamp_count), 0)

    def test_offset_prediction_has_finite_nonzero_center_gradient(self):
        maps = list(self._maps(offset=(10.0, 0.0)))
        maps[0] = maps[0].requires_grad_()

        loss, clamp_count = self._call_kfiou(*maps)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertIsNotNone(maps[0].grad)
        gradient = maps[0].grad[:, 0, 0, 0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertNotEqual(float(gradient.abs().max()), 0.0)

    def test_mismatched_rotated_covariance_cancellation_stays_finite(self):
        """The KF fused covariance must remain usable for non-commuting boxes."""
        pred_offset = torch.tensor(
            [[[[0.031]], [[-0.017]]]], dtype=torch.float32, requires_grad=True
        )
        pred_size = torch.tensor(
            [[[[3.7437]], [[-3.0823]]]], dtype=torch.float32, requires_grad=True
        )
        pred_yaw = torch.tensor(
            [[[[0.8473]], [[0.4625]]]], dtype=torch.float32, requires_grad=True
        )
        target_offset = torch.zeros(1, 2, 1, 1, dtype=torch.float32)
        target_size = torch.tensor(
            [[[[-2.7338]], [[-4.9142]]]], dtype=torch.float32
        )
        target_yaw = torch.tensor(
            [[[[-0.64132]], [[0.0006145]]]], dtype=torch.float32
        )
        reg_mask = torch.ones(1, 1, 1, dtype=torch.float32)

        loss, clamp_count = self._call_kfiou(
            pred_offset,
            pred_size,
            pred_yaw,
            target_offset,
            target_size,
            target_yaw,
            reg_mask,
        )

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        loss.backward()
        for prediction in (pred_offset, pred_size, pred_yaw):
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_reduction_uses_positive_locations_only(self):
        single_maps = self._maps(offset=(0.5, -0.25))
        single_loss, _ = self._call_kfiou(*single_maps)

        multi_maps = list(self._maps(height=1, width=2, offset=(0.5, -0.25)))
        multi_maps[0][:, 0, 0, 1] = 9.0
        multi_maps[0][:, 1, 0, 1] = -7.0
        multi_maps[1][:, :, 0, 1] = torch.tensor([math.log(0.1), math.log(12.0)])
        multi_maps[2][:, :, 0, 1] = torch.tensor([0.0, 1.0])
        multi_maps[-1][:, 0, 1] = 0.0
        multi_loss, _ = self._call_kfiou(*multi_maps)

        self.assertTrue(torch.allclose(multi_loss, single_loss, atol=1e-6, rtol=1e-6))

    def test_empty_mask_returns_graph_connected_zero_and_zero_gradients(self):
        maps = list(self._maps())
        for index in range(3):
            maps[index] = maps[index].requires_grad_()
        maps[-1].zero_()

        loss, clamp_count = self._call_kfiou(*maps)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(int(clamp_count), 0)

        loss.backward()
        for prediction in maps[:3]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction.grad)))


class ProbIoUTests(unittest.TestCase):
    """Contract for the B3 class-based ProbIoU BD/Hellinger schedule."""

    EPSILON = 1e-6
    MAX_ABS_LOG_SIZE = 10.0
    SWITCH_FRACTION = 0.5

    @staticmethod
    def _maps(
        offset=(0.0, 0.0),
        pred_size=(math.log(2.0), math.log(4.0)),
        target_size=(math.log(2.0), math.log(4.0)),
        dtype=torch.float32,
    ):
        def pair(values):
            return torch.tensor(values, dtype=dtype).view(1, 2, 1, 1)

        target_offset = torch.zeros(1, 2, 1, 1, dtype=dtype)
        pred_offset = pair(offset)
        pred_yaw = pair((1.0, 0.0))
        target_yaw = pair((1.0, 0.0))
        reg_mask = torch.ones(1, 1, 1, dtype=torch.float32)
        return (
            pred_offset,
            pair(pred_size),
            pred_yaw,
            target_offset,
            pair(target_size),
            target_yaw,
            reg_mask,
        )

    def _call_probiou(self, *args, progress=0.0):
        criterion_type = getattr(gaussian_geometry_loss, "ProbIoULoss", None)
        self.assertIsNotNone(
            criterion_type,
            "gaussian_geometry_loss.ProbIoULoss is not implemented",
        )
        criterion = criterion_type(
            switch_fraction=self.SWITCH_FRACTION,
            epsilon=self.EPSILON,
            max_abs_log_size=self.MAX_ABS_LOG_SIZE,
        )
        self.assertIsInstance(criterion, nn.Module)
        return criterion(*args, progress=progress)

    def test_public_probiou_class_is_nn_module(self):
        criterion_type = getattr(gaussian_geometry_loss, "ProbIoULoss", None)
        self.assertIsNotNone(criterion_type)
        self.assertTrue(issubclass(criterion_type, nn.Module))

    def test_identical_boxes_return_zero_for_both_forms(self):
        maps = self._maps()

        for progress in (0.0, 0.5):
            with self.subTest(progress=progress):
                loss, clamp_count = self._call_probiou(*maps, progress=progress)
                self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
                self.assertEqual(int(clamp_count), 0)

    def test_identical_hellinger_backward_has_finite_head_gradients(self):
        maps = list(self._maps())
        for index in range(3):
            maps[index] = maps[index].requires_grad_()

        loss, clamp_count = self._call_probiou(*maps, progress=0.5)
        loss.backward()

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertEqual(int(clamp_count), 0)
        for prediction in maps[:3]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.isfinite(prediction.grad).all().item())

    def test_axis_aligned_bd_uses_prob_iou_divisor_12(self):
        maps = self._maps(offset=(2.0, 0.0))

        loss, clamp_count = self._call_probiou(*maps, progress=0.49)

        # Sigma_x = length^2 / 12 = 16 / 12, so BD = 1/8 * 2^2 / (16/12) = 3/8.
        self.assertAlmostEqual(float(loss.detach()), 0.375, places=5)
        self.assertEqual(int(clamp_count), 0)

    def test_progress_switches_from_bd_to_hellinger_at_half(self):
        maps = self._maps(offset=(2.0, 0.0))

        bd, _ = self._call_probiou(*maps, progress=0.49)
        hellinger, _ = self._call_probiou(*maps, progress=0.5)

        self.assertAlmostEqual(float(bd.detach()), 0.375, places=5)
        self.assertAlmostEqual(
            float(hellinger.detach()), math.sqrt(1.0 - math.exp(-0.375)), places=5
        )
        self.assertGreater(float(hellinger.detach()), float(bd.detach()))

    def test_offset_prediction_has_finite_nonzero_gradient(self):
        maps = list(self._maps(offset=(10.0, 0.0)))
        maps[0] = maps[0].requires_grad_()

        loss, clamp_count = self._call_probiou(*maps, progress=0.25)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertIsNotNone(maps[0].grad)
        gradient = maps[0].grad[:, 0, 0, 0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertNotEqual(float(gradient.abs().max()), 0.0)

    def test_near_square_and_thin_boxes_are_finite_under_bfloat16_autocast(self):
        cases = (
            (math.log(2.0), math.log(2.000001)),
            (math.log(0.01), math.log(12.0)),
        )
        for size in cases:
            with self.subTest(size=size):
                maps = list(
                    self._maps(
                        offset=(0.25, -0.1),
                        pred_size=size,
                        target_size=size,
                        dtype=torch.bfloat16,
                    )
                )
                maps[0] = maps[0].requires_grad_()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    loss, clamp_count = self._call_probiou(*maps, progress=0.75)
                    loss.backward()

                self.assertTrue(torch.isfinite(loss).item())
                self.assertEqual(int(clamp_count), 0)
                self.assertIsNotNone(maps[0].grad)
                self.assertTrue(torch.isfinite(maps[0].grad).all().item())

    def test_empty_mask_returns_graph_connected_zero(self):
        maps = list(self._maps())
        for index in range(3):
            maps[index] = maps[index].requires_grad_()
        maps[-1].zero_()

        loss, clamp_count = self._call_probiou(*maps, progress=0.25)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(int(clamp_count), 0)

        loss.backward()
        for prediction in maps[:3]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction.grad)))


class KLDTests(unittest.TestCase):
    """Contract for the B4 class-based directional KLD comparator."""

    EPSILON = 1e-6
    MAX_ABS_LOG_SIZE = 10.0

    @staticmethod
    def _maps(
        offset=(0.0, 0.0),
        pred_size=(math.log(2.0), math.log(4.0)),
        target_size=(math.log(2.0), math.log(4.0)),
        dtype=torch.float32,
    ):
        def pair(values):
            return torch.tensor(values, dtype=dtype).view(1, 2, 1, 1)

        target_offset = torch.zeros(1, 2, 1, 1, dtype=dtype)
        pred_offset = pair(offset)
        pred_yaw = pair((1.0, 0.0))
        target_yaw = pair((1.0, 0.0))
        reg_mask = torch.ones(1, 1, 1, dtype=torch.float32)
        return (
            pred_offset,
            pair(pred_size),
            pred_yaw,
            target_offset,
            pair(target_size),
            target_yaw,
            reg_mask,
        )

    def _call_kld(self, *args, tau=1.0):
        criterion_type = getattr(gaussian_geometry_loss, "KLDLoss", None)
        self.assertIsNotNone(
            criterion_type,
            "gaussian_geometry_loss.KLDLoss is not implemented",
        )
        criterion = criterion_type(
            tau=tau,
            epsilon=self.EPSILON,
            max_abs_log_size=self.MAX_ABS_LOG_SIZE,
        )
        self.assertIsInstance(criterion, nn.Module)
        return criterion(*args)

    def test_public_kld_class_is_nn_module(self):
        criterion_type = getattr(gaussian_geometry_loss, "KLDLoss", None)
        self.assertIsNotNone(criterion_type)
        self.assertTrue(issubclass(criterion_type, nn.Module))

    def test_identical_boxes_return_zero(self):
        loss, clamp_count = self._call_kld(*self._maps())

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertEqual(int(clamp_count), 0)

    def test_axis_aligned_prediction_given_target_uses_divisor_4(self):
        maps = self._maps(
            offset=(2.0, 0.0),
            pred_size=(math.log(2.0), math.log(8.0)),
            target_size=(math.log(2.0), math.log(4.0)),
        )

        loss, clamp_count = self._call_kld(*maps)

        # Sigma_p=diag(16,1), Sigma_t=diag(4,1), d=(2,0), D_KL=1.30685281944.
        expected = 0.45530332958
        self.assertAlmostEqual(float(loss.detach()), expected, places=5)
        self.assertEqual(int(clamp_count), 0)

    def test_tau_below_one_is_rejected(self):
        with self.assertRaises(ValueError):
            self._call_kld(*self._maps(), tau=0.999)

    def test_offset_prediction_has_finite_nonzero_gradient(self):
        maps = list(self._maps(offset=(10.0, 0.0)))
        maps[0] = maps[0].requires_grad_()

        loss, clamp_count = self._call_kld(*maps)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertIsNotNone(maps[0].grad)
        gradient = maps[0].grad[:, 0, 0, 0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertNotEqual(float(gradient.abs().max()), 0.0)

    def test_near_square_and_thin_boxes_are_finite_under_bfloat16_autocast(self):
        cases = (
            (math.log(2.0), math.log(2.000001)),
            (math.log(0.01), math.log(12.0)),
        )
        for size in cases:
            with self.subTest(size=size):
                maps = list(
                    self._maps(
                        offset=(0.25, -0.1),
                        pred_size=size,
                        target_size=size,
                        dtype=torch.bfloat16,
                    )
                )
                maps[0] = maps[0].requires_grad_()
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    loss, clamp_count = self._call_kld(*maps)
                    loss.backward()

                self.assertTrue(torch.isfinite(loss).item())
                self.assertEqual(int(clamp_count), 0)
                self.assertIsNotNone(maps[0].grad)
                self.assertTrue(torch.isfinite(maps[0].grad).all().item())

    def test_empty_mask_returns_graph_connected_zero(self):
        maps = list(self._maps())
        for index in range(3):
            maps[index] = maps[index].requires_grad_()
        maps[-1].zero_()

        loss, clamp_count = self._call_kld(*maps)
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(int(clamp_count), 0)

        loss.backward()
        for prediction in maps[:3]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction.grad)))


class MGIoUTests(unittest.TestCase):
    """Contract for the gated class-based projection MGIoU prototype."""

    EPSILON = 1e-6
    MAX_ABS_LOG_SIZE = 10.0

    @staticmethod
    def _maps(
        offset=(0.0, 0.0),
        pred_size=(math.log(2.0), math.log(4.0)),
        target_size=(math.log(2.0), math.log(4.0)),
        pred_yaw=(1.0, 0.0),
        target_yaw=(1.0, 0.0),
        dtype=torch.float32,
    ):
        def pair(values):
            return torch.tensor(values, dtype=dtype).view(1, 2, 1, 1)

        target_offset = torch.zeros(1, 2, 1, 1, dtype=dtype)
        pred_offset = pair(offset)
        reg_mask = torch.ones(1, 1, 1, dtype=torch.float32)
        return (
            pred_offset,
            pair(pred_size),
            pair(pred_yaw),
            target_offset,
            pair(target_size),
            pair(target_yaw),
            reg_mask,
        )

    def _call_mgiou(self, *args):
        self.assertIsNotNone(
            mgiou_module,
            "detector.core.losses.mgiou_loss is not implemented",
        )
        criterion_type = getattr(mgiou_module, "MGIoULoss", None)
        self.assertTrue(
            criterion_type is not None,
            "mgiou_loss.MGIoULoss is not implemented",
        )
        criterion = criterion_type(
            epsilon=self.EPSILON,
            max_abs_log_size=self.MAX_ABS_LOG_SIZE,
        )
        self.assertIsInstance(criterion, nn.Module)
        return criterion(*args)

    def test_public_mgiou_class_is_nn_module(self):
        self.assertIsNotNone(mgiou_module)
        criterion_type = getattr(mgiou_module, "MGIoULoss", None)
        self.assertIsNotNone(criterion_type)
        self.assertTrue(issubclass(criterion_type, nn.Module))

    def test_identical_boxes_return_zero(self):
        loss, clamp_count, yaw_fallback_count = self._call_mgiou(*self._maps())

        self.assertAlmostEqual(float(loss.detach()), 0.0, places=6)
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 0)

    def test_axis_aligned_non_overlap_matches_analytic_loss(self):
        maps = self._maps(offset=(10.0, 0.0))

        loss, clamp_count, yaw_fallback_count = self._call_mgiou(*maps)

        # x projections have GIoU=-3/7, y projections have GIoU=1;
        # mean GIoU=2/7 and (1-mean)/2=5/14.
        self.assertAlmostEqual(float(loss.detach()), 5.0 / 14.0, places=5)
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 0)

    def test_non_overlap_has_finite_nonzero_position_gradient(self):
        maps = list(self._maps(offset=(10.0, 0.0)))
        maps[0] = maps[0].requires_grad_()

        loss, clamp_count, yaw_fallback_count = self._call_mgiou(*maps)
        loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 0)
        self.assertIsNotNone(maps[0].grad)
        gradient = maps[0].grad[:, 0, 0, 0]
        self.assertTrue(torch.isfinite(gradient).all().item())
        self.assertNotEqual(float(gradient.abs().max()), 0.0)

    def test_theta_and_theta_plus_pi_have_same_loss(self):
        theta = 0.37
        maps = self._maps(
            pred_yaw=(math.cos(2.0 * theta), math.sin(2.0 * theta)),
            target_yaw=(math.cos(2.0 * theta), math.sin(2.0 * theta)),
        )
        maps_plus_pi = list(maps)
        maps_plus_pi[2] = torch.tensor(
            [[
                math.cos(2.0 * (theta + math.pi)),
                math.sin(2.0 * (theta + math.pi)),
            ]],
            dtype=torch.float32,
        ).view(1, 2, 1, 1)

        loss, _, _ = self._call_mgiou(*maps)
        loss_plus_pi, _, _ = self._call_mgiou(*maps_plus_pi)

        self.assertTrue(torch.allclose(loss, loss_plus_pi, atol=1e-6, rtol=1e-6))

    def test_near_square_and_thin_boxes_are_finite(self):
        cases = (
            (math.log(2.0), math.log(2.000001)),
            (math.log(0.01), math.log(12.0)),
        )
        for size in cases:
            with self.subTest(size=size):
                loss, clamp_count, yaw_fallback_count = self._call_mgiou(
                    *self._maps(offset=(0.25, -0.1), pred_size=size, target_size=size)
                )

                self.assertTrue(torch.isfinite(loss).item())
                self.assertEqual(int(clamp_count), 0)
                self.assertEqual(int(yaw_fallback_count), 0)

    def test_bfloat16_autocast_forward_and_backward_are_finite(self):
        maps = list(
            self._maps(
                offset=(0.25, -0.1),
                pred_size=(math.log(0.01), math.log(12.0)),
                target_size=(math.log(0.01), math.log(12.0)),
                dtype=torch.bfloat16,
            )
        )
        maps[0] = maps[0].requires_grad_()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            loss, clamp_count, yaw_fallback_count = self._call_mgiou(*maps)
            loss.backward()

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 0)
        self.assertIsNotNone(maps[0].grad)
        self.assertTrue(torch.isfinite(maps[0].grad).all().item())

    def test_zero_doubled_yaw_uses_one_finite_fallback(self):
        maps = self._maps(pred_yaw=(0.0, 0.0))

        loss, clamp_count, yaw_fallback_count = self._call_mgiou(*maps)

        self.assertTrue(torch.isfinite(loss).item())
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 1)

    def test_empty_mask_returns_graph_connected_zero(self):
        maps = list(self._maps())
        for index in range(3):
            maps[index] = maps[index].requires_grad_()
        maps[-1].zero_()

        loss, clamp_count, yaw_fallback_count = self._call_mgiou(*maps)

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(loss.requires_grad)
        self.assertEqual(int(clamp_count), 0)
        self.assertEqual(int(yaw_fallback_count), 0)

        loss.backward()
        for prediction in maps[:3]:
            self.assertIsNotNone(prediction.grad)
            self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction.grad)))


if __name__ == "__main__":
    unittest.main()
