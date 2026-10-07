#!/usr/bin/env python3
"""Record the requested benchmark contract; readiness requires actual evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
from common import checkpoint_identity, git_metadata, input_shape, read_json, sha256, write_json


def split_record(path):
    ids = [line.strip().split(";")[0] for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Split must contain unique nonempty IDs: {path}")
    if any(not identifier.isdigit() for identifier in ids):
        raise ValueError(f"Invalid KITTI split ID: {path}")
    return {"path": str(path), "count": len(ids), "sha256": sha256(path)}, set(ids)


def compare_protocols(expected, comparator):
    """Compare metric/domain/split/selection fields; architecture may differ."""
    paths = ["evaluation.metric_mode", "evaluation.ap_samplings", "evaluation.classes",
             "evaluation.iou_thresholds", "evaluation.domain", "evaluation.difficulty_policy",
             "evaluation.orientation_policy", "splits.val.sha256", "checkpoint_selection.policy",
             "decode.score_threshold", "decode.nms_threshold", "decode.max_detections",
             "decode.nms_alpha", "decode.peak_mode", "decode.cap_scope", "decode.quality_semantics",
             "aggregation.moderate_macro_mean_classes", "aggregation.ap9_macro_mean"]
    if expected.get("evaluation", {}).get("metric_mode") == "local_bev":
        paths.extend(["evaluation.roi_geometry", "splits.train.sha256"])
    def value(report, path):
        result = report
        for key in path.split("."):
            if not isinstance(result, dict) or key not in result:
                return None
            result = result[key]
        return result
    # Missing fields cannot establish parity even when absent in both records.
    mismatches = [path for path in paths if value(expected, path) is None or
                  value(comparator, path) is None or value(expected, path) != value(comparator, path)]
    return {"matched": not mismatches, "mismatches": mismatches,
            "limits": "Matching declarations do not establish evaluator implementation parity or reproduced accuracy."}


def freeze_protocol(config_path, repo_root=ROOT, *, metric_mode="3d", comparators=(),
                    smoke_reports=(), reference_verification=None, asset_audit=None, comparator_evidence=()):
    """Freeze candidate settings and requested metrics, retaining unresolved gates."""
    if metric_mode not in {"local_bev", "3d", "bev"}:
        raise ValueError("metric_mode must be local_bev, 3d or bev")
    root = Path(repo_root)
    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    config = read_json(path)
    from tools.kitti_training_pipeline.notebook_config import validate_notebook_config
    # Validate configurable decoding without changing the source model dictionary.
    validate_notebook_config(copy.deepcopy(config))
    if metric_mode == "local_bev" and config["data"].get("box_mode", "bev") != "bev":
        raise ValueError("Local BEV protocol requires a BEV runtime config")
    train_split, training_ids = split_record(root / config["train"]["data"])
    val_split, validation_ids = split_record(root / config["val"]["data"])
    overlap = sorted(training_ids & validation_ids)
    if overlap:
        raise ValueError("Training and validation splits must be disjoint")
    result = {"version": 1, "status": "candidate policy recorded; final benchmark freeze pending gates",
        "source_config": str(path), "source_config_sha256": sha256(path),
        "resolved_config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "candidate_config": copy.deepcopy(config), "checkpoint_identity": checkpoint_identity(config),
        "splits": {"train": train_split, "val": val_split, "overlap": overlap},
        "evaluation": {"metric_mode": metric_mode, "ap_samplings": ["R11", "R40"],
            "classes": ["Car", "Pedestrian", "Cyclist"],
            "iou_thresholds": {"Car": .7, "Pedestrian": .5, "Cyclist": .5},
            "domain": "full benchmark GT; ROI metrics separate",
            "difficulty_policy": "pinned reference KITTI Easy/Moderate/Hard with ignored/neighbor/DontCare rules",
            "orientation_policy": "pi-symmetric box yaw; no AOS claims",
            "reference_evaluator": "OpenPCDet KITTI evaluator; immutable revision and parity pending task57",
            "recall_policy": "Report both reference R11 and R40 independently; never substitute local BEV R40 for AP3D"},
        "current_runtime": {"metric_mode": "bev", "ap_samplings": ["R40"],
            "protocol": "local loader-aligned rotated BEV AP; strict configured XY-center ROI",
            "box_mode": config["data"].get("box_mode", "bev"),
            "reference_equivalence_verified": False},
        "aggregation": {"moderate_macro_mean_classes": ["Car", "Pedestrian", "Cyclist"],
            "ap9_macro_mean": "All 3 classes x Easy/Moderate/Hard, separately for each metric and recall sampling",
            "missing_classes": "Report per-class missing counts; do not silently average a reduced class set"},
        "checkpoint_selection": {"policy": "minimum validation loss",
            "alternate_ap_selection": "Only if explicitly selected, tested and matched across comparators"},
        "decode": {"score_threshold": .05, "nms_threshold": .10, "max_detections": 500,
            "nms_alpha": .5, "peak_mode": "per_class", "cap_scope": "global after classwise NMS",
            "quality_semantics": "BEV IQA; changing to 3D IoU is a separate feature"},
        "training": {"seed": config["seed"], "schedule": copy.deepcopy(config["train"]),
            "augmentation": copy.deepcopy(config["augmentation"]),
            "main_attention": (f"{config['model'].get('c4_context', 'none')}, "
                               f"local {config['model'].get('local_attention', 'none')}; optional comparisons named separately"),
            "required_3d_extension": "owned z_bottom/log_height branches with explicit vertical weight1; final 3D config pending task58"},
        "pending_gates": ["reference_evaluator_revision", "reference_R11_R40_parity",
            "configured_data_and_train_only_gt_database_audit", "comparator_protocols_and_reproduced_baseline"],
        "comparison_status": "pending comparator protocols", "comparators": [],
        "long_training_allowed": False, "accuracy_measured": False, "source": git_metadata(root)}
    if metric_mode == "3d":
        result["pending_gates"].append("vertical_3d_pipeline_and_final_GPU_recipe")
    else:
        result["pending_gates"].append("reference_BEV_mode_and_R11_support")
    evaluation = config.get("evaluation", {})
    result["decode"].update(score_threshold=evaluation.get("score_threshold", .05),
                            nms_threshold=evaluation.get("nms_threshold", .10),
                            max_detections=evaluation.get("max_detections", 500))
    if metric_mode == "local_bev":
        result["evaluation"].update(metric_mode="local_bev", ap_samplings=["R40"],
            domain="strict configured XY-center ROI", difficulty_policy="local KITTI-style Easy/Moderate/Hard",
            reference_evaluator="historical local rotated BEV evaluator; not pinned reference APBEV",
            recall_policy="Local R40 only; reference R11/R40 belongs to the explicit final comparison",
            roi_geometry={key: config["data"]["kitti"]["geometry"][key] for key in ("x_min", "x_max", "y_min", "y_max")})
        result["current_runtime"]["metric_mode"] = "local_bev"
        uses_iqa = config["model"].get("header_use_iou", False)
        result["decode"].update(
            nms_alpha=(config.get("nms_alpha", config["data"]["kitti"].get("nms_alpha", .5)) if uses_iqa else 0.),
            peak_mode=config.get("peak_mode", config["data"]["kitti"].get("peak_mode", "per_class")))
        if not uses_iqa:
            result["decode"]["quality_semantics"] = "no IoU quality ranking"
        result["training"]["required_3d_extension"] = "not required for local BEV development"
        result["pending_gates"] = ["final_BEV_GPU_recipe", "configured_data_and_train_only_gt_database_audit",
                                   "comparator_protocols_and_reproduced_baseline"]
    if config["data"].get("box_mode", "bev") == "3d":
        from tools.kitti_training_pipeline.evaluate_kitti_3d import reference_provenance
        reference = reference_provenance()
        result["evaluation"].update(reference_evaluator="unchanged pinned OpenPCDet KITTI evaluator",
                                    reference_revision=reference["revision"], reference_source_sha256=reference["sha256"])
        result["current_runtime"].update(metric_mode=metric_mode, ap_samplings=["R11", "R40"],
                                        protocol="full benchmark GT; pinned reference; genuine projected boxes")
        result["training"]["required_3d_extension"] = (
            f"owned vertical branch/target/loss/decode implemented; fixed vertical weight={config['loss']['vertical_loss_weight']}")
        result["pending_gates"] = ["reference_R11_R40_parity", "final_3d_GPU_recipe",
            "configured_data_and_train_only_gt_database_audit", "comparator_protocols_and_reproduced_baseline"]
        result["verification_evidence"] = {"smoke_reports": [], "reference": None}
        seen_workers = set()
        for report_path in smoke_reports:
            report = read_json(Path(report_path))
            if (report.get("status") != "passed" or not str(report.get("device", "")).startswith("cuda") or
                    not report.get("cuda_device_name")):
                raise ValueError("GPU smoke evidence must be a passed actual CUDA report")
            if report.get("source_config_sha256") != result["resolved_config_sha256"] or report.get("box_mode") != "3d":
                raise ValueError("GPU smoke evidence config does not match the 3D candidate")
            workers = report.get("worker_count")
            if type(workers) is not int or workers < 0:
                raise ValueError("GPU smoke worker_count must be a nonnegative integer")
            precisions = report.get("precisions", {})
            if set(precisions) != {"fp32", "fp16", "bf16"}:
                raise ValueError("GPU smoke evidence requires all three precisions")
            for run in precisions.values():
                if (run.get("status") != "passed" or not run.get("vertical_gradient_verified") or
                        not run.get("vertical_target_ownership_verified") or
                        not run.get("optimizer_state_restore_verified") or
                        not run.get("scheduler_scaler_rng_restore_verified") or
                        not run.get("full_resolution_finite") or
                        not all(run.get("vertical_head_weight_changed", {}).get(g["name"]) is True
                                for g in result["checkpoint_identity"]["groups"])):
                    raise ValueError("GPU smoke precision/vertical/resume evidence is incomplete")
            seen_workers.add(workers)
            result["verification_evidence"]["smoke_reports"].append({"path": str(report_path),"sha256": sha256(Path(report_path))})
        if {0, config["train"]["num_workers"]} <= seen_workers:
            result["pending_gates"].remove("final_3d_GPU_recipe")
            result["current_runtime"]["gpu_recipe_verified"] = True
        if reference_verification is not None:
            verified = read_json(Path(reference_verification))
            provenance = verified.get("reference_provenance", {})
            if (provenance.get("revision") != reference["revision"] or provenance.get("sha256") != reference["sha256"] or
                    verified.get("gpu_reference_cases", 0) < 1 or verified.get("full", {}).get("exit_code") != 0 or
                    verified.get("full", {}).get("skips") != 0):
                raise ValueError("Pinned reference parity verification is incomplete or mismatched")
            result["pending_gates"].remove("reference_R11_R40_parity")
            result["current_runtime"]["reference_equivalence_verified"] = True
            result["verification_evidence"]["reference"] = {"path": str(reference_verification),"sha256": sha256(Path(reference_verification))}
        data_root = root / config["data"]["kitti"]["location"]
        result["configured_data"] = {"root": str(data_root),"exists": data_root.is_dir(),
            "pointcloud_exists": (data_root/"pointcloud").is_dir(), "label_exists": (data_root/"label").is_dir(),
            "train_gt_database_exists": (data_root/"gt_database/dbinfos_train.json").is_file(),
            "audit_complete": False}
    elif smoke_reports or reference_verification is not None:
        if metric_mode != "local_bev" or reference_verification is not None:
            raise ValueError("3D verification evidence requires a complete 3D candidate config")
        workers_seen = set()
        result["verification_evidence"] = {"smoke_reports": [], "reference": None}
        for report_path in smoke_reports:
            report = read_json(Path(report_path))
            if (report.get("status") != "passed" or not str(report.get("device", "")).startswith("cuda") or
                    not report.get("cuda_device_name") or report.get("box_mode") != "bev" or
                    report.get("source_config_sha256") != result["resolved_config_sha256"]):
                raise ValueError("BEV GPU smoke must be a passed actual CUDA report matching the candidate config")
            workers = report.get("worker_count")
            if type(workers) is not int or workers < 0 or set(report.get("precisions", {})) != {"fp32", "fp16", "bf16"}:
                raise ValueError("BEV GPU smoke worker/precision evidence is incomplete")
            fields = ("finite_loss", "finite_gradients", "head_weight_changed", "optimizer_membership_exact",
                      "checkpoint_restore_verified", "optimizer_state_restore_verified",
                      "scheduler_scaler_rng_restore_verified", "criterion_state_unchanged_in_validation",
                      "full_resolution_finite")
            for run in report["precisions"].values():
                if run.get("status") != "passed" or any(run.get(key) is not True for key in fields):
                    raise ValueError("BEV GPU smoke finite/update/restore/full-resolution evidence is incomplete")
                updates = run.get("optimizer_updates")
                budget = run.get("parameter_counts", {}).get("backbone_including_neck")
                if (type(updates) is not int or updates < 1 or
                        run.get("full_resolution_input_shape") != list(input_shape(config)) or
                        type(budget) is not int or not 0 < budget < 1_000_000):
                    raise ValueError("BEV GPU update count, configured full shape or backbone budget is invalid")
            workers_seen.add(workers)
            result["verification_evidence"]["smoke_reports"].append({"path": str(report_path), "sha256": sha256(Path(report_path))})
        if {0, config["train"]["num_workers"]} <= workers_seen:
            result["pending_gates"].remove("final_BEV_GPU_recipe")
            result["current_runtime"]["gpu_recipe_verified"] = True
    if asset_audit is not None:
        from tools.benchmarks.audit_kitti_assets import verify_asset_audit
        audited = verify_asset_audit(read_json(Path(asset_audit)), path, root, metric_mode=metric_mode)
        result["configured_data"] = {"root": audited["processed_root"], "kitti_root": audited["kitti_root"],
            "audit_complete": True, "inventory_sha256": audited["inventory_sha256"],
            "audit_evidence": {"path": str(asset_audit), "sha256": sha256(Path(asset_audit))}}
        result["pending_gates"].remove("configured_data_and_train_only_gt_database_audit")
    for comparator in comparators:
        comparator_path = Path(comparator)
        declared = read_json(comparator_path)
        result["comparators"].append({"path": str(comparator_path), "sha256": sha256(comparator_path),
                                     **compare_protocols(result, declared)})
    if result["comparators"]:
        result["comparison_status"] = ("declared protocols match; implementation/evidence pending"
            if all(c["matched"] for c in result["comparators"]) else "mismatched comparator protocols")
    if comparator_evidence:
        from tools.benchmarks.benchmark_evidence import verify_evidence
        result["reproduced_evidence"] = []
        for evidence_path in comparator_evidence:
            evidence = verify_evidence(read_json(Path(evidence_path)))
            if evidence["synthetic"]:
                raise ValueError("Synthetic evidence cannot clear a real benchmark baseline gate")
            other_path = Path(evidence["protocol_path"])
            other = read_json(other_path)
            comparison = compare_protocols(result, other)
            if not comparison["matched"]:
                raise ValueError(f"Reproduced comparator protocol does not match: {comparison['mismatches']}")
            from tools.benchmarks.audit_kitti_assets import repo_path
            if evidence["processed_root"] != str(repo_path(config["data"]["kitti"]["location"], root)):
                raise ValueError("Comparator evaluation must use the same configured processed assets")
            if asset_audit is not None and evidence["kitti_root"] != result["configured_data"]["kitti_root"]:
                raise ValueError("Comparator evaluation must use the audited original KITTI assets")
            if not any(Path(item["path"]).resolve() == other_path.resolve() for item in result["comparators"]):
                result["comparators"].append({"path": str(other_path), "sha256": sha256(other_path), **comparison})
            result["reproduced_evidence"].append({"path": str(evidence_path), "sha256": sha256(Path(evidence_path)),
                "protocol_sha256": evidence["protocol_sha256"], "checkpoint_sha256": evidence["checkpoint_sha256"],
                "name": evidence["name"], "metrics": evidence["metrics"]})
        if all(item["matched"] for item in result["comparators"]):
            result["pending_gates"].remove("comparator_protocols_and_reproduced_baseline")
            result["baseline_accuracy_verified"] = True
            result["comparison_status"] = "matched protocols and reproduced evaluation evidence"
    if not result["pending_gates"]:
        result.update(status="protocol frozen for supplied verified evidence", long_training_allowed=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--metric-mode", choices=("local_bev", "3d", "bev"), default="local_bev")
    parser.add_argument("--comparator-protocol", action="append", type=Path, default=[])
    parser.add_argument("--smoke-report", action="append", type=Path, default=[])
    parser.add_argument("--reference-verification", type=Path)
    parser.add_argument("--asset-audit", type=Path)
    parser.add_argument("--comparator-evidence", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = freeze_protocol(args.config, args.repo_root, metric_mode=args.metric_mode,
                             comparators=args.comparator_protocol, smoke_reports=args.smoke_report,
                             reference_verification=args.reference_verification, asset_audit=args.asset_audit,
                             comparator_evidence=args.comparator_evidence)
    write_json(args.output, result)
    print(json.dumps({"status": result["status"], "evaluation": result["evaluation"],
                      "comparison_status": result["comparison_status"], "pending_gates": result["pending_gates"],
                      "long_training_allowed": result["long_training_allowed"]}, indent=2))
    return result


if __name__ == "__main__":
    main()
