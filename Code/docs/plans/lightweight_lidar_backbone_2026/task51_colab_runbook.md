# Task51 on Colab: BEV audit, baseline evidence and protocol freeze

The [standard notebook workflow](notebook_runbook.md) uses only `configs/config.json` plus notebook controls. Its default is **BEV**, `PRESET="custom"`, main focal/grouped OGA-IQA and hybrid GT. The simplified notebook handles training/evaluation; this guide supplies the separate CLI audit/evidence/protocol steps for formal comparisons. Real Colab assets and a complete trained baseline must supply the final freeze; local synthetic tests do not supply real KITTI AP or SOTA evidence.

## Choose the comparison

A comparator is the model used as the comparison baseline. Start with the internal hist14/local3/grouped OGA-IQA reference **without focal context**: set `C4_CONTEXT="none"` with default custom controls, or select the built-in `UNDER1M_GROUPED_OGA_IQA` preset. Keep encoding, heads, objective, seed, split, schedule and augmentation matched to the main focal candidate. This isolates focal's contribution; it is an architecture ablation and does not establish superiority over external algorithms.

If no matching baseline checkpoint exists, train that baseline through the existing notebook with a distinct run name. Reuse a matching complete Colab run when available. No training is launched by the audit/evidence/protocol CLIs. Preserve `config.resolved.json`, `metrics.jsonl`, `selected/best.pt`, `selected/selection.json` and the retained checkpoint named in that selection file. A short smoke or incomplete training history is not accepted as the reproduced baseline for a longer configured schedule.

Existing model comparison CLI uses one config for all checkpoints. For a no-context versus focal architecture comparison, evaluate each checkpoint separately with its **own** resolved config, as below.

## Set actual paths

Run from the Colab repository directory. Use its installed interpreter; the laptop's temporary reference environment is not a Colab dependency. The current notebook supplies `REPO_DIR`, `CONFIG` and `RAW_KITTI_ROOT`:

```python
import json
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(REPO_DIR).resolve()
PYTHON = sys.executable
CANDIDATE_CONFIG = Path(CONFIG).resolve()  # Actual resolved runtime conditions planned for the candidate.
BASELINE_RUN = Path("/content/Lidar/artifacts/kitti/<actual_baseline_run>").resolve()
BASELINE_CONFIG = BASELINE_RUN / "config.resolved.json"
BASELINE_CHECKPOINT = BASELINE_RUN / "selected/best.pt"
RAW_TRAINING = Path(RAW_KITTI_ROOT).resolve()
if (RAW_TRAINING / "training").is_dir():
    RAW_TRAINING = RAW_TRAINING / "training"
RAW_LOCAL_PARENT = RAW_TRAINING.parent  # Historical local evaluator appends training/.
EVIDENCE_DIR = CANDIDATE_CONFIG.parent / "benchmark"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
MODE = "local_bev"

def run_tool(script, *args):
    subprocess.run([PYTHON, str(REPO_DIR / script), *map(str, args)], cwd=REPO_DIR, check=True)

candidate = json.loads(CANDIDATE_CONFIG.read_text())
baseline = json.loads(BASELINE_CONFIG.read_text())
assert candidate["data"].get("box_mode", "bev") == "bev"
assert baseline["data"].get("box_mode", "bev") == "bev"
assert candidate["model"]["c4_context"] == "focal"
assert baseline["model"].get("c4_context", "none") == "none"
assert candidate["model"]["backbone_out_dim"] == baseline["model"]["backbone_out_dim"] == 32
assert candidate["loss"] == baseline["loss"]
assert candidate["data"]["bev_encoding"] == baseline["data"]["bev_encoding"]
assert candidate["data"]["head_groups"] == baseline["data"]["head_groups"]
assert candidate["seed"] == baseline["seed"]
assert candidate["augmentation"] == baseline["augmentation"]
```

Replace the baseline run placeholder with the real Colab path. Also check training conditions: epochs, warmup, learning rate, optimizer, precision, physical batch/accumulation and backends. Different conditions must be disclosed; they weaken a controlled focal ablation even when evaluation protocols match. Source config/checkpoint hashes must correspond to the files actually used by each evaluator. If candidate runtime settings change after its smoke/audit, regenerate matching evidence.

## Audit the candidate's actual assets

```python
ASSET_AUDIT = EVIDENCE_DIR / "assets_local_bev.json"
run_tool("tools/benchmarks/audit_kitti_assets.py",
         "--config", CANDIDATE_CONFIG, "--repo-root", REPO_DIR,
         "--kitti-root", RAW_TRAINING, "--metric-mode", MODE,
         "--output", ASSET_AUDIT)
```

The audit reads every configured train/validation point file and every crop, checks finite float32 contents, processed labels, unique/disjoint IDs, original validation labels/calibration, and configured v1 GT database counts/conventions. Database source IDs must equal the train split; each entry's class, label index and box geometry must match its actual train source label. Crop point counts must match bytes. It records every audited file's SHA256 and a deterministic inventory hash without altering source assets. Data is read in bounded chunks; no dataset-sized point buffer is retained.

If the database was built from a differently formatted original manifest with the same training IDs, add `--database-source-split /actual/original_manifest.txt`. Its bytes must match database metadata and its IDs must match the current train split. Do not rewrite metadata to suppress a mismatch. The audit does not authenticate arbitrary data as genuine KITTI or match every crop point against the original scene. Freeze re-runs the audit to catch stale files, so expect another dataset read.

## Record and evaluate the baseline

```python
BASELINE_PROTOCOL = EVIDENCE_DIR / "baseline_protocol_local_bev.json"
BASELINE_EVALUATION = EVIDENCE_DIR / "baseline_evaluation_local_bev.json"
BASELINE_EVIDENCE = EVIDENCE_DIR / "baseline_evidence_local_bev.json"
BASELINE_SPLIT = Path(baseline["val"]["data"])
if not BASELINE_SPLIT.is_absolute():
    BASELINE_SPLIT = REPO_DIR / BASELINE_SPLIT

run_tool("tools/benchmarks/benchmark_protocol.py",
         "--config", BASELINE_CONFIG, "--metric-mode", MODE,
         "--output", BASELINE_PROTOCOL)
run_tool("tools/kitti_training_pipeline/evaluate_kitti_bev.py",
         "--name", "internal_no_context", "--backend", "pytorch",
         "--model", BASELINE_CHECKPOINT, "--config", BASELINE_CONFIG,
         "--detector-root", REPO_DIR / "detector", "--kitti-root", RAW_LOCAL_PARENT,
         "--split", BASELINE_SPLIT, "--device", "cuda",
         "--score-threshold", baseline.get("evaluation", {}).get("score_threshold", 0.05),
         "--nms-threshold", baseline.get("evaluation", {}).get("nms_threshold", 0.10),
         "--max-detections", baseline.get("evaluation", {}).get("max_detections", 500),
         "--output", BASELINE_EVALUATION)
run_tool("tools/benchmarks/benchmark_evidence.py",
         "--protocol", BASELINE_PROTOCOL, "--evaluation", BASELINE_EVALUATION,
         "--selection", BASELINE_RUN / "selected/selection.json",
         "--history", BASELINE_RUN / "metrics.jsonl", "--output", BASELINE_EVIDENCE)
```

This records **local ROI APBEV R40 only**. It does not invent reference R11 or call the local evaluator official KITTI APBEV. Evidence validates the entire validation split, actual input/config/checkpoint hashes, classes/IoUs/difficulty/ignore/quality/decode settings and all nine finite class/difficulty AP values, then recomputes Moderate/AP9 macros. Missing GT classes are rejected instead of silently averaging fewer classes. Minimum-loss selection must match the complete ordered configured training history, successful updates, checkpoint epoch/objective and the full retained checkpoint payload. Separate torch.save archives may differ in bytes; model/criterion/optimizer/scheduler/scaler/RNG/config state must agree and both file hashes remain recorded.

Reports from before input-asset provenance was added must be re-evaluated. `--synthetic` explicitly marks test evidence; the protocol recorder rejects such evidence for a real benchmark freeze. Supplied evaluator/training JSON remains trusted execution evidence, not cryptographic proof of remote execution. The initial evidence recorder supports this repository's PyTorch checkpoint/evaluator/selection formats. External algorithms need their own disclosed and verified protocol/evidence adapter; matching their paper's table numbers is insufficient.

## Run candidate GPU gates and freeze

```python
WORKERS = candidate["train"]["num_workers"]
SMOKE_0 = EVIDENCE_DIR / "smoke_workers0.json"
SMOKE_CONFIGURED = EVIDENCE_DIR / f"smoke_workers{WORKERS}.json"
for workers, output in {(0, SMOKE_0), (WORKERS, SMOKE_CONFIGURED)}:
    run_tool("tools/benchmarks/smoke_detector.py", "--config", CANDIDATE_CONFIG,
             "--device", "cuda", "--precisions", "fp32", "fp16", "bf16",
             "--steps", "3", "--num-workers", workers, "--full-resolution", "--output", output)

FROZEN_PROTOCOL = EVIDENCE_DIR / "candidate_protocol_local_bev.json"
run_tool("tools/benchmarks/benchmark_protocol.py",
         "--config", CANDIDATE_CONFIG, "--metric-mode", MODE,
         "--smoke-report", SMOKE_0, "--smoke-report", SMOKE_CONFIGURED,
         "--asset-audit", ASSET_AUDIT, "--comparator-evidence", BASELINE_EVIDENCE,
         "--output", FROZEN_PROTOCOL)
protocol = json.loads(FROZEN_PROTOCOL.read_text())
print(protocol["status"], protocol["pending_gates"], protocol["long_training_allowed"])
```

All FP32/FP16/BF16 runs, successful updates, exact optimizer/scheduler/scaler/RNG restore, configured full input shape and <1M backbone+neck must pass for workers0/configured workers. Unsupported BF16 is partial, not a pass. This requirement certifies the advertised precision matrix; record an explicitly revised policy if different Colab hardware cannot support it. No Inductor or TensorRT claim is implied.

A matched declared comparator protocol alone leaves baseline evidence pending. Actual comparator evidence supplies its own immutable protocol path and metrics; the recorder rechecks source files and selection history. Data roots must match the audited assets. Metric/sampling/domain/split/selection/threshold/NMS/quality/peak/cap/aggregation differences prevent freeze. Only an empty `pending_gates` produces `long_training_allowed=true`; candidate accuracy remains unmeasured until its actual training/evaluation. Freeze does not launch training or certify SOTA.

## Final 3D results later

Keep the notebook BEV default now. When explicitly selecting final 3D results, train or fine-tune a complete 3D candidate **and matched 3D baseline**; a BEV checkpoint cannot acquire z/height by changing an evaluation flag. See [the 3D runbook](colab_3d_runbook.md).

For that stage, use `MODE="3d"` or `"bev"` with complete 3D configs, `audit_kitti_assets.py --metric-mode MODE` (which additionally audits P2/images), and `evaluate_kitti_3d.py --checkpoint ... --metric-mode MODE` for each own-config checkpoint. Record separate protocols/evidence for AP3D and reference APBEV; both contain R11/R40. The recorder also requires `--reference-verification` from a successful actual pinned CUDA parity test record and matching 3D smoke reports before final freeze. Colab must generate its own compatible runtime evidence; the laptop's ignored reports are not a substitute.
