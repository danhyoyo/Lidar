# Active Point Sampling Screening Implementation Plan

**Execution status:** Closed on 2026-09-20. Static S0 failed, all oracle
points failed the safety guardrails, and final validation rejected rate 0.75
as the default. Dense A4 is retained; no policy or RL code was implemented.

Final artifacts:

- `results/research/active_sampling/gate_report.json`
- `results/research/active_sampling/final_comparison.json`
- `results/research/active_sampling/baseline_a4_epoch50.json`

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether deterministic raw-point sampling gives MobileBEV A4 a safe end-to-end speedup and whether a dynamic policy has enough oracle headroom to justify implementation.

**Architecture:** Add one deterministic sampling helper to the existing preprocessing module and expose it only through the existing KITTI evaluator during screening. A single research CLI creates the leakage-safe splits, snapshots baseline provenance, and computes the static/oracle gates from cached evaluator outputs. Do not modify training or build a policy in this plan.

**Tech Stack:** Python 3.13, NumPy, PyTorch, existing KITTI evaluator and dependency-free test scripts.

**Spec:** `docs/plans/2026-09-19-active-point-sampling-spec.md`

## Global Constraints

- Frozen model: MobileBEV-Lite A4 `RichBEV-8 + SG-FPN + Center3D`, owner-reported checkpoint epoch 50.
- Actions: sampling rates `{1.00, 0.75, 0.50, 0.25}` only.
- Sampling seed: `42`; derive each frame seed from the first eight bytes of `SHA-256("42:" + frame_id)`.
- Sampling happens before ROI filtering and `encode_bev`; `1.00` returns the original array unchanged.
- Lower-rate subsets are nested, and retained indices are sorted before indexing.
- No density correction, retraining, previous-frame state, ego velocity, PPO, or spatial routing.
- The 1,497-frame validation split remains sealed until static/policy choices are frozen.
- Static pass: 3D mAP Moderate drop `<= 0.8`, each class AP drop `<= 1.5`, each class recall drop `<= 0.03`, qualifying distance-band recall drop `<= 0.05`, preprocess improvement `>= 35%`, and E2E improvement `>= 15%`.
- Oracle pass: at matched mean rate, `>= 0.3` mAP improvement; or `>= 10` percentage-point lower mean rate within `0.2` mAP; all safety guards still apply.
- Stop after the oracle decision. A deployable contextual policy requires a separate plan only if the oracle gate passes.

## File Map

- Modify `detector/core/datasets/utils_1/preprocess.py`: deterministic point sampler beside `encode_bev`.
- Modify `tools/kitti_training_pipeline/evaluate_kitti_bev.py`: sampling CLI, split preprocessing timings, sampling metadata, and per-frame oracle quality.
- Create `tools/research/active_point_sampling.py`: baseline snapshot, deterministic split creation, static aggregation, and oracle sweep.
- Modify `tests/test_mobile_bev.py`: focused checks for sampler, evaluator integration, split determinism, and oracle action selection.
- Generate `splits/kitti/active_sampling_policy_train.txt` and `splits/kitti/active_sampling_controller_dev.txt` only through the research CLI.
- Generate run artifacts under `results/research/active_sampling/`; do not commit prediction archives.

---

### Task 1: Add deterministic nested point sampling

**Files:**
- Modify: `detector/core/datasets/utils_1/preprocess.py:1-20`
- Modify: `tests/test_mobile_bev.py:69-122, 375-415`

**Interfaces:**
- Produces: `sample_points(points: np.ndarray, rate: float, frame_id: str, seed: int = 42) -> np.ndarray`
- Guarantees: dense identity, stable per-frame permutation, nested subsets, preserved source order, and no mutation of `points`.

- [ ] **Step 1: Add a failing sampler check**

Add `hashlib` only to production code. Add this check to `tests/test_mobile_bev.py`, register it in `CHECKS`, and add a `--sampling` selector:

```python
def check_sampling():
    preprocess = load_preprocess()
    points = np.arange(80, dtype=np.float32).reshape(20, 4)
    dense = preprocess.sample_points(points, 1.0, "000123")
    half_a = preprocess.sample_points(points, 0.5, "000123")
    half_b = preprocess.sample_points(points, 0.5, "000123")
    quarter = preprocess.sample_points(points, 0.25, "000123")
    assert dense is points
    np.testing.assert_array_equal(half_a, half_b)
    assert len(half_a) == 10 and len(quarter) == 5
    assert {tuple(row) for row in quarter} <= {tuple(row) for row in half_a}
    source_positions = [np.flatnonzero((points == row).all(axis=1))[0] for row in half_a]
    assert source_positions == sorted(source_positions)
    assert preprocess.sample_points(points[:0], 0.25, "empty").shape == (0, 4)
    for invalid in (0.0, -0.1, 1.01, float("nan")):
        try:
            preprocess.sample_points(points, invalid, "000123")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid sampling rate accepted: {invalid}")
```

- [ ] **Step 2: Run the check and confirm it fails**

Run `python tests/test_mobile_bev.py --sampling`.

Expected: failure because `utils_1.preprocess.sample_points` does not exist.

- [ ] **Step 3: Implement the minimum sampler**

Add this function before `encode_bev`:

```python
def sample_points(points, rate, frame_id, seed=42):
    rate = float(rate)
    if not np.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("sampling rate must be finite and in (0, 1]")
    if rate == 1.0 or not len(points):
        return points
    digest = hashlib.sha256(f"{seed}:{frame_id}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    count = max(1, int(len(points) * rate))
    indexes = np.sort(rng.permutation(len(points))[:count])
    return points[indexes]
```

Do not add a class, registry, alternate sampling method, or configuration schema.

- [ ] **Step 4: Run the focused and full checks**

```bash
python tests/test_mobile_bev.py --sampling
python tests/test_mobile_bev.py
```

Expected: all checks print `PASS`.

- [ ] **Step 5: Commit the sampler**

```bash
git add detector/core/datasets/utils_1/preprocess.py tests/test_mobile_bev.py
git commit -m "feat: add deterministic point sampling"
```

---

### Task 2: Add sampled preprocessing to the evaluator

**Files:**
- Modify: `tools/kitti_training_pipeline/evaluate_kitti_bev.py:438-643`
- Modify: `tests/test_mobile_bev.py:319-350, 375-415`

**Interfaces:**
- Consumes: `sample_points(points, rate, frame_id, seed)` from Task 1.
- Produces: `load_bev_frame(frame_id, pointcloud_root, geometry, bev_encoding, sampling_rate, sampling_seed) -> tuple[torch.Tensor, dict, dict]`.
- Extends: `run_evaluation(..., sampling_rate: float = 1.0, sampling_seed: int = 42, command: Sequence[str] | None = None)`.
- Adds CLI: `--sampling-rate {0.25,0.5,0.75,1.0}` and `--sampling-seed 42`.

- [ ] **Step 1: Add failing evaluator checks**

Add `import tempfile` and extend `check_metrics()`:

```python
args = evaluator.parser().parse_args([
    "--name", "sampled", "--backend", "pytorch", "--model", "model.pt",
    "--config", "config.json", "--detector-root", "detector",
    "--kitti-root", "kitti", "--split", "split.txt", "--output", "out.json",
    "--sampling-rate", "0.5", "--sampling-seed", "7",
])
assert args.sampling_rate == 0.5 and args.sampling_seed == 7

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    points = np.arange(80, dtype=np.float32).reshape(20, 4)
    points.tofile(root / "000123.bin")
    voxel, timings, counts = evaluator.load_bev_frame(
        "000123", root, geometry,
        {"name": "rich8", "density_norm": 32, "intensity_scale": 1}, 0.5, 42,
    )
    assert tuple(voxel.shape) == (8, 2, 2)
    assert counts == {"raw_points": 20, "selected_points": 10}
    assert set(timings) == {"point_load", "sampling", "bev_encode", "preprocess"}
```

Use the existing 2 m × 2 m geometry from `check_encoder()`.

- [ ] **Step 2: Run `python tests/test_mobile_bev.py --metrics`**

Expected: parser rejection or missing `load_bev_frame`.

- [ ] **Step 3: Capture the unmodified dense smoke control**

Before editing the evaluator, run it on NVIDIA L4 with
`--split splits/kitti/train.txt --max-frames 200` and save the output as
`results/research/active_sampling/dense_unmodified_smoke.json`. Retain its
prediction archive. Do not inspect the sealed validation split.

- [ ] **Step 4: Implement direct evaluator preprocessing**

After `configure_detector_imports(detector_root)`, import `encode_bev` and `sample_points` from `utils_1.preprocess`. Implement `load_bev_frame` with `time.perf_counter()` around exactly:

```python
points = np.fromfile(pointcloud_root / f"{frame_id}.bin", dtype=np.float32).reshape(-1, 4)
sampled = sample_points(points, sampling_rate, frame_id, sampling_seed)
bev = encode_bev(sampled, geometry, bev_encoding)
voxel = torch.from_numpy(bev).permute(2, 0, 1)
```

`preprocess` spans all four lines. `bev_encode` includes encoding and tensor conversion. Remove the evaluator's `Dataset` construction; training and `detector/core/datasets/dataset.py` remain unchanged.

- [ ] **Step 5: Thread sampling through evaluation and output**

Validate the rate against the four allowed values and resolve:

```python
pointcloud_root = Path(config["data"]["kitti"]["location"]) / "pointcloud"
```

Accumulate `point_load`, `sampling`, and `bev_encode` beside existing timings. Add:

```python
"sampling": {
    "contract_version": 1, "rate": sampling_rate, "seed": sampling_seed,
    "raw_points": raw_point_total, "selected_points": selected_point_total,
    "mean_raw_points": raw_point_total / len(frame_ids),
    "mean_selected_points": selected_point_total / len(frame_ids),
    "per_frame": per_frame_point_counts,
},
```

Store each frame as `{"raw_points": N, "selected_points": K}` in
`per_frame_point_counts`. Update `protocol.latency_boundary` to name load,
sampling, BEV encoding, H2D, model, and decode/NMS. Add the exact CLI argv to
`runtime.command`; `main()` passes `[sys.executable, *sys.argv]`, while direct
Python callers may leave `command=None`.

- [ ] **Step 6: Add CLI flags and pass them to `run_evaluation`**

```python
value.add_argument("--sampling-rate", type=float,
                   choices=(0.25, 0.5, 0.75, 1.0), default=1.0)
value.add_argument("--sampling-seed", type=int, default=42)
```

- [ ] **Step 7: Run local checks**

```bash
python tests/test_mobile_bev.py
python tests/test_evaluation_table.py
```

Expected: both scripts complete without traceback.

- [ ] **Step 8: Commit evaluator support**

```bash
git add tools/kitti_training_pipeline/evaluate_kitti_bev.py tests/test_mobile_bev.py
git commit -m "feat: benchmark static point sampling"
```

---

### Task 3: Add the label-informed per-frame quality metric

**Files:**
- Modify: `tools/kitti_training_pipeline/evaluate_kitti_bev.py:130-320`
- Modify: `tests/test_mobile_bev.py:319-350`

**Interfaces:**
- Produces: `frame_quality(predictions: np.ndarray, labels: Sequence[GroundTruth], space: str = "3d") -> float`.
- Semantics: macro class F1 using current IoU thresholds, Moderate filtering, neighbour ignores, and current box IoU.

- [ ] **Step 1: Add a failing quality check**

```python
gt = evaluator.GroundTruth("Car", 0.0, 0, 50.0, 10.0, 0.0, 0.0,
                           4.0, 1.8, 1.6, 0.0)
hit = np.array([[0, 0.9, 10.0, 0.0, 0.0, 1.8, 4.0, 1.6, 0.0]], dtype=np.float32)
false_ped = np.array([[1, 0.8, 20.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0]], dtype=np.float32)
assert evaluator.frame_quality(hit, [gt]) == 1.0
np.testing.assert_allclose(
    evaluator.frame_quality(np.concatenate([hit, false_ped]), [gt]), 2.0 / 3.0)
```

- [ ] **Step 2: Run `python tests/test_mobile_bev.py --metrics`**

Expected: failure because `frame_quality` does not exist.

- [ ] **Step 3: Implement `frame_quality` without changing AP evaluation**

For each class: filter valid/ignored Moderate GT as `evaluate_one` does; sort predictions by score; greedily match valid GT then ignored GT; compute `FN = valid_gt - TP`; compute `2*TP/(2*TP+FP+FN)`. Score an empty class as `1.0` only when it has neither valid GT nor predictions. Return the mean of three class scores.

Keep `evaluate_one` untouched so the trusted AP path cannot regress.

- [ ] **Step 4: Run checks**

```bash
python tests/test_mobile_bev.py --metrics
python tests/test_mobile_bev.py
```

- [ ] **Step 5: Commit the metric**

```bash
git add tools/kitti_training_pipeline/evaluate_kitti_bev.py tests/test_mobile_bev.py
git commit -m "feat: score per-frame sampling quality"
```

---

### Task 4: Build the screening and oracle CLI

**Files:**
- Create: `tools/research/active_point_sampling.py`
- Modify: `tests/test_mobile_bev.py:319-350, 375-415`

**Interfaces:**
- Produces: `split_ids(frame_ids: Sequence[str], seed: int = 42, dev_count: int = 1200) -> tuple[list[str], list[str]]` returning `(policy_train, controller_dev)`.
- Produces: `choose_actions(qualities: np.ndarray, rates: np.ndarray, beta: float) -> np.ndarray`, breaking utility ties toward the lower rate.
- CLI: `baseline --source JSON --epoch 50 --output JSON`; `split --source SPLIT --policy-train SPLIT --controller-dev SPLIT`; `oracle --results JSON... --config JSON --kitti-root DIR --controller-dev SPLIT --output JSON`.

- [ ] **Step 1: Add failing split and action checks**

```python
def check_active_sampling_research():
    sys.path.insert(0, str(ROOT / "tools" / "research"))
    research = importlib.import_module("active_point_sampling")
    ids = [f"{index:06d}" for index in range(20)]
    train_a, dev_a = research.split_ids(ids, seed=42, dev_count=5)
    train_b, dev_b = research.split_ids(list(reversed(ids)), seed=42, dev_count=5)
    assert (train_a, dev_a) == (train_b, dev_b)
    assert len(train_a) == 15 and len(dev_a) == 5
    assert not set(train_a) & set(dev_a)
    assert set(train_a) | set(dev_a) == set(ids)
    rates = np.array([0.25, 0.50, 0.75, 1.00])
    qualities = np.array([[0.8, 0.8, 0.9, 0.9], [0.5, 0.7, 0.8, 0.9]])
    np.testing.assert_array_equal(research.choose_actions(qualities, rates, 0.0), [2, 3])
    np.testing.assert_array_equal(research.choose_actions(qualities, rates, 1.0), [0, 0])
```

Register it in `CHECKS` and add `--active-sampling-research`.

- [ ] **Step 2: Run `python tests/test_mobile_bev.py --active-sampling-research`**

Expected: import failure because the script does not exist.

- [ ] **Step 3: Implement baseline snapshot and split creation**

Use stdlib, NumPy, and existing `common.write_json` only:

```python
def split_ids(frame_ids, seed=42, dev_count=1200):
    unique = set(frame_ids)
    if len(unique) != len(frame_ids):
        raise ValueError("duplicate frame IDs")
    if not 0 < dev_count < len(frame_ids):
        raise ValueError("dev_count must leave both splits non-empty")
    ordered = sorted(unique, key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode("utf-8")).digest())
    return ordered[dev_count:], ordered[:dev_count]
```

Write split lines as `frame_id;kitti`. The baseline command copies the source JSON data, adds `checkpoint_epoch: 50` and `baseline_source`, then writes atomically.

- [ ] **Step 4: Implement result validation**

The oracle command must accept exactly three results per rate; require identical frame IDs, config/split/checkpoint hashes, seed, point counts, and aggregate accuracy across repeats of a rate; use the first prediction archive per rate because CUDA may introduce numerically insignificant box-coordinate differences; take the median across runs separately for `mean_ms`, `p50_ms`, `p95_ms`, and `p99_ms`; and reject missing/unknown sampling metadata.

- [ ] **Step 5: Implement oracle sweep and gates**

Load labels with `evaluate_kitti_bev.load_ground_truth`, build a `[frames, 4]` quality matrix with `frame_quality`, and sweep beta from `0.00` through `1.00` in `0.01` increments:

```python
def choose_actions(qualities, rates, beta):
    utilities = qualities - beta * rates[None, :]
    return np.asarray([
        max(range(len(rates)), key=lambda index: (row[index], -rates[index]))
        for row in utilities
    ], dtype=np.int64)
```

For each unique action vector, assemble per-frame predictions and call unchanged BEV/3D `evaluate_accuracy`. Compare every oracle point with every static point using the spec gates. Write `static_points`, `static_gate`, `oracle_points`, `oracle_gate`, and `provenance`; `oracle_gate.reason` must contain the measured margin. Save frame IDs, the quality matrix, rates, betas, and selected action indices to `<output-stem>.actions.npz`, then record its path and SHA-256 in the gate report.

- [ ] **Step 6: Run checks**

```bash
python tests/test_mobile_bev.py --active-sampling-research
python tests/test_mobile_bev.py
```

- [ ] **Step 7: Commit the CLI**

```bash
git add tools/research/active_point_sampling.py tests/test_mobile_bev.py
git commit -m "feat: add active sampling gate analysis"
```

---

### Task 5: Run S0 and make the go/no-go decision

**Files:**
- Generate: `results/research/active_sampling/baseline_a4_epoch50.json`
- Generate: `splits/kitti/active_sampling_policy_train.txt`
- Generate: `splits/kitti/active_sampling_controller_dev.txt`
- Generate: `results/research/active_sampling/r{rate}_rep{repeat}.json` and matching `.predictions.npz`
- Generate: `results/research/active_sampling/gate_report.json`
- Generate: `results/research/active_sampling/gate_report.actions.npz`

**Interfaces:**
- Consumes checkpoint SHA-256 `e3ff54229ea2feaa30f8f5a88da6555835fbcfa099d02ac49d9400ea510d222d`.
- Produces one auditable decision: stop, finalize static sampling, or authorize a separate policy plan.

- [ ] **Step 1: Snapshot provenance**

```bash
python tools/research/active_point_sampling.py baseline \
  --source /home/duyennh/Downloads/Eval_pro1/evaluation_a4.json \
  --epoch 50 \
  --output results/research/active_sampling/baseline_a4_epoch50.json
```

Verify its config, validation split, and checkpoint hashes equal the spec.

- [ ] **Step 2: Generate controller splits**

```bash
python tools/research/active_point_sampling.py split \
  --source splits/kitti/train.txt \
  --policy-train splits/kitti/active_sampling_policy_train.txt \
  --controller-dev splits/kitti/active_sampling_controller_dev.txt
wc -l splits/kitti/active_sampling_policy_train.txt splits/kitti/active_sampling_controller_dev.txt
sha256sum splits/kitti/active_sampling_policy_train.txt splits/kitti/active_sampling_controller_dev.txt
```

Expected: 4,784 policy-train and 1,200 controller-dev, total 5,984 with no overlap.

- [ ] **Step 3: Verify dense evaluator equivalence**

Run the modified evaluator at rate `1.0` on the first 200 frames of
`splits/kitti/train.txt`, using the same command and environment captured in
Task 2. Require byte-identical prediction arrays and identical AP to
`dense_unmodified_smoke`; require preprocess and E2E mean within 5%. Stop if
this fails.

- [ ] **Step 4: Run the modified dense control on NVIDIA L4**

```bash
python tools/kitti_training_pipeline/evaluate_kitti_bev.py \
  --name a4_sampling_r100_rep0 --backend pytorch \
  --model /content/drive/MyDrive/mobilebev_artifacts/mobilebev_a4_seed42_b16/selected_3d/best.pt \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json \
  --detector-root detector --kitti-root /content/KITTI_DATASET \
  --split splits/kitti/active_sampling_controller_dev.txt \
  --sampling-rate 1.0 --sampling-seed 42 \
  --output results/research/active_sampling/r100_rep0.json
```

This is the first full controller-dev timing run; the equivalence check already
passed on the fixed 200-frame smoke control.

- [ ] **Step 5: Run the remaining 11 evaluations**

Reuse Step 4 with these exact rate orders and output stems:

| Repeat | Ordered rates | Output stems |
|---|---|---|
| 0 | `0.75, 0.50, 0.25` | `r075_rep0, r050_rep0, r025_rep0` |
| 1 | `0.25, 0.50, 0.75, 1.00` | `r025_rep1, r050_rep1, r075_rep1, r100_rep1` |
| 2 | `0.50, 1.00, 0.25, 0.75` | `r050_rep2, r100_rep2, r025_rep2, r075_rep2` |

Confirm all 12 JSON files report 1,200 frames, seed 42, matching hashes, and `status: ok`.

- [ ] **Step 6: Compute the gates**

```bash
python tools/research/active_point_sampling.py oracle \
  --results results/research/active_sampling/r100_rep0.json \
            results/research/active_sampling/r075_rep0.json \
            results/research/active_sampling/r050_rep0.json \
            results/research/active_sampling/r025_rep0.json \
            results/research/active_sampling/r100_rep1.json \
            results/research/active_sampling/r075_rep1.json \
            results/research/active_sampling/r050_rep1.json \
            results/research/active_sampling/r025_rep1.json \
            results/research/active_sampling/r100_rep2.json \
            results/research/active_sampling/r075_rep2.json \
            results/research/active_sampling/r050_rep2.json \
            results/research/active_sampling/r025_rep2.json \
  --config configs/kitti/mobilebev/a4_rich8_sgfpn_center3d.json \
  --kitti-root /content/KITTI_DATASET \
  --controller-dev splits/kitti/active_sampling_controller_dev.txt \
  --output results/research/active_sampling/gate_report.json
```

- [ ] **Step 7: Apply the stop rule**

- If static fails: report the negative result; add no policy code.
- If static passes but oracle fails: freeze the best static rate, then compare dense/static once on sealed validation.
- If both pass: stop this plan, write a contextual-policy plan, and keep validation sealed.

- [ ] **Step 8: Commit only reproducibility manifests**

```bash
git add results/research/active_sampling/baseline_a4_epoch50.json \
        splits/kitti/active_sampling_policy_train.txt \
        splits/kitti/active_sampling_controller_dev.txt
git commit -m "research: lock active sampling manifests"
```

Keep timing JSON and prediction NPZ files in experiment storage; they are run outputs.

## Completion Criteria

- Local checks pass without new dependencies.
- Dense evaluator equivalence passes before sampled results are interpreted.
- The report contains four static points, median-of-three timing statistics, safety deltas, oracle points, the action-cache hash, and provenance hashes.
- Work stops at the first failed gate.
- No contextual policy or RL code exists at the end of this plan.
