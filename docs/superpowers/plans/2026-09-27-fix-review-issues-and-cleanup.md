# Review Issues Remediation & Evaluation Table Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the evaluation mode state leak and dynamic device re-allocation in `TemperatureSoftmaxUncertainty`, eliminate test suite `UserWarning`s, remove deprecated `evaluation_table` code as requested, and ensure clean asset tracking.

**Architecture:** 
1. Protect running loss scale calibration with `if self.training:` guard to guarantee evaluation passes are side-effect free.
2. Replace dangerous `self.to()` in `forward()` with an explicit device check to prevent parameter disconnection from optimizers and preserve compatibility with `torch.compile`/DDP.
3. Remove unused `evaluation_table.py` script and its test file `test_evaluation_table.py` per user instruction.
4. Eliminate `lr_scheduler.step()` warnings in `test_optimizer_scheduler.py` and track diagram assets.

**Tech Stack:** PyTorch 2.x, Python 3.12+, Pytest, NumPy.

**Spec:** Code review findings from conversation step b344cd80-9b05-4c61-b39d-3d4840641e0c and user instruction to deprecate evaluation table creation.

## Global Constraints

- Preserve 100% backward compatibility for existing checkpoints (`state_dict` loading).
- Maintain strict numerical stability under FP32, FP16, and BF16 autocast.
- Zero test failures and clean warnings across the entire test suite.
- Pure POSIX paths and atomic file operations.

---

### Task 1: Prevent EMA Running Loss Scale Leak in Evaluation Mode (P0)

**Files:**
- Modify: `detector/core/losses/uncertainty_weighting.py:106-113`
- Test: `tests/test_uncertainty_weighting.py`

**Interfaces:**
- Consumes: `self.training` boolean flag from `nn.Module`.
- Produces: `TemperatureSoftmaxUncertainty.forward()` that only mutates `running_loss_means` and `initialized` when `self.training is True`.

- [ ] **Step 1: Write the failing test for evaluation mode immutability**

Add test method `test_eval_mode_freezes_running_loss_means` to `tests/test_uncertainty_weighting.py`:

```python
    def test_eval_mode_freezes_running_loss_means(self):
        weighting = TemperatureSoftmaxUncertainty(
            task_names=["cls", "offset", "size", "yaw", "geo"],
            temperature=2.0,
            ema_momentum=0.9,
        )
        # Step in train mode to initialize buffer
        weighting.train()
        train_losses = {k: torch.tensor(1.0) for k in weighting.task_names}
        weighting(train_losses)
        initial_means = weighting.running_loss_means.clone()

        # Switch to eval mode
        weighting.eval()
        eval_losses = {k: torch.tensor(50.0) for k in weighting.task_names}
        weighting(eval_losses)

        # Running means must remain strictly unchanged during eval
        self.assertTrue(
            torch.allclose(weighting.running_loss_means, initial_means),
            f"Evaluation mode mutated running loss means: {weighting.running_loss_means} != {initial_means}",
        )

        # Switch back to train mode and verify updates resume
        weighting.train()
        weighting(eval_losses)
        self.assertFalse(
            torch.allclose(weighting.running_loss_means, initial_means),
            "Training mode failed to update running loss means.",
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_uncertainty_weighting.py::TestTemperatureSoftmaxUncertainty::test_eval_mode_freezes_running_loss_means -v
```
Expected: FAIL with `Evaluation mode mutated running loss means`.

- [ ] **Step 3: Implement `if self.training:` guard in `TemperatureSoftmaxUncertainty.forward`**

In `detector/core/losses/uncertainty_weighting.py`, update lines 106-113:

```python
        if self.training:
            with torch.no_grad():
                detached = stacked_losses.detach().clamp_min(1e-4)
                if not self.initialized:
                    self.running_loss_means.copy_(detached)
                    self.initialized.fill_(True)
                else:
                    self.running_loss_means.lerp_(detached, 1.0 - self.ema_momentum)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_uncertainty_weighting.py -v
```
Expected: PASS (all tests in `test_uncertainty_weighting.py` pass).

- [ ] **Step 5: Commit changes**

```bash
git add detector/core/losses/uncertainty_weighting.py tests/test_uncertainty_weighting.py
git commit -m "fix(loss): freeze uncertainty running loss calibration during evaluation mode"
```

---

### Task 2: Remove `self.to()` in `forward()` and Enforce Explicit Device Placement (P1)

**Files:**
- Modify: `detector/core/losses/uncertainty_weighting.py:89-92`
- Test: `tests/test_uncertainty_weighting.py`

**Interfaces:**
- Consumes: `task_losses` dictionary containing scalar tensors on target device.
- Produces: `RuntimeError` if loss module parameters are on a different device from input losses, instead of mutating module device at runtime and severing optimizer parameter bindings.

- [ ] **Step 1: Write the failing test for device mismatch verification**

Add test method `test_device_mismatch_raises_runtime_error` to `tests/test_uncertainty_weighting.py`:

```python
    def test_device_mismatch_raises_runtime_error(self):
        weighting = TemperatureSoftmaxUncertainty(task_names=["cls", "offset", "size", "yaw", "geo"])
        # Mock parameter on a dummy device by monkeypatching or using mismatching tensor device
        # If CUDA is available, put weighting on CPU and pass CUDA loss, or vice versa
        if torch.cuda.is_available():
            weighting.cpu()
            cuda_losses = {k: torch.tensor(1.0, device="cuda") for k in weighting.task_names}
            with self.assertRaises(RuntimeError) as ctx:
                weighting(cuda_losses)
            self.assertIn("Device mismatch in TemperatureSoftmaxUncertainty", str(ctx.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_uncertainty_weighting.py::TestTemperatureSoftmaxUncertainty::test_device_mismatch_raises_runtime_error -v
```
Expected: FAIL (currently it calls `self.to()` instead of raising `RuntimeError`).

- [ ] **Step 3: Replace `self.to()` with explicit device validation**

In `detector/core/losses/uncertainty_weighting.py`, replace lines 89-92:

```python
        target_device = next(iter(task_losses.values())).device
        if self.log_scales.device != target_device:
            raise RuntimeError(
                f"Device mismatch in TemperatureSoftmaxUncertainty: log_scales is on {self.log_scales.device} "
                f"while input losses are on {target_device}. Ensure criterion is moved to the target device "
                "via criterion.to(device) before building the optimizer and starting training."
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_uncertainty_weighting.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add detector/core/losses/uncertainty_weighting.py tests/test_uncertainty_weighting.py
git commit -m "fix(loss): eliminate silent runtime self.to re-allocation and validate device alignment"
```

---

### Task 3: Remove Deprecated `evaluation_table` Code and Tests

**Files:**
- Delete: `tests/test_evaluation_table.py`
- Delete: `tools/kitti_training_pipeline/evaluation_table.py`

**Interfaces:**
- Consumes: None (independent reporting script).
- Produces: Cleaner codebase with zero dead table generation code.

- [ ] **Step 1: Verify no dependencies exist in training or evaluation pipeline**

Run check across `tools/kitti_training_pipeline`:
```bash
git grep -n "evaluation_table" tools/
```
Expected: Only `tools/kitti_training_pipeline/evaluation_table.py` itself.

- [ ] **Step 2: Delete `evaluation_table.py` and `test_evaluation_table.py`**

Remove the files:
```bash
rm tests/test_evaluation_table.py
rm tools/kitti_training_pipeline/evaluation_table.py
```

- [ ] **Step 3: Run the test suite to ensure clean collection**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/ -v
```
Expected: All tests collect and pass without `test_evaluation_table.py`.

- [ ] **Step 4: Commit deletion**

```bash
git rm tests/test_evaluation_table.py tools/kitti_training_pipeline/evaluation_table.py
git commit -m "refactor(eval): remove obsolete evaluation_table tool and test"
```

---

### Task 4: Fix Scheduler Test Warning and Track Architecture Diagram Assets

**Files:**
- Modify: `tests/test_optimizer_scheduler.py:89-102, 131-134`
- Untracked: `diagrams/`, `tools/generate_diagrams.py`

- [ ] **Step 1: Update `tests/test_optimizer_scheduler.py` to step optimizer before scheduler**

In `tests/test_optimizer_scheduler.py`:
In `test_build_scheduler_cosine_warmup`:
```python
        # Step through 5 warmup epochs
        for ep in range(5):
            optimizer.step()
            scheduler.step()

        # At end of warmup (milestone epoch 5), LR reaches peak 0.002
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.002, places=5)

        # Step remaining 95 epochs
        for ep in range(95):
            optimizer.step()
            scheduler.step()
```

In `test_build_scheduler_state_dict_save_load`:
```python
        for _ in range(10):
            optimizer1.step()
            scheduler1.step()
```

- [ ] **Step 2: Run `test_optimizer_scheduler.py` and check for warnings**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/test_optimizer_scheduler.py -v
```
Expected: PASS with 0 `UserWarning`s.

- [ ] **Step 3: Track `diagrams/` and `tools/generate_diagrams.py`**

```bash
git add diagrams/ tools/generate_diagrams.py
git commit -m "docs(diagrams): track high-resolution architecture diagrams and generator script"
```

- [ ] **Step 4: Commit scheduler test improvements**

```bash
git add tests/test_optimizer_scheduler.py
git commit -m "test(train): step optimizer before scheduler in tests to eliminate PyTorch UserWarning"
```

---

### Task 5: Complete Suite Verification & Regression Check

**Files:**
- All tests in `tests/`

- [ ] **Step 1: Run entire test suite cleanly**

Run:
```bash
PYTHONPATH="" /home/duyennh/miniconda3/envs/AI_env/bin/python -m pytest -p no:launch_testing -p no:launch_testing_ros tests/ -v
```
Expected: 100% tests pass (71 passed, 0 failures, 0 errors).

- [ ] **Step 2: Check `git status` and `git diff`**

Run:
```bash
git status
```
Expected: Working tree clean.

---
