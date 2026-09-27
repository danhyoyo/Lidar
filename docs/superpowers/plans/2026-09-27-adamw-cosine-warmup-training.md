# AdamW and Cosine Annealing with Warmup Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Modernize the KITTI 3D LiDAR training pipeline for 100-epoch runs by integrating AdamW with decoupled bias/norm weight decay, Linear Warmup + Cosine Annealing learning rate schedule, TF32 acceleration, and synchronized JSON configurations.

**Architecture:** Refactor optimizer and scheduler instantiation in `tools/kitti_training_pipeline/train.py` into dedicated, unit-tested builder functions (`build_optimizer` and `build_scheduler`). Enable TensorFloat32 (`tf32`) matmul precision globally for CUDA devices to accelerate non-autocast FP32 kernels alongside BF16 mixed precision. Update training configs to specify 100 epochs, AdamW, and 5-epoch warmup with Cosine Annealing.

**Tech Stack:** PyTorch 2.x, PyTorch AMP (BF16), torch.optim (AdamW), torch.optim.lr_scheduler (LinearLR, CosineAnnealingLR, SequentialLR), pytest.

**Spec:** User requirement: "nhưng tôi dùng bf16 thì có nên bật tf32 không? thế hãy sửa code lại cho 100 epoch thay thành adamW và dùng cosine annealing + warnup cho tôi đi".

## Global Constraints

- Python environment for tests: `/home/duyennh/miniconda3/envs/AI_env/bin/python` with `PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline`.
- Pytest execution flags: must pass `-p no:launch_testing -p no:launch_testing_ros`.
- Maintain 100% backward compatibility for existing checkpoints (`last.pt`) and legacy configs specifying `MultiStepLR`.
- Do not apply weight decay to 1D parameters (biases, normalization scale/shift) or criterion parameters (loss log-scales).
- Preserve exact reproducibility and deterministic random seeds.

---

### Task 1: Add TensorFloat32 Activation and Modular `build_optimizer` with AdamW

**Files:**
- Modify: `tools/kitti_training_pipeline/train.py:260-280`
- Create: `tests/test_optimizer_scheduler.py`

**Interfaces:**
- Consumes: `model: nn.Module`, `criterion: nn.Module`, `config: dict`
- Produces: `build_optimizer(model, criterion, config) -> torch.optim.Optimizer`

- [x] **Step 1: Write the failing test for `build_optimizer`**

Create `tests/test_optimizer_scheduler.py` with test cases verifying:
1. `build_optimizer` defaults to `AdamW` with correct parameter grouping (2D weights get `weight_decay`, biases and 1D norm params get `0.0`, criterion params get `0.0`).
2. Custom optimizer options (e.g. `"optimizer": "adam"`) remain supported for backward compatibility.

```python
import unittest
import torch
import torch.nn as nn
from tools.kitti_training_pipeline.train import build_optimizer


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 16, 3)
        self.bn = nn.BatchNorm2d(16)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        return self.fc(self.bn(self.conv(x)).mean(dim=[-2, -1]))


class DummyCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([0.5]))


class TestOptimizerBuilder(unittest.TestCase):
    def test_build_optimizer_adamw_groups(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.002,
                "weight_decay": 0.0001,
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(len(optimizer.param_groups), 3)

        decay_group = optimizer.param_groups[0]
        no_decay_group = optimizer.param_groups[1]
        crit_group = optimizer.param_groups[2]

        self.assertEqual(decay_group["weight_decay"], 0.0001)
        self.assertEqual(no_decay_group["weight_decay"], 0.0)
        self.assertEqual(crit_group["weight_decay"], 0.0)
        self.assertEqual(decay_group["lr"], 0.002)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_optimizer_scheduler.py -k test_build_optimizer_adamw_groups -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL with `ImportError: cannot import name 'build_optimizer'`

- [x] **Step 3: Implement TF32 setup and `build_optimizer` in `train.py`**

In `tools/kitti_training_pipeline/train.py`:
1. Add `set_matmul_precision()` helper:
   ```python
   def configure_matmul_precision() -> None:
       if torch.cuda.is_available():
           torch.set_float32_matmul_precision("high")
   ```
2. Implement `build_optimizer`:
   ```python
   def build_optimizer(
       model: torch.nn.Module,
       criterion: torch.nn.Module,
       config: dict,
   ) -> torch.optim.Optimizer:
       train_cfg = config.get("train", {})
       opt_type = train_cfg.get("optimizer", "adamw").lower()
       lr = float(train_cfg["learning_rate"])
       weight_decay = float(train_cfg.get("weight_decay", 0.0001))

       decay_params = []
       no_decay_params = []
       for name, param in model.named_parameters():
           if not param.requires_grad:
               continue
           if param.ndim <= 1 or name.endswith(".bias"):
               no_decay_params.append(param)
           else:
               decay_params.append(param)

       groups = [
           {"params": decay_params, "weight_decay": weight_decay},
           {"params": no_decay_params, "weight_decay": 0.0},
       ]

       crit_params = [p for p in criterion.parameters() if p.requires_grad]
       if crit_params:
           groups.append({"params": crit_params, "weight_decay": 0.0})

       if opt_type == "adam":
           return torch.optim.Adam(groups, lr=lr)
       elif opt_type == "adamw":
           return torch.optim.AdamW(groups, lr=lr)
       else:
           raise ValueError(f"Unsupported optimizer type: {opt_type}")
   ```
3. Call `configure_matmul_precision()` and replace inline optimizer instantiation with `optimizer = build_optimizer(model, criterion, config)` in `main()`.

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_optimizer_scheduler.py -k test_build_optimizer_adamw_groups -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS

- [x] **Step 5: Commit**

```bash
git add tools/kitti_training_pipeline/train.py tests/test_optimizer_scheduler.py
git commit -m "feat(pipeline): add TF32 precision setup and modular AdamW optimizer builder"
```

---

### Task 2: Implement Modular `build_scheduler` with Linear Warmup + Cosine Annealing

**Files:**
- Modify: `tools/kitti_training_pipeline/train.py:278-295`
- Modify: `tests/test_optimizer_scheduler.py`

**Interfaces:**
- Consumes: `optimizer: torch.optim.Optimizer`, `config: dict`, `epochs: int`
- Produces: `build_scheduler(optimizer, config, epochs) -> torch.optim.lr_scheduler.LRScheduler`

- [x] **Step 1: Write failing tests for `build_scheduler`**

Add tests to `tests/test_optimizer_scheduler.py`:
1. Verify `build_scheduler` with `"scheduler": "cosine"` creates a warmup + cosine schedule whose learning rate increases linearly for `warmup_epochs` and then decreases following cosine decay towards `min_lr`.
2. Verify backward compatibility when `"scheduler": "multistep"` is requested.
3. Verify checkpoint serialization and restoration compatibility with `scheduler.state_dict()`.

```python
    def test_build_scheduler_cosine_warmup(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "optimizer": "adamw",
                "learning_rate": 0.002,
                "scheduler": "cosine",
                "warmup_epochs": 5,
                "min_lr": 1e-6,
            }
        }
        epochs = 100
        optimizer = build_optimizer(model, criterion, config)
        scheduler = build_scheduler(optimizer, config, epochs)

        # Initial LR before any step
        self.assertLess(optimizer.param_groups[0]["lr"], 0.002)

        # Step 5 warmup epochs
        for ep in range(5):
            scheduler.step()

        # At end of warmup, LR reaches peak
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.002, places=5)

        # Step remaining 95 epochs
        for ep in range(95):
            scheduler.step()

        # At end of training, LR reaches min_lr
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-6, places=5)

    def test_build_scheduler_multistep_compatibility(self):
        model = DummyModel()
        criterion = DummyCriterion()
        config = {
            "train": {
                "learning_rate": 0.005,
                "scheduler": "multistep",
                "lr_decay_at": [65, 85],
            }
        }
        optimizer = build_optimizer(model, criterion, config)
        scheduler = build_scheduler(optimizer, config, 100)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
```

- [x] **Step 2: Run test to verify it fails**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_optimizer_scheduler.py -k test_build_scheduler_cosine_warmup -p no:launch_testing -p no:launch_testing_ros
```
Expected: FAIL with `ImportError: cannot import name 'build_scheduler'`

- [x] **Step 3: Implement `build_scheduler` in `train.py`**

In `tools/kitti_training_pipeline/train.py`:
```python
def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    train_cfg = config.get("train", {})
    sched_type = train_cfg.get("scheduler", "cosine").lower()
    base_lr = float(train_cfg["learning_rate"])

    if sched_type == "cosine":
        warmup_epochs = int(train_cfg.get("warmup_epochs", 5))
        min_lr = float(train_cfg.get("min_lr", 1e-6))
        if warmup_epochs > 0:
            start_factor = min(1.0, max(1e-4, min_lr / base_lr))
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, epochs - warmup_epochs),
                eta_min=min_lr,
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
                eta_min=min_lr,
            )
    elif sched_type == "multistep":
        milestones = list(train_cfg.get("lr_decay_at", [65, 85]))
        gamma = float(train_cfg.get("lr_decay_gamma", 0.1))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")
```
In `main()` of `train.py`, replace:
```python
    scheduler = torch.optim.lr_scheduler.MultiStepLR(...)
```
with:
```python
    scheduler = build_scheduler(optimizer, config, epochs)
```

- [x] **Step 4: Run test to verify it passes**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_optimizer_scheduler.py -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS (all tests pass)

- [x] **Step 5: Commit**

```bash
git add tools/kitti_training_pipeline/train.py tests/test_optimizer_scheduler.py
git commit -m "feat(pipeline): add modular Cosine Annealing with Warmup scheduler builder"
```

---

### Task 3: Update Config Files for 100 Epochs, AdamW, and Cosine Annealing

**Files:**
- Modify: `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`
- Modify: `configs/kitti/backbone_branch/kitti_mobilepixornext_litemla.json`

**Interfaces:**
- Consumes: JSON config schema
- Produces: Updated JSON configs with `"epochs": 100`, `"optimizer": "adamw"`, `"scheduler": "cosine"`, `"warmup_epochs": 5`, `"min_lr": 1e-6`, `"learning_rate": 0.002`

- [x] **Step 1: Update `configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json`**

Update `train` section:
```json
  "train": {
    "accumulation_steps": 1,
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "optimizer": "adamw",
    "physical_batch_size": 16,
    "precision": "bf16",
    "save_every": 5,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "weight_decay": 0.0001
  }
```

- [x] **Step 2: Update `configs/kitti/backbone_branch/kitti_mobilepixornext_litemla.json`**

Update `train` section similarly:
```json
  "train": {
    "accumulation_steps": 1,
    "data": "splits/kitti/train.txt",
    "epochs": 100,
    "learning_rate": 0.002,
    "min_lr": 1e-06,
    "optimizer": "adamw",
    "physical_batch_size": 16,
    "precision": "bf16",
    "save_every": 5,
    "scheduler": "cosine",
    "warmup_epochs": 5,
    "weight_decay": 0.0001
  }
```

- [x] **Step 3: Run pipeline tests to verify config loads and runs properly**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_training_pipeline_oga.py -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS

- [x] **Step 4: Commit**

```bash
git add configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json configs/kitti/backbone_branch/kitti_mobilepixornext_litemla.json
git commit -m "feat(config): configure 100 epochs, AdamW and Cosine Annealing with Warmup"
```

---

### Task 4: Update Google Colab Standard Training Notebook

**Files:**
- Modify: `3D_Lidar_Object_Detection_Notebook_standard.ipynb`

**Interfaces:**
- Consumes: Colab interactive notebook
- Produces: Notebook with `EPOCHS = 100`, TF32 precision configured, and updated variant defaults

- [x] **Step 1: Update Configuration cell in notebook**

In `3D_Lidar_Object_Detection_Notebook_standard.ipynb`:
1. In cell 1 (Configuration):
   Set `EPOCHS = 100` (instead of `EPOCHS = 50`).
   Set `PHYSICAL_BATCH_SIZE = 16` and `ACCUMULATION_STEPS = 1`.
   Add TF32 precision call:
   ```python
   if torch.cuda.is_available():
       torch.set_float32_matmul_precision("high")
   ```
2. In cell 2 (Dependencies and variant):
   Ensure `VARIANT_CONFIGS` includes the new dedicated path:
   `"MOBILEPIXORNEXT_OGA": "configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_litemla_oga.json"` alongside `"MOBILEPIXORNEXT_LITEMLA"`.

- [x] **Step 2: Run notebook validation test**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest tests/test_standard_training_notebook.py -p no:launch_testing -p no:launch_testing_ros
```
Expected: PASS

- [x] **Step 3: Commit**

```bash
git add 3D_Lidar_Object_Detection_Notebook_standard.ipynb
git commit -m "feat(notebook): set 100 epochs, TF32 precision, and register MOBILEPIXORNEXT_OGA variant"
```

---

### Task 5: Full Test Suite Verification and Remote Synchronization

**Files:**
- All modified files

- [x] **Step 1: Run complete pytest suite**

Run:
```bash
PYTHONPATH=detector:detector/core/datasets:tools/kitti_training_pipeline /home/duyennh/miniconda3/envs/AI_env/bin/pytest -p no:launch_testing -p no:launch_testing_ros
```
Expected: 100% tests pass (all 65+ tests pass with 0 failures).

- [x] **Step 2: Push changes to remote feature branch**

```bash
git push origin feature/mobilepixornext-oga-loss
```
Expected: Branch `feature/mobilepixornext-oga-loss` successfully pushed to GitHub.
