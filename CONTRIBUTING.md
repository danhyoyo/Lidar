# Contributing Guidelines

Thank you for your interest in contributing to the **LiDAR BEV 3D Object Detection** project (featuring **MobilePixorNeXt** and **MobilePIXOR**), developed at **FPT University Can Tho Campus**!

We welcome community feedback, issue reports, documentation improvements, and upstream contributions that help make this real-time 3D object detection framework even more robust and accessible for researchers.

---

## 1. Important Licensing & Contribution Notice

> [!IMPORTANT]
> This project is distributed under the **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License (CC BY-NC-ND 4.0)** with an explicit software addendum (see [`LICENSE`](file:///home/duyennh/AI_projects/research_lidar/Lidar/LICENSE)).
>
> - **Public Redistribution**: External redistribution of modified forks or derivative versions is strictly prohibited.
> - **Upstream Contributions**: By submitting a Pull Request to this upstream repository, you agree that your submitted contributions will be licensed under the same terms and integrated into the official codebase maintained by the project author (`DuyenNH2401`) and FPT University Can Tho Campus.

---

## 2. Code of Conduct & Getting Help

- Be respectful, constructive, and collaborative in discussions, issues, and code reviews.
- If you encounter a bug, have a question, or wish to propose an enhancement, please open a GitHub Issue before initiating large changes.

---

## 3. How to Contribute

### 3.1 Reporting Bugs

When reporting an issue, please include:
1. **Clear description**: What happened vs. what you expected to happen.
2. **Reproduction steps**: Minimal commands or scripts to reproduce the issue.
3. **System environment**:
   - OS & version (e.g., Ubuntu 22.04 LTS)
   - Python version (e.g., Python 3.10 / 3.11)
   - PyTorch version and CUDA release (e.g., PyTorch 2.1+, CUDA 12.x)
   - GPU model (e.g., NVIDIA RTX 5060 Ti / RTX 4090 / Colab T4)
4. **Full error traceback**: Wrapped in a markdown code block.

### 3.2 Proposing Enhancements

For new backbones, loss functions, or deployment tools:
- Open a GitHub Discussion or Issue describing the proposed feature and academic motivation.
- Highlight expected latency / AP trade-offs and compatibility with existing configuration files.

---

## 4. Development Setup

### 4.1 Prerequisites

- Linux (Ubuntu 20.04/22.04/24.04 recommended)
- Python 3.10 or 3.11
- NVIDIA GPU with CUDA 12.x support (CPU is supported for functional smoke testing)

### 4.2 Setting Up the Environment

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danhyoyo/Lidar.git
   cd Lidar
   ```

2. **Create and activate a virtual environment**:
   ```bash
   conda create -n lidar_env python=3.10 -y
   conda activate lidar_env
   ```

3. **Install PyTorch with CUDA support**:
   ```bash
   # Example for CUDA 12.1+ / 12.8:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install repository dependencies**:
   ```bash
   pip install -r requirements-kitti.txt
   pip install pytest
   ```

> [!TIP]
> If you have ROS / ROS 2 installed on your machine, external ROS Python paths may cause test runner conflicts. You can isolate your testing session by executing:
> ```bash
> PYTHONPATH=. python -m pytest -p no:launch_testing tests/ -q
> ```

---

## 5. Coding Standards & Conventions

To maintain high code quality and consistency with the existing codebase:

- **Indentation**: Use 4 spaces for indentation (no tabs).
- **Naming Conventions**:
  - `snake_case` for modules, functions, methods, and variables.
  - `PascalCase` for classes (e.g., `MobilePixorNeXt`, `Rich8Encoding`).
  - `UPPER_SNAKE_CASE` for constants and default flags.
- **Backbone & Loss Registry Pattern**:
  - All backbones must be registered via the model registry in `detector/core/models/`.
  - All loss strategies must be registered in `detector/core/losses/`.
  - Do not hardcode custom logic outside the modular registry pattern.
- **Configuration Compatibility**:
  - Preserve existing configuration keys in `configs/config.json`.
  - New parameters must include sensible defaults to maintain backwards compatibility with existing experiments and Colab notebooks.
- **Documentation**:
  - Preserve all existing comments and docstrings.
  - Add docstrings to newly introduced classes, loss functions, and dataset transforms.

---

## 6. Testing & Quality Assurance

All pull requests must pass the automated test suite before being considered for merge.

### 6.1 Running Tests

Execute pytest from the repository root:

```bash
# Run the entire test suite
PYTHONPATH=. pytest tests/ -q

# Run specific test modules
PYTHONPATH=. pytest tests/test_ci_smoke.py -v
PYTHONPATH=. pytest tests/test_model_registry.py -v
PYTHONPATH=. pytest tests/test_optimizer_scheduler.py -v
```

### 6.2 Writing New Tests

- Place tests in the `tests/` directory following the naming pattern `test_<feature>.py`.
- Prefer **small synthetic tensors / geometry** over loading large external KITTI point clouds to ensure fast and deterministic execution in CI.
- Verify tensor shapes, gradient flow, and edge cases (e.g., empty point clouds, zero detections).

### 6.3 Checking Syntax & Bytecode

Verify that all Python files compile cleanly without syntax errors:

```bash
python -m compileall -q detector/ tools/ tests/
```

---

## 7. Data & Experiment Hygiene

- **Do Not Commit Datasets or Checkpoints**:
  - Raw datasets (`data/kitti/raw`), processed BEV maps (`data/kitti/processed`), and trained weights/checkpoints (`artifacts/kitti/`) are explicitly excluded in `.gitignore`. Never commit them to Git.
- **Disjoint Train / Validation Splits**:
  - Ground Truth databases (`build_gt_database.py`) must strictly be sampled from the training split (`splits/kitti/train.txt`) to prevent data leakage.
- **Reproducibility**:
  - Always record seed, split configuration, and save `config.resolved.json` alongside model evaluation results.

---

## 8. Commit Message Conventions

We adhere to the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat(scope): add new feature or capability`
- `fix(scope): fix a bug or unexpected behavior`
- `docs(scope): update documentation, guides, or docstrings`
- `refactor(scope): refactor code without changing external behavior`
- `test(scope): add or improve unit/regression tests`
- `ci(scope): update CI workflows or build scripts`
- `perf(scope): performance optimization or latency improvement`

**Example:**
```bash
git commit -m "feat(backbone): integrate reparameterized 7x7 depthwise fusion block"
git commit -m "fix(eval): correct rotated BEV bounding box IoU calculation in edge case"
```

---

## 9. Pull Request Checklist

Before submitting your PR, please verify:

- [ ] Branch is based on the latest `main` branch.
- [ ] Code strictly follows the 4-space indentation and naming conventions.
- [ ] All unit and regression tests pass locally (`pytest tests/ -q`).
- [ ] Bytecode compilation succeeds (`python -m compileall -q detector/ tools/ tests/`).
- [ ] No temporary files, datasets, or `.pt` checkpoints are tracked in Git.
- [ ] Commit history is clean and follows conventional commit messages.
- [ ] PR description details the changes, motivation, and hardware/software verification results.

---

## 10. Contact & Community

For research inquiries, academic collaboration, or institutional questions:
- **Author**: DuyenNH2401 (`duyennhce200017@gmail.com`)
- **Institution**: FPT University Can Tho Campus (Trường Đại học FPT Cần Thơ)
