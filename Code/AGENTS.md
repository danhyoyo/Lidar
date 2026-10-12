# Repository Guidelines

## Project Structure & Module Organization

- `detector/core/` contains BEV encoding, datasets and augmentors, model backbones/heads, and loss strategies; `detector/postprocess.py` decodes predictions and applies NMS.
- `tools/kitti_training_pipeline/` provides data preparation, training, evaluation, ONNX export, and TensorRT deployment. `tools/benchmarks/` and `tools/visualization/` contain profiling and inspection utilities.
- `configs/config.json` is the master configuration; augmentation recipes and experiment overrides live in `configs/augmentation/` and `configs/experiments/`.
- `tests/` contains regression and integration tests, with reference JSON in `tests/fixtures/`. `docs/` and `diagrams/` document architecture and experiments. The root notebook supports Colab workflows.

## Build, Test, and Development Commands

Run commands from the repository root using Python 3.10 and a suitable PyTorch installation:

```bash
python3 -m pip install -r requirements-kitti.txt
python3 -m pip install pytest
python3 -m pytest tests/ -q
python3 -m pytest tests/test_model_registry.py -q
python3 tools/kitti_training_pipeline/train.py --config configs/config.json --detector-root detector --output-root artifacts/kitti
```

These install runtime/test dependencies, run the complete or focused suite, and start training after data preparation. Follow `tools/kitti_training_pipeline/README.md` for `prepare_kitti.py` and `evaluate_kitti_bev.py` commands. ONNX/TensorRT exports use separate scripts; TensorRT is optional.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for Python modules/functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Match nearby code and preserve existing configuration keys. Keep backbone and loss registration in their respective registries. No repository-wide formatter or linter configuration is present.

## Testing Guidelines

Tests use pytest and `unittest.TestCase`; name files `test_<feature>.py` and cases `test_<behavior>`. Add regression coverage for changed tensor shapes, geometry, losses, configuration compatibility, and augmentation. Prefer small synthetic inputs over external KITTI data. No numerical coverage threshold is configured. Run relevant tests before the full suite and report hardware-dependent skips.

## Commit & Pull Request Guidelines

History commonly uses `fix(scope): description`, `feat(scope): description`, and `docs(scope): description`, alongside plain imperative summaries. Keep commits focused. PRs should explain the behavior change, affected configurations, and validation commands/results; link related issues when applicable. For model experiments, record seed, split, resolved configuration, AP, and latency measurement conditions.

## Data & Experiment Hygiene

Keep datasets, checkpoints, and generated results out of commits; use ignored `data/` and `artifacts/` directories. Keep train/validation manifests disjoint and build GT databases from training IDs only. Use distinct run names for incompatible configurations and preserve `config.resolved.json` for reproducibility.
