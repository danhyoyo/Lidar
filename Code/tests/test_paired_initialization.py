"""Paired random state, provenance and restoration must remain reproducible."""

import copy
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/kitti_training_pipeline"), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets")]

from common import build_model, checkpoint_identity, validate_evaluation_checkpoint, validate_deployment_config
from paired_initialization import build_initialized_model, paired_policy, state_dict_digest
from test_grouped_training import configured_model
import train


def config_for(name="hist22", grouped=False):
    config = configured_model(grouped=grouped)
    config["seed"] = 42
    config["initialization"] = {"mode": "hist14_hist22_paired_random", "version": 1, "seed": 42}
    config["data"]["bev_encoding"] = {"name": name}
    config["data"]["kitti"]["geometry"].update(x_min=0., x_max=4.8, x_res=.1,
        y_min=-1.6, y_max=1.6, y_res=.1)
    return config


def draws():
    return random.random(), np.random.random(), torch.rand(5)


def assert_draws_equal(left, right):
    assert left[:2] == right[:2]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def test_pair_copies_every_tensor_and_bn_buffer_without_consuming_caller_rng():
    train.seed_everything(77)
    expected = draws()
    train.seed_everything(77)
    baseline, pa = build_initialized_model(config_for("hist14"))
    candidate, pb = build_initialized_model(config_for())
    assert_draws_equal(draws(), expected)
    assert pa == pb
    a, b = baseline.state_dict(), candidate.state_dict()
    stem = pa["input_weight_key"]
    assert stem == "backbone.stem.0.weight"
    for key in a:
        torch.testing.assert_close(b[key][:, :14] if key == stem else b[key], a[key], rtol=0, atol=0)
    assert not b[stem][:, 14:].any()
    assert pa["source_state_sha256"] == state_dict_digest(a)
    assert state_dict_digest(dict(reversed(list(a.items())))) == pa["source_state_sha256"]


@pytest.mark.parametrize("field,value", [("version", True), ("version", 1.), ("version", 2),
    ("seed", True), ("seed", 42.), ("seed", 43), ("checkpoint", "trained.pt")])
def test_policy_rejects_malformed_or_checkpoint_transfer_inputs(field, value):
    config = config_for()
    config["initialization"][field] = value
    with pytest.raises(ValueError, match="initialization|paired|seed"):
        paired_policy(config)


def test_legacy_identity_untouched_and_paired_identity_normalized():
    config = config_for("hist14")
    policy = config.pop("initialization")
    legacy = checkpoint_identity(config)
    assert "initialization" not in legacy["training"]
    config["initialization"] = policy
    paired = checkpoint_identity(config)
    assert paired["training"].pop("initialization") == policy
    assert paired == legacy


@pytest.mark.parametrize("name,paired", [("hist22", False), ("hist22", True), ("hist14", True)])
def test_raw_weights_cannot_bypass_new_identity(name, paired):
    config = config_for(name)
    if not paired:
        del config["initialization"]
    model = build_model(config)
    with pytest.raises(ValueError, match="identity|metadata"):
        validate_evaluation_checkpoint(model.state_dict(), config)


def test_hist22_deployment_deferred_and_run_name_explicit():
    from common import generate_run_name
    config = config_for()
    assert "hist22" in generate_run_name(config)
    with pytest.raises(ValueError, match="[Hh]ist22|hist22"):
        validate_deployment_config(config, "ONNX")


def parts(config):
    model, provenance = build_initialized_model(config)
    config["initialization_provenance"] = provenance
    criterion = train.build_training_criterion(config, torch.device("cpu"))
    optimizer = train.build_optimizer(model, criterion, config)
    scheduler = train.build_scheduler(optimizer, config, 4)
    return (model, criterion, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
            torch.Generator().manual_seed(42))


def save(config, p):
    return copy.deepcopy(train.checkpoint_payload(*p[:5], 1, {"loss": 2.}, 2., config,
        loader_generator=p[5]))


def restore(saved, config, p):
    return train.restore_checkpoint(saved, config, *p[:5], loader_generator=p[5])


@pytest.mark.parametrize("tamper", ["missing", "digest", "config_digest", "policy", "raw", "live_digest"])
def test_provenance_rejection_is_atomic(tamper):
    config = config_for()
    source = parts(config)
    saved = save(config, source)
    if tamper == "missing":
        saved.pop("initialization_provenance")
    elif tamper == "digest":
        saved["initialization_provenance"]["source_state_sha256"] = "0" * 64
    elif tamper == "config_digest":
        saved["config"]["initialization_provenance"]["source_state_sha256"] = "0" * 64
    elif tamper == "policy":
        saved["config"]["initialization"]["seed"] = 43
    elif tamper == "raw":
        saved = saved["model_state_dict"]
    destination = parts(config)
    if tamper == "live_digest":
        destination[0]._paired_initialization["source_state_sha256"] = "0" * 64
    before = {k: v.clone() for k, v in destination[0].state_dict().items()}
    criterion_before = state_dict_digest(destination[1].state_dict())
    scheduler_before = copy.deepcopy(destination[3].state_dict())
    scaler_before = copy.deepcopy(destination[4].state_dict())
    loader_before = destination[5].get_state().clone()
    train.seed_everything(99)
    expected = draws()
    train.seed_everything(99)
    with pytest.raises(ValueError, match="initialization|provenance|identity|seed"):
        restore(saved, config, destination)
    assert_draws_equal(draws(), expected)
    for key, value in destination[0].state_dict().items():
        assert torch.equal(value, before[key])
    assert state_dict_digest(destination[1].state_dict()) == criterion_before
    assert not destination[2].state
    assert destination[3].state_dict() == scheduler_before
    assert destination[4].state_dict() == scaler_before
    assert torch.equal(destination[5].get_state(), loader_before)


@pytest.mark.parametrize("change", ["source_digest", "missing_provenance", "schema", "policy"])
def test_evaluation_rejects_invalid_pair_metadata(change):
    config = config_for()
    saved = save(config, parts(config))
    if change == "source_digest":
        saved["initialization_provenance"]["source_state_sha256"] = "0" * 64
    elif change == "missing_provenance":
        del saved["initialization_provenance"]
    elif change == "schema":
        config["data"]["bev_encoding"] = {"name": "hist14"}
    else:
        del config["initialization"]
    with pytest.raises(ValueError, match="provenance|identity"):
        validate_evaluation_checkpoint(saved, config)


def test_unpaired_hist22_raw_resume_requires_metadata_but_legacy_hist14_remains_supported():
    for name in ("hist14", "hist22"):
        config = config_for(name)
        del config["initialization"]
        model = build_model(config)
        criterion = train.build_training_criterion(config, torch.device("cpu"))
        optimizer = train.build_optimizer(model, criterion, config)
        scheduler = train.build_scheduler(optimizer, config, 4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        args = (model.state_dict(), config, model, criterion, optimizer, scheduler, scaler)
        if name == "hist22":
            with pytest.raises(ValueError, match="identity metadata"):
                train.restore_checkpoint(*args)
        else:
            validate_evaluation_checkpoint(model.state_dict(), config)
            assert train.restore_checkpoint(*args)["mode"] == "legacy_weights"


def test_checkpoint_serialized_roundtrip_with_valid_provenance(tmp_path):
    config = config_for()
    source = parts(config)
    saved = save(config, source)
    path = tmp_path / "checkpoint.pt"
    torch.save(saved, path)
    saved = torch.load(path, weights_only=True)
    destination = parts(config)
    assert restore(saved, config, destination)["mode"] == "resume"
    validate_evaluation_checkpoint(saved, config)
    assert state_dict_digest(source[0].state_dict()) == state_dict_digest(destination[0].state_dict())


@pytest.mark.parametrize("grouped,box_mode", [(False, "bev"), (True, "bev"), (True, "3d")])
def test_initial_encoded_output_parity_and_actual_criterion_step_updates_all_new_slices(tmp_path, grouped, box_mode):
    from core.bev_encoding import resolve_bev_encoding
    from core.datasets.dataset import Dataset
    from utils_1.hist22_backend import encode_hist22_chw
    config = config_for(grouped=grouped)
    config["data"]["box_mode"] = box_mode
    if box_mode == "3d":
        config["loss"]["vertical_loss_weight"] = 1.
    candidate, provenance = build_initialized_model(config)
    baseline_config = copy.deepcopy(config)
    baseline_config["data"]["bev_encoding"] = {"name": "hist14"}
    baseline, _ = build_initialized_model(baseline_config)
    # Dense, anisotropic volume yields finite nonzero features in all 8 slices.
    rng = np.random.default_rng(19)
    points = rng.uniform([.05, -1.55, -2.4, 0.], [4.75, 1.55, .9, 1.], (2000, 4)).astype(np.float32)
    voxel = torch.from_numpy(encode_hist22_chw(points,
        resolve_bev_encoding(config["data"]["bev_encoding"], config["data"]["kitti"]["geometry"]))).unsqueeze(0)
    assert (voxel[:, 14:].flatten(2).abs().sum(2) > 0).all()
    candidate.eval()
    baseline.eval()
    with torch.no_grad():
        actual, expected = candidate(voxel), baseline(voxel[:, :14].contiguous())
    def compare(left, right):
        for key in left:
            if isinstance(left[key], dict):
                compare(left[key], right[key])
            else:
                torch.testing.assert_close(left[key], right[key], atol=2e-6, rtol=1e-5)
    compare(actual, expected)
    # Build actual target maps at the matching 0.1m schema/grid dimensions.
    target_builder = Dataset.__new__(Dataset)
    target_builder.config = config["data"]
    target_builder.output_shape = [12, 8]
    target_builder.out_size_factor = 4
    target_builder.num_classes = 3
    target_builder.cls_encoding = "gaussian"
    target_builder.target_backend = "python"
    from common import detection_spec
    target_builder.task_groups = detection_spec(config).groups if grouped else ()
    boxes = np.array([[0, 1.5, .8, 1.2, 2., 0., -1.7, .2]], np.float32)
    target = target_builder.get_label(torch.from_numpy(boxes), config["data"]["kitti"]["geometry"])
    target = train.move_tensor_batch(torch.utils.data.default_collate([target]), torch.device("cpu"))
    target["voxel"] = voxel
    criterion = train.build_training_criterion(config, torch.device("cpu"))
    optimizer = train.build_optimizer(candidate, criterion, config)
    candidate.train()
    train.set_loss_epoch(criterion, 0)
    result = criterion(candidate(voxel), target)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    stem = dict(candidate.named_parameters())[provenance["input_weight_key"]]
    gradient = stem.grad[:, 14:]
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().sum(dim=(0, 2, 3)) > 0).all()
    optimizer.step()
    assert torch.isfinite(stem[:, 14:]).all()
    assert (stem[:, 14:].abs().sum(dim=(0, 2, 3)) > 0).all()


def test_pair_data_rng_sampler_order_and_validation_iteration_are_independent():
    from torch.utils.data import DataLoader, TensorDataset
    results = []
    for name in ("hist14", "hist22"):
        config = config_for(name)
        train.seed_everything(42)
        build_initialized_model(config)
        generator = torch.Generator().manual_seed(42)
        dataset = TensorDataset(torch.arange(12))
        sampler = DataLoader(dataset, batch_size=3, shuffle=True, generator=generator)
        validation = DataLoader(dataset, batch_size=4, generator=torch.Generator().manual_seed(42))
        before = torch.get_rng_state().clone()
        list(validation)
        assert torch.equal(before, torch.get_rng_state())
        results.append((draws(), torch.cat([batch[0] for batch in sampler])))
    assert_draws_equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])


@pytest.mark.parametrize("argument", [["--warm-start", "trained.pt"], ["--seed", "43"]])
def test_cli_rejects_pair_conflicts_before_dataset_or_checkpoint_access(tmp_path, argument):
    import json
    config = config_for()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="paired|seed"):
        train.main(["--config", str(path), "--detector-root", str(ROOT / "detector"),
                    "--output-root", str(tmp_path / "runs"), "--device", "cpu", *argument])


def test_package_metadata_remains_pure_in_a_fresh_process():
    import subprocess
    script = '''
import builtins, json
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy'}:
        raise AssertionError('metadata imported ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools.kitti_training_pipeline.common import checkpoint_identity
config = json.load(open('configs/config.json'))
legacy = checkpoint_identity(config)
config['seed'] = 42
config['initialization'] = {'mode':'hist14_hist22_paired_random','version':1,'seed':42}
assert checkpoint_identity(config)['training']['initialization']['seed'] == 42
assert 'initialization' not in legacy['training']
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_package_only_model_helper_import_in_a_fresh_process():
    import subprocess
    script = '''
import json
from tools.kitti_training_pipeline.common import configure_detector_imports
from tools.kitti_training_pipeline.paired_initialization import build_initialized_model
configure_detector_imports('detector')
config = json.load(open('configs/config.json'))
config['seed'] = 42
config['data']['bev_encoding'] = {'name': 'hist22'}
config['initialization'] = {'mode':'hist14_hist22_paired_random','version':1,'seed':42}
model, provenance = build_initialized_model(config)
assert provenance['source_channels'] == 14
assert model.backbone.stem[0].in_channels == 22
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mutation", ["key", "dtype", "shape"])
def test_tensor_preflight_rejects_before_copying_any_shared_values(mutation):
    from paired_initialization import _copy_pair
    baseline, provenance = build_initialized_model(config_for("hist14"))
    target, _ = build_initialized_model(config_for())
    # Distinguish every live value from the source to detect partial copying.
    with torch.no_grad():
        for value in target.state_dict().values():
            value.add_(1)
    if mutation == "key":
        target.register_buffer("extra", torch.zeros(1))
    elif mutation == "dtype":
        target.header = target.header.double()
    else:
        target.header.register_buffer("bad_shape", torch.zeros(2))
        baseline.header.register_buffer("bad_shape", torch.zeros(3))
    before = state_dict_digest(target.state_dict())
    with pytest.raises(ValueError, match="keys|dtype|shape"):
        _copy_pair(target, baseline, provenance)
    assert state_dict_digest(target.state_dict()) == before


def cli_config(tmp_path, name="hist22"):
    from test_grouped_targets import make_dataset
    dataset = make_dataset(tmp_path, grouped=False)
    config = config_for(name)
    geometry = config["data"]["kitti"]["geometry"]
    config["data"] = dataset.config
    config["data"]["bev_encoding"] = {"name": name}
    config["data"]["kitti"]["geometry"] = geometry
    root = Path(config["data"]["kitti"]["location"])
    for folder, suffix in (("pointcloud", "bin"), ("label", "txt")):
        (root / folder / f"000001.{suffix}").write_bytes((root / folder / f"000000.{suffix}").read_bytes())
    val = tmp_path / "validation.txt"
    val.write_text("000001;kitti\n")
    config["augmentation"] = {"mode": "openpcdet", "AUG_CONFIG_LIST": []}
    config["train"].update(data=str(dataset.data_file), epochs=2, physical_batch_size=1,
                           accumulation_steps=1, save_every=1, num_workers=0,
                           target_backend="python", compile_model=True, precision="fp32")
    config["val"] = {"data": str(val), "physical_batch_size": 1}
    return config


def test_actual_cli_pair_fresh_and_resume_restore_rng_after_compile_warmup(tmp_path, monkeypatch):
    import json
    observations, validation_generators, seed_calls = [], [], []
    original_loader, original_seed = train.DataLoader, train.seed_everything
    class ObservedLoader(original_loader):
        def __iter__(self):
            if self.dataset.task == "train":
                observations.append((draws(), self.generator.get_state().clone()))
            else:
                validation_generators.append(self.generator)
                before = torch.get_rng_state().clone()
            iterator = super().__iter__()
            if self.dataset.task != "train":
                assert torch.equal(before, torch.get_rng_state())
            return iterator
    def noisy_warmup(model, voxel, *args):
        # Deliberately different setup consumption between encoder widths.
        for _ in range(voxel.shape[1]):
            draws()
    def seed_spy(seed):
        seed_calls.append(seed)
        return original_seed(seed)
    monkeypatch.setattr(train, "DataLoader", ObservedLoader)
    monkeypatch.setattr(train, "warmup_model", noisy_warmup)
    monkeypatch.setattr(torch, "compile", lambda model: model)
    monkeypatch.setattr(train, "seed_everything", seed_spy)
    config = cli_config(tmp_path)
    path = tmp_path / "config.json"
    def run(name, checkpoint=None):
        config["data"]["bev_encoding"] = {"name": name}
        path.write_text(json.dumps(config))
        run_name = name + ("_resume" if checkpoint else "")
        argv = ["--config", str(path), "--detector-root", str(ROOT / "detector"),
                "--output-root", str(tmp_path / "runs"), "--run-name", run_name, "--device", "cpu"]
        if checkpoint:
            argv += ["--resume", str(checkpoint)]
        train.main(argv)
        return tmp_path / "runs" / run_name
    a = run("hist14")
    b = run("hist22")
    assert len(observations) == 4
    assert_draws_equal(observations[0][0], observations[2][0])
    assert torch.equal(observations[0][1], observations[2][1])
    original_seed(42)
    assert_draws_equal(observations[0][0], draws())
    assert all(generator is not None for generator in validation_generators)
    saved_a = torch.load(a / "checkpoints/1epoch.pt", weights_only=True)
    saved_b = torch.load(b / "checkpoints/1epoch.pt", weights_only=True)
    assert saved_a["initialization_provenance"] == saved_b["initialization_provenance"]
    seed_calls.clear()
    resumed = run("hist22", b / "checkpoints/1epoch.pt")
    # Initial CLI seed is allowed; the post-warmup boundary must restore, never reseed.
    assert seed_calls == [42]
    assert_draws_equal(observations[3][0], observations[4][0])
    assert torch.equal(observations[3][1], observations[4][1])
    uninterrupted = torch.load(b / "checkpoints/2epoch.pt", weights_only=True)
    restored = torch.load(resumed / "checkpoints/2epoch.pt", weights_only=True)
    assert state_dict_digest(uninterrupted["model_state_dict"]) == state_dict_digest(restored["model_state_dict"])
    assert state_dict_digest(uninterrupted["criterion_state_dict"]) == state_dict_digest(restored["criterion_state_dict"])


def test_legacy_cli_keeps_original_model_and_validation_generator_path(tmp_path, monkeypatch):
    import json
    config = cli_config(tmp_path, "hist14")
    del config["initialization"]
    config["train"].update(epochs=1, compile_model=False)
    observed = []
    original_loader = train.DataLoader
    def loader(*args, **kwargs):
        observed.append(kwargs.get("generator"))
        return original_loader(*args, **kwargs)
    def unexpected(*args):
        raise AssertionError("legacy model construction entered paired helper")
    monkeypatch.setattr(train, "DataLoader", loader)
    monkeypatch.setattr(train, "build_initialized_model", unexpected)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    train.main(["--config", str(path), "--detector-root", str(ROOT / "detector"),
                "--output-root", str(tmp_path / "runs"), "--device", "cpu"])
    assert observed[0] is not None and observed[1] is None
