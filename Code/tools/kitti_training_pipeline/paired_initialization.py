"""Reproducible untrained Hist14/Hist22 pairs; never a checkpoint adapter."""

from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import random

MODE = "hist14_hist22_paired_random"


def _common():
    import importlib
    return importlib.import_module((__package__ + "." if __package__ else "") + "common")


def paired_policy(config):
    """Return the strict opt-in policy, preserving historical init modes."""
    value = config.get("initialization")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("initialization must be an object")
    if value.get("mode") != MODE:
        return None
    if set(value) != {"mode", "version", "seed"}:
        raise ValueError("paired initialization requires exactly mode, version and seed")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("paired initialization version must be integer 1")
    seed = value["seed"]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("paired initialization seed must be an integer in [0, 2**32)")
    if type(config.get("seed", 42)) is not int or seed != config.get("seed", 42):
        raise ValueError("paired initialization seed must match config seed")
    return {"mode": MODE, "version": 1, "seed": seed}


@contextmanager
def _isolated_rng(seed):
    import numpy as np
    import torch
    python_state, numpy_state = random.getstate(), np.random.get_state()
    # fork_rng restores CPU and every initialized CUDA generator on exceptions.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def state_dict_digest(state):
    """Hash sorted names, dtype, shape and exact tensor bytes with framing."""
    import torch
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not isinstance(key, str) or not torch.is_tensor(value) or value.layout != torch.strided:
            raise ValueError("paired initialization state must contain named dense tensors")
        header = json.dumps([key, str(value.dtype), list(value.shape)],
                            separators=(",", ":")).encode()
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        for part in (header, raw):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _source_config(config):
    schema = _common().bev_encoding_spec(config)
    if schema.name not in {"hist14", "hist22"}:
        raise ValueError("paired initialization supports only Hist14 and Hist22")
    if str(config["model"].get("backbone", "")).lower() != "mobilepixornext":
        raise ValueError("paired initialization requires MobilePixorNeXt")
    source = copy.deepcopy(config)
    source["data"]["bev_encoding"] = {"name": "hist14", "backend": "numpy",
        "density_norm": schema.density_norm, "intensity_scale": schema.intensity_scale}
    return source


def _source_and_provenance(config, policy):
    common = _common()
    source_config = _source_config(config)
    with _isolated_rng(policy["seed"]):
        source = common.build_model(source_config)
    provenance = {**policy, "source_state_sha256": state_dict_digest(source.state_dict()),
                  "source_encoding_sha256": common.bev_encoding_spec(source_config).semantic_hash,
                  "input_weight_key": "backbone.stem.0.weight", "source_channels": 14}
    return source, provenance


def _copy_pair(model, source, provenance):
    """Preflight every tensor before the single strict load changes the target."""
    import torch
    stem = provenance["input_weight_key"]
    actual_conv = model.backbone.stem[0]
    if (type(actual_conv) is not torch.nn.Conv2d or
            dict(model.named_parameters()).get(stem) is not actual_conv.weight):
        raise ValueError("paired initialization input convolution topology differs")
    expected, actual = source.state_dict(), model.state_dict()
    if expected.keys() != actual.keys():
        raise ValueError("paired initialization tensor keys differ")
    updated = {}
    for key, src in expected.items():
        dst = actual[key]
        if dst.dtype != src.dtype:
            raise ValueError(f"paired initialization dtype differs: {key}")
        if dst.shape == src.shape:
            updated[key] = src
        elif (key == stem and src.shape == (32, 14, 3, 3) and
              dst.shape == (32, 22, 3, 3)):
            updated[key] = torch.zeros_like(dst)
            updated[key][:, :14].copy_(src)
        else:
            raise ValueError(f"paired initialization tensor shape differs: {key}")
    model.load_state_dict(updated, strict=True)


def build_initialized_model(config):
    """Build a fresh model and return its policy provenance (None for legacy)."""
    build_model = _common().build_model
    policy = paired_policy(config)
    if policy is None:
        return build_model(config), None
    source, provenance = _source_and_provenance(config, policy)
    with _isolated_rng(policy["seed"]):
        model = build_model(config)
    _copy_pair(model, source, provenance)
    model._paired_initialization = copy.deepcopy(provenance)
    return model, provenance


def validate_paired_provenance(checkpoint, config, model=None):
    """Recompute the untrained source before touching any live training state."""
    policy = paired_policy(config)
    if policy is None:
        return
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict) or paired_policy(saved_config) != policy:
        raise ValueError("paired initialization checkpoint policy is missing or incompatible")
    _, expected = _source_and_provenance(config, policy)
    if (checkpoint.get("initialization_provenance") != expected or
            saved_config.get("initialization_provenance") != expected):
        raise ValueError("paired initialization source provenance does not match config")
    if config.get("initialization_provenance", expected) != expected:
        raise ValueError("paired initialization current config provenance does not match source")
    if model is not None and getattr(model, "_paired_initialization", None) != expected:
        raise ValueError("paired initialization live model provenance does not match source")
