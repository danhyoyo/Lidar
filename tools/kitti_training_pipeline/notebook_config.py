"""Fresh, reproducible configuration resolution for the Colab notebook.

Precedence: base -> model selection -> augmentation recipe -> file override
-> notebook runtime. These helpers do not load datasets or GT databases.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

try:
    from .common import create_experiment_config, input_shape, read_json, write_json
except ImportError:
    from common import create_experiment_config, input_shape, read_json, write_json


BEV_CHANNELS = {"rich8": 8, "rich10": 10, "rich11": 11, "rich12": 12}
MODEL_DEFAULTS = {
    "backbone": "mobilepixornext", "backbone_out_dim": 16,
    "neck_type": None, "scale_gated_fpn": True,
    "use_reparam": False, "header_use_iou": False,
    "header_use_bn": True, "header_act": "silu",
    "c4_attention": "litemla", "c4_attention_scales": [5],
    "c4_attention_qk_norm": "none", "deploy": False,
}


def resolve_notebook_config(
    repo_dir, *, base_path="configs/config.json", preset="custom",
    custom_overrides=None, augmentation="standard", file_override=None,
    runtime_overrides=None,
):
    """Resolve a fresh configuration; never mutate defaults or cached selections."""
    root = Path(repo_dir)
    config = read_json(root / base_path)
    if preset == "custom":
        config = create_experiment_config(config, custom_overrides)
    elif preset != "config":
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"Unknown PRESET: {preset!r}; choose custom, config or {list(PRESET_CONFIGS)}")
        defaults = {"model": MODEL_DEFAULTS, "loss": {"name": "baseline", "use_iou": False}}
        recipe = create_experiment_config(defaults, PRESET_CONFIGS[preset])
        config = create_experiment_config(config, recipe)
        encoding = config["data"]["bev_encoding"]
        encoding.pop("out_channels", None)
        if encoding["name"] in BEV_CHANNELS:
            encoding["out_channels"] = BEV_CHANNELS[encoding["name"]]

    if augmentation in ("standard", "compose", "none"):
        # Replace the whole recipe so an OpenPCDet queue cannot survive a mode switch.
        config["augmentation"] = {
            "mode": "compose" if augmentation == "compose" else "one_of",
            "p": 0.0 if augmentation == "none" else 0.5,
            "rotation": {"use": True, "limit_angle": 20, "p": 1},
            "scaling": {"use": True, "range": [0.95, 1.05], "p": 1},
            "translation": {"use": True, "scale": 0.4, "scale_z": 0.4, "p": 1},
        }
    elif augmentation in ("openpcdet_global", "openpcdet_gt"):
        profile = read_json(root / "configs/augmentation" / f"{augmentation}.json")
        config["augmentation"] = copy.deepcopy(profile["augmentation"])
    elif augmentation != "config":
        raise ValueError(f"Unknown AUGMENTATION: {augmentation!r}")

    if file_override:
        overrides = read_json(root / file_override)
        # An override changing the mode starts a new recipe, rather than merging stale keys.
        aug = overrides.get("augmentation")
        if aug is not None and aug.get("mode", config["augmentation"].get("mode")) != config["augmentation"].get("mode"):
            config["augmentation"] = {}
        bev = overrides.get("data", {}).get("bev_encoding", {})
        if "name" in bev and bev["name"] != config["data"]["bev_encoding"].get("name"):
            config["data"]["bev_encoding"].pop("out_channels", None)
        config = create_experiment_config(config, overrides)
        # Changing only the IQA flag also changes its loss supervision.
        if "header_use_iou" in overrides.get("model", {}) and "use_iou" not in overrides.get("loss", {}):
            config["loss"]["use_iou"] = config["model"]["header_use_iou"]
    config = create_experiment_config(config, runtime_overrides)
    validate_notebook_config(config)
    return config


def validate_notebook_config(config):
    """Reject options the advertised notebook backbones cannot actually honor."""
    model, loss, train = config["model"], config["loss"], config["train"]
    backbone = model.get("backbone")
    if backbone not in ("mobilepixornext", "mobilepixor", "mobilepixor_coordatt"):
        raise ValueError(f"Unsupported notebook BACKBONE: {backbone!r}")
    if model.get("neck_type") not in (None, "scale_gated_fpn", "sgfpn", "rc_sgfpn", "rc_bisgfpn"):
        raise ValueError("Unsupported NECK_TYPE")
    channels = model.get("backbone_out_dim", 16)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("BACKBONE_OUT_DIM must be a positive integer")
    if backbone != "mobilepixornext":
        if channels != 16:
            raise ValueError("BACKBONE_OUT_DIM must be 16 for MobilePIXOR backbones")
        if model.get("neck_type") in ("rc_sgfpn", "rc_bisgfpn") or model.get("use_reparam", False):
            raise ValueError("Range-conditioned necks and USE_REPARAM require mobilepixornext")
        model["c4_attention"] = "none"
    if model.get("c4_attention", "none") not in ("none", "litemla"):
        raise ValueError("C4_ATTENTION must be none or litemla")
    if model.get("c4_attention", "none") == "none":
        model["c4_attention_scales"] = []
        model["c4_attention_qk_norm"] = "none"
    else:
        scales = model.get("c4_attention_scales", [5])
        if not scales or any(isinstance(k, bool) or not isinstance(k, int) or k <= 0 or k % 2 == 0 for k in scales):
            raise ValueError("C4_ATTENTION_SCALES must contain positive odd integers")
        if model.get("c4_attention_qk_norm", "none") not in ("none", "rmsnorm", "layernorm"):
            raise ValueError("Unsupported C4_ATTENTION_QK_NORM")
    if loss.get("name") not in ("baseline", "oga", "q_oga", "gw_qal", "uwag"):
        raise ValueError("Unsupported LOSS_NAME")
    use_iou = model.get("header_use_iou", False)
    if bool(loss.get("use_iou", False)) != bool(use_iou):
        raise ValueError("header_use_iou and loss.use_iou must agree")
    if use_iou and loss["name"] not in ("baseline", "oga"):
        raise ValueError("IQA supervision requires baseline or oga loss")
    if config["data"].get("out_size_factor", 4) != 4:
        raise ValueError("These backbones output stride 4; out_size_factor must be 4")
    input_shape(config)
    epochs, warmup = train["epochs"], train.get("warmup_epochs", 0)
    if not isinstance(epochs, int) or not isinstance(warmup, int) or epochs <= 0 or not 0 <= warmup < epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
    for section, name in ((train, "physical_batch_size"), (train, "accumulation_steps"), (config["val"], "physical_batch_size")):
        if not isinstance(section[name], int) or isinstance(section[name], bool) or section[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(train["learning_rate"]) or train["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if train["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("Unsupported PRECISION")
    if train["target_backend"] not in ("python", "numba"):
        raise ValueError("Unsupported TARGET_BACKEND")
    if not isinstance(train["num_workers"], int) or train["num_workers"] < 0:
        raise ValueError("NUM_WORKERS must be a nonnegative integer")
    aug = config["augmentation"]
    if aug.get("mode", "one_of") not in ("one_of", "compose", "openpcdet"):
        raise ValueError("Unsupported augmentation.mode")
    if aug.get("mode") == "openpcdet":
        if not isinstance(aug.get("AUG_CONFIG_LIST"), list):
            raise ValueError("OpenPCDet augmentation requires AUG_CONFIG_LIST")
    elif not 0 <= aug.get("p", 0) <= 1:
        raise ValueError("augmentation.p must be between 0 and 1")


def save_notebook_config(run_dir, config):
    """Keep the original config intact if a saved training run would be incompatible."""
    run_dir = Path(run_dir)
    if run_dir.name in (".", ".."):
        raise ValueError("Invalid run directory")
    path = run_dir / "config.json"
    trained = (run_dir / "run.json").is_file() or any((run_dir / "checkpoints").glob("*.pt"))
    if path.is_file() and trained and read_json(path) != config:
        raise ValueError(
            "Run đã có metadata/checkpoint với config khác. Đổi EXPERIMENT_TAG hoặc "
            "CUSTOM_RUN_NAME để tạo run mới; config cũ được giữ nguyên."
        )
    write_json(path, config)
    return path


PRESET_CONFIGS = {
    "TRUC_A_SOTA": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "c4_attention": "litemla",
            "c4_attention_scales": [
                3,
                5
            ],
            "c4_attention_qk_norm": "rmsnorm",
            "use_reparam": True,
            "header_use_iou": True
        },
        "loss": {
            "name": "oga",
            "use_iou": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "TRUC_B_SOTA": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "c4_attention": "litemla",
            "c4_attention_scales": [
                3,
                5
            ],
            "c4_attention_qk_norm": "rmsnorm",
            "use_reparam": True,
            "header_use_iou": False
        },
        "loss": {
            "name": "q_oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "TRUC_A_M1": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "use_reparam": True,
            "header_use_iou": False,
            "c4_attention": "none"
        },
        "loss": {
            "name": "oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "TRUC_A_M2": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "c4_attention": "litemla",
            "c4_attention_scales": [
                3,
                5
            ],
            "c4_attention_qk_norm": "rmsnorm",
            "use_reparam": True,
            "header_use_iou": False
        },
        "loss": {
            "name": "oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "TRUC_B_M1": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "use_reparam": True,
            "header_use_iou": False,
            "c4_attention": "none"
        },
        "loss": {
            "name": "q_oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RICH8_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RICH10_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich10"
            }
        }
    },
    "RICH11_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich11"
            }
        }
    },
    "RICH12_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich12"
            }
        }
    },
    "RC_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "neck_type": "rc_sgfpn",
            "scale_gated_fpn": True
        },
        "loss": {
            "name": "q_oga"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RC_BISGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "neck_type": "rc_bisgfpn",
            "scale_gated_fpn": True
        },
        "loss": {
            "name": "q_oga"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "MOBILEPIXOR_BASELINE": {
        "model": {
            "backbone": "mobilepixor",
            "scale_gated_fpn": False,
            "c4_attention": "none",
            "header_use_bn": False,
            "header_act": "relu"
        },
        "loss": {
            "name": "baseline"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "LEGACY35_BASELINE": {
        "model": {
            "backbone": "mobilepixor",
            "scale_gated_fpn": False,
            "c4_attention": "none",
            "header_use_bn": False,
            "header_act": "relu"
        },
        "loss": {
            "name": "baseline"
        },
        "data": {
            "bev_encoding": {
                "name": "binary_slices"
            }
        }
    }
}
