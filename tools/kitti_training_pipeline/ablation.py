"""Pure config helpers shared by Colab and ablation tests."""

from copy import deepcopy
import hashlib
import json


def resolve_ablation_config(
    config, *, bev_encoding=None, scale_gated_fpn=None, c5_attention=None,
):
    """Apply explicit overrides to a copy; None preserves the source config."""
    result = deepcopy(config)
    if result["model"]["backbone"] != "mobilepixor":
        raise ValueError("Controlled backbone ablations require backbone='mobilepixor'")
    if bev_encoding is not None:
        result["data"].setdefault("bev_encoding", {})["name"] = bev_encoding
    if scale_gated_fpn is not None:
        result["model"]["scale_gated_fpn"] = scale_gated_fpn
    if c5_attention is not None:
        result["model"]["c5_attention"] = c5_attention
    if result["data"].get("bev_encoding", {}).get("name", "binary_slices") not in (
        "binary_slices", "rich8",
    ):
        raise ValueError("BEV encoding must be 'binary_slices' or 'rich8'")
    if type(result["model"].get("scale_gated_fpn", False)) is not bool:
        raise ValueError("scale_gated_fpn must be a JSON boolean")
    if result["model"].get("c4_attention", "none") not in ("none", "lsk", "litemla"):
        raise ValueError("c4_attention must be 'none', 'lsk', or 'litemla'")
    if result["model"].get("c5_attention", "none") not in ("none", "c2psa"):
        raise ValueError("c5_attention must be 'none' or 'c2psa'")
    return result


def ablation_label(config):
    """Describe the actual switches, independent of any user-supplied label."""
    model = config["model"]
    encoding = config["data"].get("bev_encoding", {}).get("name", "binary_slices")
    encoding = "legacy35" if encoding == "binary_slices" else encoding
    fpn = "sgfpn" if model.get("scale_gated_fpn", False) else "sumfpn"
    return (
        f"{encoding}_c4-{model.get('c4_attention', 'none')}_"
        f"c5-{model.get('c5_attention', 'none')}_{fpn}"
    )


def config_digest(config):
    content = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
