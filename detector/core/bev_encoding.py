"""Pure BEV encoding contracts shared by processing, models and tooling.

This module resolves semantics only. Requesting a backend does not import it or
imply that its rasterizer is available. Rasterization lives in dataset utilities.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

_RICH_MINIMUM = {"rich8": 8, "rich10": 10, "rich11": 11, "rich12": 12}
_HIST14_CHANNELS = (
    "height_count_b0", "height_count_b1", "height_count_b2", "height_count_b3",
    "z_max", "z_mean", "z_span", "z_std", "intensity_max", "intensity_mean",
    "log_density", "occupancy", "point_offset_x", "point_offset_y",
)
_RICH_CHANNELS = (
    "height_occupancy_b0", "height_occupancy_b1", "height_occupancy_b2",
    "z_max", "z_mean", "intensity_max", "intensity_mean", "log_density",
    "z_span", "z_std", "intensity_contrast", "range_compensated_density",
)
_HIST14_OPTIONS = frozenset({
    "name", "version", "out_channels", "density_norm", "intensity_scale", "backend",
})


def minimum_channels(name: str) -> int | None:
    """Return fixed/minimum width, or None for geometry-dependent binary slices."""
    if name == "binary_slices":
        return None
    if name == "hist14":
        return 14
    if name in {"pillar32", "pillar_rich", "pillar_rich_gate"}:
        return 32
    if name in _RICH_MINIMUM:
        return _RICH_MINIMUM[name]
    raise ValueError(f"unsupported BEV encoding: {name!r}")


def _positive_float(value, name, lower_bound):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and greater than {lower_bound}") from exc
    if not math.isfinite(result) or result <= lower_bound:
        raise ValueError(f"{name} must be finite and greater than {lower_bound}")
    return result


def _pillar_pooling_options(encoding):
    name = encoding.get("name")
    pooling = encoding.get("pooling", "max")
    if not isinstance(pooling, str) or pooling != "max":
        raise ValueError(f"{name} pooling must be max")
    return pooling


def _options(encoding):
    if encoding is None:
        encoding = {}
    if not isinstance(encoding, Mapping):
        raise ValueError("bev_encoding must be a mapping")
    name = encoding.get("name", "binary_slices")
    if not isinstance(name, str):
        raise ValueError("BEV encoding name must be a string")
    minimum = minimum_channels(name)
    version = encoding.get("version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"unsupported {name} version: {version!r}; expected integer 1")
    configured = encoding.get("out_channels")
    if name in {"pillar32", "pillar_rich", "pillar_rich_gate"}:
        supported = {"name", "version", "out_channels", "backend", "intensity_scale", "density_norm"}
        if name in {"pillar_rich", "pillar_rich_gate"}:
            supported.add("pooling")
            _pillar_pooling_options(encoding)
        unsupported = set(encoding) - supported
        if unsupported:
            raise ValueError(f"unsupported {name} options: {sorted(unsupported, key=str)}")
        if "out_channels" in encoding and (type(configured) is not int or configured != 32):
            raise ValueError(f"{name} out_channels must be exactly integer 32")
        if encoding.get("backend", "torch") != "torch":
            raise ValueError(f"{name} backend must be torch")
        # Legacy rich recipes may retain density_norm when deep-merged.
        # pillar32 ignores it; rich pillar variants use it for rich8 statistics.
        if "density_norm" in encoding:
            _positive_float(encoding["density_norm"], "density_norm", 1)
        intensity = _positive_float(encoding.get("intensity_scale", 1), "intensity_scale", 0)
        density = (_positive_float(encoding.get("density_norm", 32), "density_norm", 1)
                   if name in {"pillar_rich", "pillar_rich_gate"} else None)
        return name, version, 32, density, intensity, "torch"
    elif name == "hist14":
        unsupported = set(encoding) - _HIST14_OPTIONS
        if unsupported:
            raise ValueError(f"unsupported hist14 options: {sorted(unsupported, key=str)}")
        if "out_channels" in encoding and (type(configured) is not int or configured != 14):
            raise ValueError("hist14 v1 out_channels must be exactly integer 14")
        backend = encoding.get("backend", "numpy")
        if backend not in ("numpy", "numba"):
            raise ValueError("hist14 backend must be numpy or numba")
        channels = 14
    elif minimum is not None:
        # Preserve int coercion, minimum clamping and zero padding of rich paths.
        try:
            channels = minimum if configured is None else max(int(configured), minimum)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"invalid {name} out_channels: {configured!r}") from exc
        backend = "numpy"
    else:
        # Legacy voxelize ignores out_channels and normalization options.
        return name, version, None, None, None, "numpy"
    density = _positive_float(encoding.get("density_norm", 32.0), "density_norm", 1)
    intensity = _positive_float(encoding.get("intensity_scale", 1.0), "intensity_scale", 0)
    return name, version, channels, density, intensity, backend


def _geometry(geometry):
    if not isinstance(geometry, Mapping):
        raise ValueError("geometry must be a mapping with xyz bounds and resolutions")
    normalized, shape, cells_by_axis = [], [], []
    for axis in "xyz":
        try:
            lower, upper, resolution = (float(geometry[f"{axis}_{suffix}"])
                                        for suffix in ("min", "max", "res"))
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {axis}-axis geometry") from exc
        if not all(math.isfinite(v) for v in (lower, upper, resolution)) or resolution <= 0 or upper <= lower:
            raise ValueError(f"invalid {axis}-axis geometry")
        cells = (upper - lower) / resolution
        if not math.isfinite(cells) or cells <= 0:
            raise ValueError(f"invalid {axis}-axis geometry")
        rounded = round(cells)
        # Same tolerance as legacy np.isclose(cells, round(cells), atol=1e-6).
        if abs(cells - rounded) > 1e-6 + 1e-5 * abs(rounded) or rounded <= 0:
            raise ValueError(f"{axis}-axis range must be divisible by its resolution")
        normalized.extend((f"{axis}_{suffix}", value) for suffix, value in
                          zip(("min", "max", "res"), (lower, upper, resolution)))
        shape.append(int(rounded))
        cells_by_axis.append(cells)
    return tuple(normalized), tuple(shape), tuple(cells_by_axis)


@dataclass(frozen=True)
class BEVEncodingSpec:
    """Immutable effective schema; backend is separate from semantic identity."""

    name: str
    version: int
    channels: int
    channel_names: tuple[str, ...]
    geometry: tuple[tuple[str, float], ...]
    grid_shape: tuple[int, int, int]  # XYZ
    bin_edges: tuple[float, ...]
    density_norm: float | None
    intensity_scale: float | None
    backend: str
    pooling: str = "max"

    @property
    def is_packed(self) -> bool:
        return self.name in {"pillar32", "pillar_rich", "pillar_rich_gate"}

    @property
    def is_rich_pillar(self) -> bool:
        return self.name in {"pillar_rich", "pillar_rich_gate"}

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return self.grid_shape[1], self.grid_shape[0], self.channels

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return 1, self.channels, self.grid_shape[1], self.grid_shape[0]

    def semantic_metadata(self) -> dict:
        """Return a fresh JSON-serializable identity, never mutable internal state."""
        if self.is_packed:
            result = {
                "name": self.name, "version": self.version, "channels": self.channels,
                "channel_names": list(self.channel_names), "geometry": dict(self.geometry),
                "grid_shape_xyz": list(self.grid_shape), "bin_edges": list(self.bin_edges),
                "density_norm": self.density_norm, "intensity_scale": self.intensity_scale,
                "layout": {"encoding": "packed_points", "model": "packed_pillars", "bev": "BCYX"},
                "boundary": {"interval": "open", "epsilon": .001},
                "finite_filter": "first_four_columns",
                "xy_indexing": "legacy_float32_floor_divide",
                "precision": {"features": "float32", "indices": "int64"},
                "learned_encoder": {
                    "point_features": 10,
                    "feature_names": ["x", "y", "z", "intensity", "cluster_dx", "cluster_dy",
                                      "cluster_dz", "center_dx", "center_dy", "center_dz"],
                    "architecture": "linear_bn_relu", "pooling": "max",
                    "pillar_center_z": "roi_midpoint", "point_limit": None, "pillar_limit": None,
                    "singleton_bn": "running_statistics", "empty_bev": "zero",
                },
            }
            if self.is_rich_pillar:
                result["learned_encoder"]["out_channels"] = 24
                result["handcrafted_encoder"] = resolve_bev_encoding({
                    "name": "rich8", "density_norm": self.density_norm,
                    "intensity_scale": self.intensity_scale,
                }, dict(self.geometry)).semantic_metadata()
                result["fusion"] = {"operation": "concat_before_scatter", "rich8_channels": [0, 8],
                                    "learned_channels": [8, 32], "rich8_cast": "pooled_dtype"}
                if self.name == "pillar_rich_gate":
                    result["learned_encoder"].update({
                        "gate": {"type": "rich_conditioned_channel_gate",
                                 "descriptor": "concat(learned_max24, rich8)",
                                 "architecture": "linear_relu_linear", "input_channels": 32,
                                 "hidden_channels": 4, "output_channels": 24, "bias": True,
                                 "scope": "per_occupied_pillar", "activation": "2 * sigmoid",
                                 "precision": {"mlp": "model_autocast", "gate_and_scaling": "float32"},
                                 "initial_scale": 1, "last_linear_init": "zero",
                                 "fusion": "max * gate", "rich8_passthrough": True},
                    })
            return result
        histogram = self.name == "hist14"
        binary = self.name == "binary_slices"
        result = {
            "name": self.name, "version": self.version, "channels": self.channels,
            "channel_names": list(self.channel_names), "geometry": dict(self.geometry),
            "grid_shape_xyz": list(self.grid_shape), "bin_edges": list(self.bin_edges),
            "density_norm": self.density_norm, "intensity_scale": self.intensity_scale,
            "layout": {"encoding": "YXC", "model": "BCYX"},
            "boundary": {"interval": "open", "epsilon": 0.001},
            "finite_filter": "xyz_roi" if binary else "first_four_columns",
            "height_bin_ties": "upper",
            "precision": {
                "coordinates": "float64" if histogram else "input_dtype",
                "index": "int64" if histogram else "int32",
                "accumulation": "float64" if histogram else ("binary_assignment" if binary else "legacy_mixed"),
                "output": "float32",
            },
        }
        if self.name in _RICH_MINIMUM and self.channels >= 12:
            result["range_density"] = {"range_scale_m": 20.0, "density_multiplier": 16.0}
        return result

    @property
    def semantic_hash(self) -> str:
        payload = json.dumps(self.semantic_metadata(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_bev_encoding(encoding, geometry) -> BEVEncodingSpec:
    """Resolve a complete schema, validating geometry before any allocation."""
    name, version, channels, density, intensity, backend = _options(encoding)
    normalized, grid, cells = _geometry(geometry)
    geom = dict(normalized)
    if name == "binary_slices":
        # Match voxelize's int division, including historical float roundoff.
        grid = tuple(int(value) for value in cells)
        if min(grid) <= 0:
            raise ValueError("binary_slices geometry must produce positive grid sizes")
        channels = grid[2]
        names = tuple(f"height_slice_{index}" for index in range(channels))
        edges = tuple(geom["z_min"] + index * geom["z_res"] for index in range(channels + 1))
    elif name == "hist14":
        names = _HIST14_CHANNELS
        edges = tuple(geom["z_min"] + index * (geom["z_max"] - geom["z_min"]) / 4
                      for index in range(5))
    elif name == "pillar32":
        names = tuple(f"learned_feature_{index}" for index in range(32))
        edges = (geom["z_min"], geom["z_max"])
    elif name in {"pillar_rich", "pillar_rich_gate"}:
        names = _RICH_CHANNELS[:8] + tuple(f"learned_feature_{index}" for index in range(24))
        edges = tuple(geom["z_min"] + index * (geom["z_max"] - geom["z_min"]) / 3
                      for index in range(4))
    else:
        names = tuple(_RICH_CHANNELS[index] if index < min(channels, 12) and
                      (index < 8 or channels >= 10) else f"reserved_zero_{index}"
                      for index in range(channels))
        edges = tuple(geom["z_min"] + index * (geom["z_max"] - geom["z_min"]) / 3
                      for index in range(4))
    pooling = (_pillar_pooling_options(encoding)
               if name in {"pillar_rich", "pillar_rich_gate"} else "max")
    return BEVEncodingSpec(name, version, channels, names, normalized, grid, edges,
                           density, intensity, backend, pooling)


def resolve_input_channels(encoding=None, geometry=None, *, default_channels=35) -> int:
    """Resolve width, retaining geometry-free legacy model construction.

    Binary slices use the explicit constructor width if geometry is absent;
    fixed-width encodings can be resolved without spatial geometry.
    """
    if geometry is not None:
        return resolve_bev_encoding(encoding, geometry).channels
    _, _, channels, _, _, _ = _options(encoding)
    return default_channels if channels is None else channels
