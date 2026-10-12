"""Resolve KITTI left-camera dimensions from per-frame JSON or PNG headers."""
from __future__ import annotations

import json
import struct
from pathlib import Path


def training_root(path):
    root = Path(path).expanduser().resolve()
    return root / 'training' if (root / 'training').is_dir() else root


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate image metadata key: {key}')
        result[key] = value
    return result


def read_image_sizes(path):
    """Validate the versioned training/image_2 metadata and return (width, height)."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding='utf-8') as stream:
        data = json.load(stream, object_pairs_hook=_unique_keys)
    if not isinstance(data, dict) or any(data.get(key) != value for key, value in {
            'format': 'kitti_image_sizes_v1', 'split': 'training', 'camera': 'image_2'}.items()):
        raise ValueError(f'Expected kitti_image_sizes_v1 for training/image_2: {path}')
    frames = data.get('frames')
    if (not isinstance(frames, dict) or not frames or type(data.get('image_count')) is not int
            or data['image_count'] != len(frames)):
        raise ValueError(f'Image metadata count must match nonempty frames: {path}')
    sizes = {}
    for identifier, frame in frames.items():
        if len(identifier) != 6 or not identifier.isascii() or not identifier.isdigit():
            raise ValueError(f'Invalid KITTI frame ID {identifier!r}: {path}')
        if not isinstance(frame, dict) or any(
                type(frame.get(key)) is not int or frame[key] <= 0 for key in ('width', 'height')):
            raise ValueError(f'Invalid width/height for frame {identifier}: {path}')
        sizes[identifier] = (frame['width'], frame['height'])
    return sizes


def png_image_size(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Expected KITTI PNG image with IHDR: {path}')
    dimensions = struct.unpack('>II', header[16:24])
    if min(dimensions) <= 0:
        raise ValueError(f'Invalid evaluation image dimensions: {path}')
    return dimensions


def load_image_sizes(kitti_root, frame_ids, *, metadata_path=None):
    """Return per-frame sizes and the actual input paths to hash for provenance.

    Explicit JSON takes precedence; otherwise look in the raw training directory
    (or its dataset parent), then retain the historical PNG-header workflow.
    A present but incomplete/invalid JSON is an error rather than a fallback.
    """
    root = training_root(kitti_root)
    ids = list(frame_ids)
    if metadata_path is not None:
        path = Path(metadata_path).expanduser().resolve()
    else:
        candidates = [root / 'kitti_image_sizes.json']
        if root.name == 'training':
            candidates.append(root.parent / 'kitti_image_sizes.json')
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is not None:
        sizes = read_image_sizes(path)
        missing = [identifier for identifier in ids if identifier not in sizes]
        if missing:
            raise ValueError(f'Image metadata missing {len(missing)} requested frame(s): '
                             f'{", ".join(missing[:10])}; source: {path}')
        return {identifier: sizes[identifier] for identifier in ids}, [path]
    assets = [root / 'image_2' / f'{identifier}.png' for identifier in ids]
    return {identifier: png_image_size(path) for identifier, path in zip(ids, assets)}, assets
