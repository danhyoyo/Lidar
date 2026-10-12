"""Per-frame camera dimensions can replace PNGs without changing projection."""
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tools.kitti_training_pipeline import image_metadata


def write_metadata(path, frames=None):
    frames = frames if frames is not None else {
        '000000': {'width': 1224, 'height': 370},
        '000001': {'width': 1242, 'height': 375}}
    data = dict(format='kitti_image_sizes_v1', split='training', camera='image_2',
                image_count=len(frames), frames=frames)
    path.write_text(json.dumps(data))
    return data


def test_json_only_inputs_resolve_both_dataset_and_training_roots(tmp_path):
    root = tmp_path / 'training'
    root.mkdir()
    metadata = root / 'kitti_image_sizes.json'
    write_metadata(metadata)
    for candidate in (root, tmp_path):
        sizes, assets = image_metadata.load_image_sizes(candidate, ['000001', '000000'])
        assert sizes == {'000001': (1242, 375), '000000': (1224, 370)}
        assert assets == [metadata.resolve()]
    assert not (root / 'image_2').exists()


@pytest.mark.parametrize('problem', ['format', 'camera', 'split', 'count', 'zero', 'bool',
                                     'float', 'identifier', 'missing', 'duplicate'])
def test_invalid_or_incomplete_metadata_is_rejected(tmp_path, problem):
    path = tmp_path / 'kitti_image_sizes.json'
    data = write_metadata(path)
    if problem in ('format', 'camera', 'split'):
        data[problem] = 'wrong'
    elif problem == 'count':
        data['image_count'] = 3
    elif problem in ('zero', 'bool', 'float'):
        data['frames']['000000']['width'] = {'zero': 0, 'bool': True, 'float': 1224.5}[problem]
    elif problem == 'identifier':
        data['frames']['bad'] = data['frames'].pop('000000')
    elif problem == 'missing':
        data['frames'].pop('000000')
        data['image_count'] = 1
    path.write_text(json.dumps(data))
    if problem == 'duplicate':
        path.write_text(path.read_text().replace('"width": 1224', '"width": 1224, "width": 1242'))
    with pytest.raises(ValueError):
        image_metadata.load_image_sizes(tmp_path, ['000000'])


def test_explicit_metadata_path_takes_precedence_and_missing_path_is_an_error(tmp_path):
    explicit = tmp_path / 'custom.json'
    write_metadata(explicit)
    write_metadata(tmp_path / 'kitti_image_sizes.json', {'000002': {'width': 10, 'height': 20}})
    sizes, assets = image_metadata.load_image_sizes(tmp_path, ['000000'], metadata_path=explicit)
    assert sizes['000000'] == (1224, 370) and assets == [explicit.resolve()]
    with pytest.raises(FileNotFoundError):
        image_metadata.load_image_sizes(tmp_path, ['000000'], metadata_path=tmp_path / 'absent.json')


def test_png_fallback_and_json_produce_identical_annotations(tmp_path):
    from tools.kitti_training_pipeline.kitti_box_conversion import Calibration, predictions_to_annotation
    folder = tmp_path / 'image_2'
    folder.mkdir()
    png = folder / '000000.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 1224, 370))
    png_sizes, assets = image_metadata.load_image_sizes(tmp_path, ['000000'])
    assert assets == [png.resolve()]
    write_metadata(tmp_path / 'kitti_image_sizes.json')
    json_sizes, _ = image_metadata.load_image_sizes(tmp_path, ['000000'])
    transform = np.array([[0., -1., 0., 0.], [0., 0., -1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    calibration = Calibration(transform, np.array([[700., 0., 620., 0.], [0., 700., 180., 0.], [0., 0., 1., 0.]]))
    rows = np.array([[0, .9, 10, 0, -1.5, 1.8, 4., 1.5, 0.]])
    left = predictions_to_annotation(rows, calibration, {'Car': 0}, image_size=png_sizes['000000'])
    right = predictions_to_annotation(rows, calibration, {'Car': 0}, image_size=json_sizes['000000'])
    assert len(left['name']) == 1
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


def test_bundled_user_metadata_covers_all_training_ids():
    path = Path(__file__).resolve().parents[1] / 'configs/datasets/kitti_image_sizes.json'
    sizes = image_metadata.read_image_sizes(path)
    assert set(sizes) == {f'{i:06d}' for i in range(7481)}
    assert sizes['000000'] == (1224, 370)
    assert len(set(sizes.values())) == 4
