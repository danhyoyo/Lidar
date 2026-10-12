"""Match the user's image ZIP and preserve exact KITTI training filenames."""
import importlib
from pathlib import Path
import struct
import sys
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def api():
    return importlib.import_module('tools.kitti_training_pipeline.extract_kitti_images')


def png_header(width=1242, height=375):
    return b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', width, height)


def test_finds_named_zip_in_another_mydrive_folder(tmp_path):
    raw = tmp_path / 'MyDrive/KITTI_DATASET_ZIP'
    raw.mkdir(parents=True)
    images = tmp_path / 'MyDrive/downloads/kitti_images.zip'
    images.parent.mkdir()
    images.touch()
    assert api().find_images_zip([raw, tmp_path / 'MyDrive']) == images.resolve()


def test_explicit_path_wins_and_ambiguous_search_requires_a_path(tmp_path):
    for name in ('first', 'second'):
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'kitti_images.zip').touch()
    with pytest.raises(ValueError, match='Multiple'):
        api().find_images_zip([tmp_path])
    explicit = tmp_path / 'first/kitti_images.zip'
    assert api().find_images_zip([tmp_path], explicit=explicit) == explicit.resolve()
    assert api().find_images_zip([tmp_path / 'missing']) is None


@pytest.mark.parametrize('prefix', ['training/image_2', 'wrapper/training/image_2', 'image_2', ''])
def test_extracts_training_left_camera_and_ignores_test_and_right_images(tmp_path, prefix):
    source = tmp_path / 'kitti_images.zip'
    with ZipFile(source, 'w') as archive:
        for identifier, size in [('000001', (1242, 375)), ('000002', (1224, 370))]:
            archive.writestr('/'.join(part for part in (prefix, f'{identifier}.png') if part), png_header(*size))
        archive.writestr('testing/image_2/000001.png', png_header(100, 50))
        archive.writestr('training/image_3/000001.png', png_header(99, 49))
    target = tmp_path / 'raw/training/image_2'
    report = api().extract_training_images(source, target, expected_count=2)
    assert report['image_count'] == 2
    assert report['image_sizes'] == {'000001': [1242, 375], '000002': [1224, 370]}
    assert (target / '000001.png').read_bytes() == png_header()
    assert not (tmp_path / 'raw/testing').exists()


@pytest.mark.parametrize('problem', ['testing_only', 'too_few', 'duplicate', 'invalid_png', 'traversal'])
def test_wrong_or_incomplete_archives_fail_before_writing_dataset(tmp_path, problem):
    source = tmp_path / 'kitti_images.zip'
    with ZipFile(source, 'w') as archive:
        if problem == 'testing_only':
            archive.writestr('testing/image_2/000001.png', png_header())
        else:
            path = 'training/image_2/000001.png' if problem != 'traversal' else '../training/image_2/000001.png'
            archive.writestr(path, b'invalid' if problem == 'invalid_png' else png_header())
            if problem == 'duplicate':
                archive.writestr('copy/training/image_2/000001.png', png_header())
    target = tmp_path / 'raw/training/image_2'
    with pytest.raises(ValueError):
        api().extract_training_images(source, target, expected_count=2 if problem == 'too_few' else 1)
    assert not target.exists()
