"""Archive-to-canonical-KITTI regressions, using actual small archives."""
import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
from zipfile import ZipFile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import kitti_notebook_data as data

IDS = ['000000', '000001']


def metadata(path, identifiers=IDS):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(format='kitti_image_sizes_v1', split='training', camera='image_2',
        image_count=len(identifiers), frames={i: dict(width=1242, height=375) for i in identifiers})))
    return path


def contents(component):
    if component == 'velodyne':
        return struct.pack('<ffff', 10, 0, 0, .5)
    if component == 'calib':
        return b'P2: 1 0 0 0 0 1 0 0 0 0 1 0\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n'
    return b'Car 0 0 0 0 0 100 100 1.5 1.6 4.2 0 1.5 20 0\n'


def zip_components(folder, *, nested=True):
    folder.mkdir(parents=True, exist_ok=True)
    for component in ('velodyne', 'calib', 'label_2'):
        suffix = '.bin' if component == 'velodyne' else '.txt'
        with ZipFile(folder / (component + '.zip'), 'w') as archive:
            for identifier in IDS:
                name = f'object/training/{component}/{identifier}{suffix}' if nested else identifier + suffix
                archive.writestr(name, contents(component))
                archive.writestr(f'testing/{component}/{identifier}{suffix}', b'testing must not be selected')


@pytest.mark.parametrize('nested', [True, False])
def test_drive_archive_directory_is_extracted_and_rerun_preserves_files(tmp_path, nested):
    source = tmp_path / 'KITTI_DATASET_ZIP'
    zip_components(source, nested=nested)
    sizes = metadata(tmp_path / 'sizes.json')
    result = data.prepare_dataset(source, tmp_path / 'local', IDS, fallback_metadata=sizes)
    root = Path(result['training_root'])
    assert result['staged'] is True
    assert result['counts'] == dict(velodyne=2, calib=2, label_2=2)
    assert not (root / 'image_2').exists()
    for component in ('velodyne', 'calib', 'label_2'):
        suffix = '.bin' if component == 'velodyne' else '.txt'
        assert (root / component / ('000000' + suffix)).read_bytes() == contents(component)
    frame = root / 'velodyne/000000.bin'
    stamp = frame.stat().st_mtime_ns
    repeated = data.prepare_dataset(source, tmp_path / 'local', IDS, fallback_metadata=sizes)
    assert frame.stat().st_mtime_ns == stamp
    assert repeated['reused_files'] == 6
    # Exact requested IDs, rather than all archive files or a random split.
    smaller = data.prepare_dataset(source, tmp_path / 'val_only', ['000001'], fallback_metadata=sizes)
    assert sorted(p.name for p in (Path(smaller['training_root']) / 'velodyne').iterdir()) == ['000001.bin']


def test_nested_tar_and_existing_raw_root_are_supported(tmp_path):
    archive = tmp_path / 'KITTI.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for component in ('velodyne', 'calib', 'label_2'):
            suffix = '.bin' if component == 'velodyne' else '.txt'
            for identifier in IDS:
                info = tarfile.TarInfo(f'KITTI/object/training/{component}/{identifier}{suffix}')
                value = contents(component)
                info.size = len(value)
                stream.addfile(info, io.BytesIO(value))
    sizes = metadata(tmp_path / 'sizes.json')
    result = data.prepare_dataset(archive, tmp_path / 'local', IDS, fallback_metadata=sizes)
    again = data.prepare_dataset(result['training_root'], tmp_path / 'unused', IDS, fallback_metadata=sizes)
    assert again['staged'] is False
    assert again['training_root'] == result['training_root']


def test_missing_components_and_missing_metadata_fail_before_partial_extraction(tmp_path):
    folder = tmp_path / 'archives'
    zip_components(folder)
    (folder / 'calib.zip').unlink()
    with pytest.raises(FileNotFoundError, match='calib'):
        data.prepare_dataset(folder, tmp_path / 'local', IDS, fallback_metadata=metadata(tmp_path / 'sizes.json'))
    assert not (tmp_path / 'local/training/velodyne/000000.bin').exists()
    zip_components(folder)
    with pytest.raises(FileNotFoundError, match='image_2|kitti_image_sizes'):
        data.prepare_dataset(folder, tmp_path / 'local', IDS)
    with pytest.raises(ValueError, match='missing'):
        data.prepare_dataset(folder, tmp_path / 'local', IDS, metadata_path=metadata(tmp_path / 'incomplete.json', ['000000']))


def test_selected_archive_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / 'velodyne.zip'
    with ZipFile(archive, 'w') as stream:
        stream.writestr('../training/velodyne/000000.bin', contents('velodyne'))
    with pytest.raises(ValueError, match='path'):
        data.prepare_dataset(archive, tmp_path / 'local', ['000000'], fallback_metadata=metadata(tmp_path / 'sizes.json'))


def test_disk_preflight_uses_uncompressed_selected_size(tmp_path, monkeypatch):
    from collections import namedtuple
    folder = tmp_path / 'archives'
    zip_components(folder)
    usage = namedtuple('usage', 'total used free')
    monkeypatch.setattr(data.shutil, 'disk_usage', lambda p: usage(1, 1, 0))
    with pytest.raises(RuntimeError, match='disk|Disk|space'):
        data.prepare_dataset(folder, tmp_path / 'local', IDS, fallback_metadata=metadata(tmp_path / 'sizes.json'))


def test_empty_ground_truth_label_is_a_valid_background_frame(tmp_path):
    folder = tmp_path / 'archives'
    zip_components(folder)
    with ZipFile(folder / 'label_2.zip', 'w') as archive:
        for identifier in IDS:
            archive.writestr('training/label_2/' + identifier + '.txt', b'')
    result = data.prepare_dataset(folder, tmp_path / 'local', IDS, fallback_metadata=metadata(tmp_path / 'sizes.json'))
    assert (Path(result['training_root']) / 'label_2/000000.txt').read_text() == ''


@pytest.mark.parametrize('multipart', [False, True])
def test_real_solid_or_multipart_rar_components_with_nested_paths(tmp_path, multipart):
    rar, unrar = shutil.which('rar'), shutil.which('unrar')
    if not rar or not unrar:
        pytest.skip('RAR command line tools needed to create/extract the fixture')
    pytest.importorskip('rarfile')
    folder, build = tmp_path / 'archives', tmp_path / 'build'
    folder.mkdir()
    for component in ('velodyne', 'calib', 'label_2'):
        suffix = '.bin' if component == 'velodyne' else '.txt'
        for identifier in IDS:
            path = build / 'training' / component / (identifier + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents(component) * (2048 if multipart and component == 'velodyne' else 1))
        options = ['-m0', '-v16k'] if multipart else ['-s']
        subprocess.run([rar, 'a', '-idq', *options, str(folder / (component + '.rar')), 'training/' + component],
                       cwd=build, check=True)
    result = data.prepare_dataset(folder, tmp_path / 'local', IDS, fallback_metadata=metadata(tmp_path / 'sizes.json'))
    assert (Path(result['training_root']) / 'label_2/000001.txt').read_bytes() == contents('label_2')
    assert result['counts']['velodyne'] == 2


@pytest.mark.parametrize('notebook_name,copy_setting,count', [
    ('OpenPCDet_KITTI_Validation_Benchmark', 'copy_val_to_local', 1),
    ('PIXOR_KITTI_Train_50Epochs', 'copy_data_to_local', 2),
])
def test_standalone_notebook_archive_cell_updates_settings_without_project_or_gpu(tmp_path, notebook_name, copy_setting, count):
    notebook = json.loads((ROOT / (notebook_name + '.ipynb')).read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    configuration = next(''.join(c['source']) for c in cells if 'SETTINGS = dict(' in ''.join(c['source']))
    configuration = configuration.replace("ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()", 'ON_COLAB = False')
    embedded = next(''.join(c['source']) for c in cells if 'embedded-source' in c['metadata'].get('tags', []))
    preparation = next(''.join(c['source']) for c in cells if 'DATA_RESULT = prepare_dataset(' in ''.join(c['source']))
    archives = tmp_path / 'KITTI_DATASET_ZIP'
    zip_components(archives)
    script = tmp_path / 'execute_notebook_data_cells.py'
    # Keep this integration small; canonical full manifests are checked elsewhere.
    overrides = "\nKITTI_ROOT = " + repr(str(archives)) + "\nEXTRACT_DIR = " + repr(str(tmp_path / 'local')) + "\n"
    overrides += "(WORK_DIR / 'splits/train.txt').write_text('000000\\n')\n"
    overrides += "(WORK_DIR / 'splits/val.txt').write_text('000001\\n')\n"
    assertions = f'''\nassert len(FRAME_IDS) == {count}
assert SETTINGS[{copy_setting!r}] is False
assert SETTINGS['kitti_root'] == str(Path(EXTRACT_DIR) / 'training')
assert Path(SETTINGS['image_sizes_json']).is_file()
assert json.loads(SETTINGS_PATH.read_text())['kitti_root'] == SETTINGS['kitti_root']
assert not (Path(SETTINGS['kitti_root']) / 'image_2').exists()
print('STANDALONE_DATA_OK')
'''
    script.write_text(configuration + '\n' + embedded + overrides + preparation + assertions)
    # -I drops PYTHONPATH and the surrounding project, exactly the standalone use case.
    result = subprocess.run([sys.executable, '-I', str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STANDALONE_DATA_OK' in result.stdout
