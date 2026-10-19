"""CPU contracts for the standalone OpenPCDet benchmark; no Torch required."""
import importlib.util
import csv
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
SPEC = importlib.util.spec_from_file_location('openpcdet_benchmark', ROOT / 'tools/openpcdet_benchmark.py')
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_split_normalization_preserves_ids_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'ids.txt'
    path.write_text('000001;kitti\n000003\n')
    assert benchmark.read_ids(path) == ['000001', '000003']
    path.write_text('000001\n000001;kitti\n')
    with pytest.raises(ValueError, match='Duplicate'):
        benchmark.read_ids(path)


def test_partition_rejects_leakage_and_malformed_ids(tmp_path):
    with pytest.raises(ValueError, match='overlap'):
        benchmark.check_partition(['000001'], ['000001'])
    path = tmp_path / 'ids.txt'
    for text in ('../123\n', '1\n', '000001;other\n', ''):
        path.write_text(text)
        with pytest.raises(ValueError):
            benchmark.read_ids(path)


def test_original_kitti_labels_retain_neighbors_and_dimension_order(tmp_path):
    path = tmp_path / '000000.txt'
    path.write_text('Van 0.1 1 -0.2 10 20 30 40 1.5 1.6 4.2 2 3 15 0.3\n'
                    'DontCare -1 -1 -10 0 0 20 20 -1 -1 -1 -1000 -1000 -1000 -10\n')
    annotation = benchmark.read_annotation(path)
    assert annotation['name'] == ['Van', 'DontCare']
    assert annotation['dimensions'][0] == [4.2, 1.5, 1.6]
    assert annotation['bbox'][0] == [10, 20, 30, 40]


def test_prediction_validation_rejects_bad_scores_and_accepts_empty(tmp_path):
    path = tmp_path / '000000.txt'
    path.write_text('')
    assert benchmark.read_annotation(path, prediction=True)['name'] == []
    path.write_text('Car -1 -1 0 0 0 10 20 1 2 3 4 5 6 0 1.2\n')
    with pytest.raises(ValueError, match='score'):
        benchmark.read_annotation(path, prediction=True)


def test_latency_summary_uses_mean_for_fps_and_linear_percentiles():
    result = benchmark.timing_summary([10, 20, 30, 40])
    assert result['mean_ms'] == 25
    assert result['p50_ms'] == 25
    assert result['p95_ms'] == pytest.approx(38.5)
    assert result['fps'] == 40
    assert benchmark.timing_summary([])['fps'] is None
    with pytest.raises(ValueError):
        benchmark.timing_summary([float('nan')])


def test_report_summary_keeps_failed_and_smoke_runs_out_of_benchmark_scores():
    report = {'name': 'second', 'status': 'ok', 'phase': 'smoke', 'benchmark_complete': False,
              'accuracy': {'3d': {'R40': {'map_moderate_percent': 80}}},
              'latency': {'input_to_detections': {'fps': 10}}}
    row = benchmark.summary_row(report)
    assert row['benchmark_complete'] is False
    assert row['ap3d_r40_moderate'] is None
    assert row['pipeline_fps'] is None
    assert benchmark.summary_row({'name': 'second', 'status': 'error', 'error': 'OOM'})['error'] == 'OOM'


def test_fingerprint_changes_with_timing_settings_and_checkpoint():
    common = {'checkpoint_sha256': 'a', 'warmup_frames': 10, 'phase': 'full'}
    assert benchmark.fingerprint(common) == benchmark.fingerprint(dict(reversed(list(common.items()))))
    assert benchmark.fingerprint(common) != benchmark.fingerprint({**common, 'warmup_frames': 20})
    assert benchmark.fingerprint(common) != benchmark.fingerprint({**common, 'checkpoint_sha256': 'b'})


def test_summary_exports_real_full_run_and_preserves_other_model_failures(tmp_path):
    settings = dict(work_dir=str(tmp_path / 'work'), output_root=str(tmp_path),
                    run_name='results', models=['pointpillars', 'second'])
    destination = tmp_path / 'results'
    report = dict(name='pointpillars', status='ok', phase='full', benchmark_complete=True,
                  frames=3769, model={'parameters': 123},
                  accuracy={'3d': {'R40': {'map_moderate_percent': 61.5}}},
                  latency={'input_to_detections': {'mean_ms': 20, 'fps': 50}})
    benchmark.write_json(destination / 'full/pointpillars/report.json', report)
    benchmark.write_json(destination / 'smoke/second/failure.json',
                         dict(name='second', status='error', phase='smoke', error='CUDA OOM'))
    benchmark.summarize(settings)
    rows = list(csv.DictReader((destination / 'comparison.csv').open()))
    assert rows[0]['ap3d_r40_moderate'] == '61.5'
    assert rows[0]['pipeline_fps'] == '50'
    assert rows[1]['error'] == 'CUDA OOM'
    assert rows[1]['ap3d_r40_moderate'] == ''


def test_png_header_and_training_root_support_raw_dataset_layout(tmp_path):
    import struct
    training = tmp_path / 'training'
    for directory in ('velodyne', 'image_2', 'calib', 'label_2'):
        (training / directory).mkdir(parents=True)
    assert benchmark.normalize_training_root(tmp_path) == training
    assert benchmark.normalize_training_root(training) == training
    png = training / 'image_2/000000.png'
    png.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\x0dIHDR' + struct.pack('>II', 1242, 375))
    assert benchmark.image_shape(png) == [375, 1242]
    png.write_bytes(b'not a png')
    with pytest.raises(ValueError):
        benchmark.image_shape(png)


def test_settings_reject_invalid_subset_warmup_and_dataset_write_locations(tmp_path):
    settings = dict(models=['pointpillars'], warmup_frames=10, max_frames=None,
                    run_name='benchmark', work_dir=str(tmp_path / 'work'),
                    output_root=str(tmp_path / 'output'), kitti_root=str(tmp_path / 'raw'))
    path = tmp_path / 'settings.json'
    benchmark.write_json(path, settings)
    assert benchmark.load_settings(path)['models'] == ['pointpillars']
    for patch in ({'warmup_frames': -1}, {'max_frames': 0}, {'run_name': '../raw'},
                  {'models': ['unknown']}, {'work_dir': str(tmp_path / 'raw/work')}):
        benchmark.write_json(path, {**settings, **patch})
        with pytest.raises(ValueError):
            benchmark.load_settings(path)


def test_cpp_compatibility_edits_are_idempotent_and_leave_evaluator_untouched(tmp_path):
    cpp = tmp_path / 'pcdet/ops/example/src/test.cpp'
    cpp.parent.mkdir(parents=True)
    cpp.write_text('AT_CHECK(x.type().is_cuda());\nAT_DISPATCH_FLOATING_TYPES(x.type(), "x", x.data<float>());\n')
    evaluator = tmp_path / 'pcdet/datasets/kitti/kitti_object_eval_python/eval.py'
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text('AT_CHECK = "leave this source unchanged"\n')
    assert benchmark.patch_extension_apis(tmp_path)[0]['path'].endswith('test.cpp')
    updated = cpp.read_text()
    assert 'TORCH_CHECK(x.is_cuda())' in updated
    assert 'AT_DISPATCH_FLOATING_TYPES(x.scalar_type()' in updated
    assert 'x.data_ptr<float>()' in updated
    assert benchmark.patch_extension_apis(tmp_path) == []
    assert evaluator.read_text() == 'AT_CHECK = "leave this source unchanged"\n'


def test_wrong_canonical_split_is_rejected_before_inference(tmp_path):
    folder = tmp_path / 'data/kitti/ImageSets'
    folder.mkdir(parents=True)
    (folder / 'train.txt').write_text('000000\n')
    (folder / 'val.txt').write_text('000001\n')
    with pytest.raises(ValueError, match='differs from the audited'):
        benchmark.audit_splits(tmp_path)


@pytest.mark.parametrize('use_metadata', [False, True])
def test_data_audit_stages_only_validation_and_keeps_gt_out_of_inference_infos(tmp_path, monkeypatch, use_metadata):
    import hashlib
    import shutil
    import struct
    work = tmp_path / 'work with spaces'
    repo = work / 'OpenPCDet'
    manifests = repo / 'data/kitti/ImageSets'
    manifests.mkdir(parents=True)
    train, val = ['000000'], ['000001', '000002']
    for name, identifiers in [('train', train), ('val', val)]:
        (manifests / (name + '.txt')).write_text('\n'.join(identifiers) + '\n')
    monkeypatch.setattr(benchmark, 'SPLIT_HASHES', {
        name: hashlib.sha256(('\n'.join(identifiers) + '\n').encode()).hexdigest()
        for name, identifiers in [('train', train), ('val', val)]})
    reference = 'pcdet/datasets/kitti/kitti_object_eval_python'
    destination = repo / reference
    destination.mkdir(parents=True)
    for name in benchmark.EVALUATOR_HASHES:
        shutil.copyfile(ROOT / 'tools/kitti_training_pipeline/kitti_reference' / name, destination / name)
    raw = tmp_path / 'raw dataset/training'
    png = b'\x89PNG\r\n\x1a\n' + b'\x00\x00\x00\x0dIHDR' + struct.pack('>II', 1242, 375)
    for identifier in val:
        for folder, suffix, data in [
            ('velodyne', '.bin', struct.pack('<ffff', 10, 0, 0, .5)),
            ('image_2', '.png', png), ('calib', '.txt', b'P2: 1 0 0 0 0 1 0 0 0 0 1 0\n'),
            ('label_2', '.txt', b'Van 0 0 0 10 20 30 40 1 2 3 4 5 6 0\n')]:
            if use_metadata and folder == 'image_2':
                continue
            path = raw / folder / (identifier + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    settings = dict(work_dir=str(work), output_root=str(tmp_path / 'output'), run_name='audit',
                    kitti_root=str(raw), copy_val_to_local=True)
    if use_metadata:
        path = raw / 'kitti_image_sizes.json'
        path.write_text(json.dumps(dict(format='kitti_image_sizes_v1', split='training', camera='image_2',
            image_count=len(val), frames={i: dict(width=1242, height=375) for i in val})))
        settings['image_sizes_json'] = str(path)
    benchmark.prepare_data(settings)
    result = json.loads((work / 'validation_assets.json').read_text())
    assert result['frame_ids'] == val
    assert result['source_training_root'] == str(raw)
    assert result['training_root'] != str(raw)
    assert all('annos' not in info for info in result['infos'])
    assert result['infos'][0]['image']['image_shape'] == [375, 1242]
    assert sorted(p.stem for p in (Path(result['training_root']) / 'velodyne').glob('*.bin')) == val
    assert benchmark.read_annotation(Path(result['training_root']) / 'label_2/000001.txt')['name'] == ['Van']


def test_notebook_configuration_cell_executes_without_gpu_or_project_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    notebook = json.loads((ROOT / 'OpenPCDet_KITTI_Validation_Benchmark.ipynb').read_text())
    code = next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and 'SETTINGS = dict(' in ''.join(c['source']))
    # Force the local default path, independent of the CI runner's /content mount.
    code = code.replace("ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()", 'ON_COLAB = False')
    namespace = {}
    exec(code, namespace)
    settings = benchmark.load_settings(namespace['SETTINGS_PATH'])
    assert settings['models'] == ['pointpillars', 'second', 'pointrcnn', 'parta2_anchor', 'pv_rcnn']
    assert settings['max_frames'] is None
    assert settings['warmup_frames'] == 10
    assert namespace['PROCESS_ENV']['CUDA_VISIBLE_DEVICES'] == '0'


def test_notebook_is_standalone_and_embeds_current_runner(tmp_path):
    notebook = json.loads((ROOT / 'OpenPCDet_KITTI_Validation_Benchmark.ipynb').read_text())
    namespace = {'WORK_DIR': tmp_path}
    embedded_cell = next(''.join(c['source']) for c in notebook['cells']
                         if 'embedded-source' in c.get('metadata', {}).get('tags', []))
    exec(embedded_cell, namespace)
    assert namespace['RUNNER_SOURCE'] == (ROOT / 'tools/openpcdet_benchmark.py').read_text()
    assert (tmp_path / 'openpcdet_benchmark.py').read_text() == namespace['RUNNER_SOURCE']
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<notebook>', 'exec')
            assert cell.get('outputs', []) == []


@pytest.mark.parametrize('name', ['OpenPCDet_KITTI_Validation_Benchmark', 'PIXOR_KITTI_Train_50Epochs'])
def test_installation_cell_is_short_and_embedded_source_displays_real_lines(name):
    notebook = json.loads((ROOT / (name + '.ipynb')).read_text())
    code_cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    setup = next(c for c in code_cells if "'setup', '--settings'" in ''.join(c['source']))
    setup_source = ''.join(setup['source'])
    assert len(setup_source.splitlines()) <= 30
    assert 'RUNNER_SOURCE =' not in setup_source and 'BUNDLE =' not in setup_source
    assert 'embedded-source' not in setup['metadata'].get('tags', [])
    embedded = [c for c in code_cells if 'embedded-source' in c['metadata'].get('tags', [])]
    assert embedded
    for c in embedded:
        assert c['metadata']['cellView'] == 'form'
        assert c['metadata']['jupyter']['source_hidden'] is True
        assert ''.join(c['source']).startswith('#@title ')
    # A repr() of a whole source file creates a many-thousand-character line.
    assert max(len(line) for c in code_cells for line in ''.join(c['source']).splitlines()) < 1000


def test_five_models_include_audited_additional_checkpoint_metadata():
    assert set(benchmark.MODELS) == {'pointpillars', 'second', 'pointrcnn', 'parta2_anchor', 'pv_rcnn'}
    for name, epoch, iteration in [('pointrcnn', 77, 17864), ('parta2_anchor', 80, 12400)]:
        benchmark.validate_checkpoint_metadata({'epoch': epoch, 'it': iteration, 'model_state': {}}, name)
        with pytest.raises(ValueError, match='metadata'):
            benchmark.validate_checkpoint_metadata({'epoch': 80, 'it': 999, 'model_state': {}}, name)


def test_environment_recovers_from_missing_ensurepip_and_partial_venv(tmp_path, monkeypatch):
    import subprocess
    from types import SimpleNamespace
    environment = tmp_path / 'env'
    python = environment / 'bin/python'
    python.parent.mkdir(parents=True)
    python.touch()  # A failed venv often leaves this executable behind.
    pip_available = False
    calls = []
    def check_pip(*args, **kwargs):
        return SimpleNamespace(returncode=0 if pip_available else 1)
    def execute(command, **kwargs):
        nonlocal pip_available
        command = list(map(str, command))
        calls.append(command)
        if command[2] == 'venv':
            raise subprocess.CalledProcessError(1, command)
        if command[2] == 'virtualenv':
            pip_available = True
            assert str(environment.parent / '.venv_bootstrap') in kwargs['env']['PYTHONPATH']
    monkeypatch.setattr(benchmark.subprocess, 'run', check_pip)
    monkeypatch.setattr(benchmark, 'run_command', execute)
    assert benchmark.ensure_python_environment(environment) == python
    assert any(command[2] == 'virtualenv' for command in calls)


def test_kitti_registry_import_recovers_without_argo2_dependency(tmp_path):
    import hashlib
    import subprocess
    repo = tmp_path / 'OpenPCDet'
    datasets = repo / 'pcdet/datasets'
    for name in ('pcdet', 'pcdet/datasets', 'pcdet/datasets/kitti', 'pcdet/datasets/argo2'):
        directory = repo / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / '__init__.py').write_text('')
    (datasets / 'dataset.py').write_text('class DatasetTemplate: pass\n')
    (datasets / 'kitti/kitti_dataset.py').write_text('class KittiDataset: pass\n')
    (datasets / 'argo2/argo2_dataset.py').write_text('from av2.utils.io import read_feather\n')
    registry = datasets / '__init__.py'
    original = ('from .dataset import DatasetTemplate\n'
                'from .kitti.kitti_dataset import KittiDataset\n'
                'from .argo2.argo2_dataset import Argo2Dataset\n'
                "__all__ = {\n    'DatasetTemplate': DatasetTemplate,\n"
                "    'KittiDataset': KittiDataset,\n    'Argo2Dataset': Argo2Dataset\n}\n"
                'def build_dataloader():\n    return __all__["KittiDataset"]()\n')
    registry.write_text(original)
    evaluator = datasets / 'kitti/evaluator.py'
    evaluator.write_text('REFERENCE_UNCHANGED = True\n')
    code = ('import sys; sys.path.insert(0, sys.argv[1]); '
            'from pcdet.datasets import KittiDataset, build_dataloader; '
            'assert type(build_dataloader()) is KittiDataset; '
            'assert "pcdet.datasets.argo2.argo2_dataset" not in sys.modules')
    command = [sys.executable, '-I', '-c', code, str(repo)]
    failed = subprocess.run(command, capture_output=True, text=True)
    assert failed.returncode != 0 and "No module named 'av2'" in failed.stderr

    manifest = benchmark.patch_kitti_dataset_imports(repo)
    subprocess.run(command, capture_output=True, text=True, check=True)
    assert evaluator.read_text() == 'REFERENCE_UNCHANGED = True\n'
    assert (datasets / 'kitti/kitti_dataset.py').read_text() == 'class KittiDataset: pass\n'
    assert manifest['before_sha256'] == hashlib.sha256(original.encode()).hexdigest()
    assert manifest['after_sha256'] == benchmark.sha256(registry)
    assert manifest['disabled_datasets'] == ['Argo2Dataset']
    assert benchmark.patch_kitti_dataset_imports(repo) == manifest
    assert json.loads((tmp_path / 'dataset_import_compatibility.json').read_text()) == manifest


def test_notebook_displays_and_logs_child_traceback(tmp_path, monkeypatch, capsys):
    import subprocess
    monkeypatch.chdir(tmp_path)
    notebook = json.loads((ROOT / 'OpenPCDet_KITTI_Validation_Benchmark.ipynb').read_text())
    code = next(''.join(c['source']) for c in notebook['cells']
                if c['cell_type'] == 'code' and 'SETTINGS = dict(' in ''.join(c['source']))
    code = code.replace("ON_COLAB = 'COLAB_RELEASE_TAG' in os.environ or Path('/content').is_dir()", 'ON_COLAB = False')
    namespace = {}
    exec(code, namespace)
    capsys.readouterr()
    command = [sys.executable, '-c',
               'import sys; print("before import", flush=True); '
               'raise ModuleNotFoundError("No module named av2")']
    with pytest.raises(subprocess.CalledProcessError) as caught:
        namespace['execute'](command)
    displayed = capsys.readouterr().out
    assert 'before import' in displayed
    assert 'ModuleNotFoundError: No module named av2' in displayed
    assert 'ModuleNotFoundError: No module named av2' in caught.value.output
    log = namespace['WORK_DIR'] / 'logs/commands.log'
    assert 'ModuleNotFoundError: No module named av2' in log.read_text()
    assert '[exit code: 1]' in log.read_text()

    freeze = tmp_path / 'freeze.txt'
    with freeze.open('w') as stream:
        result = namespace['execute']([sys.executable, '-c', 'print("numpy==2.2.6")'], stdout=stream)
    assert result.returncode == 0
    assert freeze.read_text() == 'numpy==2.2.6\n'
    assert 'numpy==2.2.6' not in capsys.readouterr().out.splitlines()
