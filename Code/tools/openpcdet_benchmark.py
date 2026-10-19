#!/usr/bin/env python3
"""Standalone KITTI checkpoint evaluation and explicit sequential GPU profiling.

Only the GPU/setup commands import Torch, NumPy or OpenPCDet. The notebook embeds
this file so it can be used without uploading the surrounding project.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import struct
import subprocess
import sys
import time
import traceback


REVISION = '233f849829b6ac19afb8af8837a0246890908755'
CLASSES = ('Car', 'Pedestrian', 'Cyclist')
DIFFICULTIES = ('Easy', 'Moderate', 'Hard')
SPLIT_HASHES = {
    'train': 'e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c',
    'val': '6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8',
}
EVALUATOR_HASHES = {
    'eval.py': 'd9260a41f68083d685fc43fecac1c417b14a1bc53df4915a46b2924f3a6a3c64',
    'rotate_iou.py': 'd9b2820b5be5d15ee7233147643134c5d73eff129c773aef4b59ce73a8ecd7f4',
}
MODELS = {
    'pointpillars': dict(config='pointpillar.yaml', filename='pointpillar_7728.pth',
        drive_id='1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm', epoch=80, it=9280),
    'second': dict(config='second.yaml', filename='second_7862.pth',
        drive_id='1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR', epoch=80, it=9280),
    'pointrcnn': dict(config='pointrcnn.yaml', filename='pointrcnn_official.pth',
        drive_id='1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU', epoch=77, it=17864),
    'parta2_anchor': dict(config='PartA2.yaml', filename='parta2_anchor_official.pth',
        drive_id='10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY', epoch=80, it=12400),
    'pv_rcnn': dict(config='pv_rcnn.yaml', filename='pv_rcnn_8369.pth',
        drive_id='1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-', epoch=77, it=17864),
}
STAGES = ('preprocess', 'host_to_device', 'network', 'decode_nms', 'input_to_detections')


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_ids(path):
    result = []
    for line in Path(path).read_text().splitlines():
        fields = line.strip().split(';')
        if not line.strip():
            continue
        if len(fields) > 2 or (len(fields) == 2 and fields[1].strip() != 'kitti'):
            raise ValueError(f'Invalid KITTI manifest row: {line!r}')
        identifier = fields[0].strip()
        if not re.fullmatch(r'[0-9]{6}', identifier):
            raise ValueError(f'Invalid KITTI frame ID: {identifier!r}')
        result.append(identifier)
    if not result:
        raise ValueError(f'Empty split: {path}')
    if len(result) != len(set(result)):
        raise ValueError(f'Duplicate frame IDs: {path}')
    return result


def check_partition(train, val):
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f'Train/validation overlap: {overlap[:10]}')


def audit_splits(repo):
    lists = {}
    evidence = {}
    for name in ('train', 'val'):
        path = Path(repo) / 'data/kitti/ImageSets' / (name + '.txt')
        lists[name] = read_ids(path)
        canonical = hashlib.sha256(('\n'.join(lists[name]) + '\n').encode()).hexdigest()
        if canonical != SPLIT_HASHES[name]:
            raise ValueError(f'{name} differs from the audited KITTI split: {path}')
        evidence[name] = dict(count=len(lists[name]), canonical_ids_sha256=canonical)
    check_partition(lists['train'], lists['val'])
    return lists, evidence


def verify_evaluator(repo):
    directory = Path(repo) / 'pcdet/datasets/kitti/kitti_object_eval_python'
    for name, expected in EVALUATOR_HASHES.items():
        if sha256(directory / name) != expected:
            raise ValueError(f'Pinned evaluator modified: {directory / name}')
    return dict(revision=REVISION, source_sha256=EVALUATOR_HASHES)


def read_annotation(path, *, prediction=False):
    """Keep original full-frame GT, including Van, Person_sitting and DontCare."""
    result = {key: [] for key in ('name', 'truncated', 'occluded', 'alpha', 'bbox',
                                 'dimensions', 'location', 'rotation_y', 'score')}
    for line_number, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != (16 if prediction else 15):
            raise ValueError(f'Invalid KITTI row at {path}:{line_number}')
        values = [float(value) for value in fields[1:]]
        if not all(math.isfinite(value) for value in values) or values[1] != int(values[1]):
            raise ValueError(f'Invalid numeric annotation at {path}:{line_number}')
        if prediction and not 0 <= values[-1] <= 1:
            raise ValueError(f'Invalid prediction score at {path}:{line_number}')
        if prediction and any(value <= 0 for value in values[7:10]):
            raise ValueError(f'Invalid prediction dimensions at {path}:{line_number}')
        row = dict(name=fields[0], truncated=values[0], occluded=int(values[1]),
                   alpha=values[2], bbox=values[3:7],
                   dimensions=[values[9], values[7], values[8]],  # KITTI h,w,l -> evaluator l,h,w
                   location=values[10:13], rotation_y=values[13],
                   score=values[14] if prediction else 0.0)
        for key, value in row.items():
            result[key].append(value)
    return result


def array_annotation(annotation):
    import numpy as np
    result = {}
    for key, value in annotation.items():
        dtype = str if key == 'name' else (np.int64 if key == 'occluded' else np.float64)
        shape = (-1, 4) if key == 'bbox' else ((-1, 3) if key in ('dimensions', 'location') else (-1,))
        result[key] = np.asarray(value, dtype=dtype).reshape(shape)
    return result


def timing_summary(values):
    if not values:
        return dict(samples=0, mean_ms=None, p50_ms=None, p95_ms=None, p99_ms=None, fps=None)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('Latency samples must be finite and nonnegative')
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lower, upper = math.floor(index), math.ceil(index)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    mean = statistics.fmean(values)
    return dict(samples=len(values), mean_ms=mean, p50_ms=percentile(.5),
                p95_ms=percentile(.95), p99_ms=percentile(.99),
                fps=1000 / mean if mean > 0 else None)


def summary_row(report):
    complete = report.get('status') == 'ok' and report.get('benchmark_complete') is True
    row = dict(model=report.get('name'), status=report.get('status'), phase=report.get('phase'),
               benchmark_complete=complete, frames=report.get('frames'),
               parameters=report.get('model', {}).get('parameters'), error=report.get('error'))
    for mode in ('3d', 'bev'):
        for recall in ('R11', 'R40'):
            metrics = report.get('accuracy', {}).get(mode, {}).get(recall, {}) if complete else {}
            row[f'ap{mode}_{recall.lower()}_moderate'] = metrics.get('map_moderate_percent')
            for name in CLASSES:
                row[f'ap{mode}_{recall.lower()}_{name.lower()}_moderate'] = metrics.get('per_class', {}).get(name, {}).get('Moderate')
    for stage, short in (('network', 'network'), ('decode_nms', 'decode_nms'),
                         ('input_to_detections', 'pipeline')):
        timing = report.get('latency', {}).get(stage, {}) if complete else {}
        for statistic in ('mean_ms', 'p50_ms', 'p95_ms', 'p99_ms', 'fps'):
            row[f'{short}_{statistic}'] = timing.get(statistic)
    return row


def run_command(command, *, cwd=None, env=None):
    import shlex
    print('+ ' + shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def ensure_python_environment(environment):
    """Create a pip-enabled overlay, including runtimes without ensurepip."""
    environment = Path(environment)
    python = environment / 'bin/python'
    def usable():
        return python.exists() and subprocess.run([str(python), '-m', 'pip', '--version'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if usable():
        return python
    try:
        run_command([sys.executable, '-m', 'venv', '--system-site-packages', environment])
        if not usable():
            raise RuntimeError('venv did not provide pip')
    except (subprocess.CalledProcessError, RuntimeError):
        print('Bootstrapping virtualenv because this Python has no working ensurepip.', flush=True)
        bootstrap = environment.parent / '.venv_bootstrap'
        run_command([sys.executable, '-m', 'pip', 'install', '--target', bootstrap, 'virtualenv>=20,<21'])
        env = os.environ.copy()
        env['PYTHONPATH'] = str(bootstrap) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        run_command([sys.executable, '-m', 'virtualenv', '--system-site-packages', environment], env=env)
        if not usable():
            raise RuntimeError('Could not create a pip-enabled environment; use a runtime with venv/virtualenv support')
    return python


def load_settings(path):
    result = json.loads(Path(path).read_text())
    if not result.get('models') or any(name not in MODELS for name in result['models']):
        raise ValueError('Choose models from: ' + ', '.join(MODELS))
    if type(result.get('warmup_frames')) is not int or result['warmup_frames'] < 1:
        raise ValueError('warmup_frames must be a positive integer')
    maximum = result.get('max_frames')
    if maximum is not None and (type(maximum) is not int or maximum < 1):
        raise ValueError('max_frames must be null or a positive integer')
    if type(result.get('smoke_frames', 3)) is not int or result.get('smoke_frames', 3) < 2:
        raise ValueError('smoke_frames must be at least two')
    if type(result.get('build_jobs', 2)) is not int or result.get('build_jobs', 2) < 1:
        raise ValueError('build_jobs must be a positive integer')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', result['run_name']) or result['run_name'] in ('.', '..'):
        raise ValueError('run_name must be a simple directory name')
    for key in ('work_dir', 'output_root', 'kitti_root'):
        result[key] = str(Path(result[key]).expanduser().resolve())
    # Work and output directories must never be placed inside the source dataset.
    raw = Path(result['kitti_root'])
    for key in ('work_dir', 'output_root'):
        path_value = Path(result[key])
        if path_value == raw or raw in path_value.parents:
            raise ValueError(f'{key} must be outside the KITTI source dataset')
    return result


def paths_for(settings):
    work = Path(settings['work_dir'])
    return work, work / 'OpenPCDet', Path(settings['output_root']) / settings['run_name']


def normalize_training_root(path, image_sizes_json=None):
    path = Path(path).expanduser().resolve()
    if (path / 'training').is_dir():
        path = path / 'training'
    required = ('velodyne', 'label_2', 'calib')
    if any(not (path / name).is_dir() for name in required):
        raise FileNotFoundError(f'KITTI needs {required} under {path}')
    metadata = Path(image_sizes_json).expanduser().resolve() if image_sizes_json else next((p for p in
        (path / 'kitti_image_sizes.json', path.parent / 'kitti_image_sizes.json') if p.is_file()), None)
    if not (path / 'image_2').is_dir() and (metadata is None or not metadata.is_file()):
        raise FileNotFoundError(f'KITTI needs image_2 PNGs or per-frame kitti_image_sizes.json under {path}')
    return path


def image_shape(path):
    with Path(path).open('rb') as stream:
        header = stream.read(24)
    if len(header) < 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Expected original KITTI PNG: {path}')
    width, height = struct.unpack('>II', header[16:24])
    if not width or not height:
        raise ValueError(f'Invalid image dimensions: {path}')
    return [height, width]


def load_image_dimensions(root, identifiers, metadata_path=None):
    from kitti_training_pipeline.image_metadata import load_image_sizes
    dimensions, sources = load_image_sizes(root, identifiers, metadata_path=metadata_path)
    shapes = {identifier: [height, width] for identifier, (width, height) in dimensions.items()}
    metadata = str(sources[0]) if len(sources) == 1 and sources[0].suffix.lower() == '.json' else None
    return shapes, sources, metadata


def prepare_data(settings):
    work, repo, output = paths_for(settings)
    lists, split_evidence = audit_splits(repo)
    verify_evaluator(repo)
    source_root = normalize_training_root(settings['kitti_root'], settings.get('image_sizes_json'))
    shapes, image_sources, metadata = load_image_dimensions(source_root, lists['val'], settings.get('image_sizes_json'))
    file_specs = [('velodyne', '.bin'), ('calib', '.txt'), ('label_2', '.txt')]
    if metadata is None:
        file_specs.append(('image_2', '.png'))
    root = source_root
    if settings.get('copy_val_to_local', False):
        root = work / 'kitti_validation/training'
        if root.resolve() == source_root:
            raise ValueError('Validation staging destination must differ from the source dataset.')
        pending = []
        for identifier in lists['val']:
            for directory, extension in file_specs:
                original = source_root / directory / (identifier + extension)
                target = root / directory / original.name
                stat = original.stat()
                if not target.exists() or (target.stat().st_size, target.stat().st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                    pending.append((original, target, stat.st_size))
        required = sum(size for _, _, size in pending)
        if shutil.disk_usage(work).free < required + 512 * 1024 * 1024:
            raise RuntimeError(f'Not enough local disk for validation staging ({required / 1e9:.2f} GB).')
        for index, (original, target, _) in enumerate(pending):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + '.copying')
            shutil.copy2(original, temporary)
            temporary.replace(target)
            if (index + 1) % 1000 == 0:
                print(f'Validation staging: {index + 1}/{len(pending)} files', flush=True)
    infos = []
    assets = hashlib.sha256()
    for path in image_sources:
        assets.update(str(path).encode())
        if path.suffix.lower() == '.json':
            assets.update(path.read_bytes())
        else:
            with path.open('rb') as stream:
                assets.update(stream.read(24))
    for identifier in lists['val']:
        for directory, extension in file_specs:
            path = root / directory / (identifier + extension)
            if not path.is_file():
                raise FileNotFoundError(path)
            if directory == 'velodyne' and (path.stat().st_size == 0 or path.stat().st_size % 16):
                raise ValueError(f'Invalid float32 XYZI point cloud: {path}')
            assets.update((str(path) + ':' + str(path.stat().st_size) + ':' + str(path.stat().st_mtime_ns)).encode())
            if directory in ('calib', 'label_2'):
                assets.update(path.read_bytes())
            if directory == 'label_2':
                read_annotation(path)
        infos.append(dict(point_cloud=dict(num_features=4, lidar_idx=identifier),
                          image=dict(image_idx=identifier, image_shape=shapes[identifier])))
    result = dict(training_root=str(root), source_training_root=str(source_root), splits=split_evidence, frame_ids=lists['val'], infos=infos,
                  asset_signature=assets.hexdigest(), image_sizes_json=metadata,
                  asset_signature_scope='Paths, sizes and mtimes of validation inputs, complete label/calibration bytes, and image metadata JSON or PNG headers; not a full point-cloud content hash.')
    write_json(work / 'validation_assets.json', result)
    write_json(output / 'validation_assets.json', {k: v for k, v in result.items() if k != 'infos'})
    print(f'Validated {len(infos)} frames; train=3712, val=3769, no overlap. No GT database or train preprocessing needed.', flush=True)


def patch_extension_apis(repo):
    """Mechanical C++ compatibility only; evaluator and Python models untouched."""
    records = []
    for path in sorted((Path(repo) / 'pcdet/ops').rglob('*')):
        if path.suffix not in ('.cpp', '.cu', '.h'):
            continue
        original = path.read_text()
        updated = original.replace('.type().is_cuda()', '.is_cuda()')
        updated = re.sub(r'\bAT_CHECK\b', 'TORCH_CHECK', updated)
        updated = re.sub(r'\.data<', '.data_ptr<', updated)
        updated = re.sub(r'(AT_DISPATCH_\w+\s*\(\s*\w+)\.type\(\)', r'\1.scalar_type()', updated)
        if updated != original:
            records.append(dict(path=str(path.relative_to(repo)),
                                before_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                after_sha256=hashlib.sha256(updated.encode()).hexdigest()))
            path.write_text(updated)
    manifest = Path(repo).parent / 'extension_compatibility.json'
    if records or not manifest.exists():
        write_json(manifest, records)
    return records


def patch_kitti_dataset_imports(repo):
    """Avoid optional non-KITTI SDKs in this KITTI-only benchmark checkout."""
    repo = Path(repo)
    path = repo / 'pcdet/datasets/__init__.py'
    manifest_path = repo.parent / 'dataset_import_compatibility.json'
    original = path.read_text()
    before = hashlib.sha256(original.encode()).hexdigest()
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if before == manifest['after_sha256']:
            return manifest
        if before != manifest['before_sha256']:
            raise RuntimeError('Dataset registry changed since the recorded compatibility patch.')
    optional = {
        'nuscenes.nuscenes_dataset': 'NuScenesDataset', 'waymo.waymo_dataset': 'WaymoDataset',
        'pandaset.pandaset_dataset': 'PandasetDataset', 'lyft.lyft_dataset': 'LyftDataset',
        'once.once_dataset': 'ONCEDataset', 'argo2.argo2_dataset': 'Argo2Dataset',
        'custom.custom_dataset': 'CustomDataset',
    }
    tree = ast.parse(original)
    removed, imports, entries = set(), set(), set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module in optional:
            name = optional[node.module]
            if len(node.names) != 1 or node.names[0].name != name or node.names[0].asname:
                raise RuntimeError('Unexpected optional dataset import; inspect the pinned checkout.')
            imports.add(name)
            removed.update(range(node.lineno, node.end_lineno + 1))
        if (isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == '__all__' for target in node.targets)
                and isinstance(node.value, ast.Dict)):
            for key, value in zip(node.value.keys, node.value.values):
                if isinstance(key, ast.Constant) and key.value in optional.values():
                    if not isinstance(value, ast.Name) or value.id != key.value or key.lineno != value.end_lineno:
                        raise RuntimeError('Unexpected optional dataset registry entry.')
                    # The pinned registry has one entry per line; retain all other bytes.
                    if original.splitlines()[key.lineno - 1].strip().startswith(('"', "'")):
                        entries.add(key.value)
                        removed.add(key.lineno)
    if imports != entries:
        raise RuntimeError('Optional dataset imports and registry entries do not match.')
    updated = ''.join(line for number, line in enumerate(original.splitlines(keepends=True), 1)
                      if number not in removed)
    ast.parse(updated)  # Validate before changing an existing checkout.
    manifest = dict(path='pcdet/datasets/__init__.py', scope='KITTI-only dataset registry',
                    disabled_datasets=sorted(imports), before_sha256=before,
                    after_sha256=hashlib.sha256(updated.encode()).hexdigest())
    if updated != original:
        path.write_text(updated)
    write_json(manifest_path, manifest)
    return manifest


def setup(settings):
    if platform.system() != 'Linux' or not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError('Use a Linux GPU runtime with Python 3.10–3.13 and CUDA PyTorch (e.g. Colab).')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Enable a GPU runtime first. CUDA PyTorch is required.')
    nvcc = shutil.which('nvcc')
    if not nvcc:
        raise RuntimeError('nvcc is missing. Install the CUDA toolkit matching your PyTorch CUDA major version, then rerun setup.')
    nvcc_text = subprocess.check_output([nvcc, '--version'], text=True)
    match = re.search(r'release (\d+)\.(\d+)', nvcc_text)
    if not match or not torch.version.cuda or int(match[1]) != int(torch.version.cuda.split('.')[0]):
        raise RuntimeError(f'CUDA toolkit / PyTorch major mismatch: nvcc={nvcc_text}, torch CUDA={torch.version.cuda}')
    work, repo, output = paths_for(settings)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if not repo.exists():
        run_command(['git', 'clone', '--filter=blob:none', 'https://github.com/open-mmlab/OpenPCDet.git', repo])
        run_command(['git', 'checkout', '--detach', REVISION], cwd=repo)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'{repo} must be at {REVISION}; use another work_dir for an existing checkout.')
    verify_evaluator(repo)
    audit_splits(repo)
    patch_kitti_dataset_imports(repo)
    environment = work / 'env'
    python = ensure_python_environment(environment)
    constraints = work / 'constraints.txt'
    constraints.write_text('torch==' + torch.__version__ + '\n')
    # The overlay inherits the user's CUDA Torch rather than replacing it.
    cuda = tuple(map(int, torch.version.cuda.split('.')[:2]))
    default_spconv = 'spconv-cu126==2.3.8' if cuda >= (12, 6) else (
        'spconv-cu124==2.3.8' if cuda >= (12, 4) else (
        'spconv-cu120==2.3.8' if cuda >= (12, 0) else 'spconv-cu118==2.3.8'))
    packages = ['numpy>=1.26,<2.3', 'numba>=0.61.2,<0.62', 'scipy', 'scikit-image>=0.24',
                'setuptools<81', 'wheel', 'ninja', 'easydict', 'pyyaml', 'tqdm',
                'tensorboardX', 'SharedArray', 'pyquaternion', 'gdown>=5.2,<6',
                'kornia', 'einops', 'timm', 'torchvision',
                settings.get('spconv_package') or default_spconv]
    run_command([python, '-m', 'pip', 'install', '--constraint', constraints, *packages])
    patch_extension_apis(repo)
    env = os.environ.copy()
    env['MAX_JOBS'] = str(settings.get('build_jobs', 2))
    major, minor = torch.cuda.get_device_capability()
    env['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    run_command([python, 'setup.py', 'build_ext', '--inplace'], cwd=repo, env=env)
    verify_evaluator(repo)
    write_json(work / 'setup.json', dict(python=str(python), revision=REVISION,
               torch=torch.__version__, torch_cuda=torch.version.cuda, nvcc=nvcc_text,
               spconv_package=packages[-1], extensions_patch_manifest=str(work / 'extension_compatibility.json')))
    print(f'Environment ready: {python}', flush=True)


def activate_repo(repo):
    repo = Path(repo).resolve()
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != REVISION:
        raise RuntimeError('OpenPCDet revision changed; rerun setup with the pinned checkout.')
    sys.path.insert(0, str(repo))
    os.chdir(repo / 'tools')  # cfg_from_yaml_file resolves its _BASE_CONFIG_ here.
    verify_evaluator(repo)
    patch_kitti_dataset_imports(repo)


def runtime_info():
    import torch
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'numba', 'numba-cuda', 'spconv-cu126',
                 'spconv-cu124', 'spconv-cu120', 'spconv-cu118', 'cumm-cu126', 'gdown'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return dict(python=platform.python_version(), executable=sys.executable,
                packages=versions, torch_cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                cpu=platform.processor(), cpu_count=os.cpu_count(),
                cuda_home=os.environ.get('CUDA_HOME'),
                nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
                driver_version=subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).strip(),
                cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                allow_tf32=False, precision='fp32', batch_size=1)


def configure_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this benchmark.')
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    if os.environ.get('CUDA_LAUNCH_BLOCKING', '0') not in ('', '0'):
        raise RuntimeError('Unset CUDA_LAUNCH_BLOCKING for benchmark timing; this runner synchronizes explicitly.')
    try:
        import spconv.constants
        spconv.constants.SPCONV_ALLOW_TF32 = False
    except ImportError:
        pass


def gpu_check(settings):
    import numpy as np
    import torch
    work, repo, output = paths_for(settings)
    activate_repo(repo)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work / 'dataset_import_compatibility.json', output / 'dataset_import_compatibility.json')
    configure_gpu()
    from pcdet.ops.iou3d_nms import iou3d_nms_utils
    from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
    boxes = torch.tensor([[0, 0, 0, 2, 2, 2, 0]], device='cuda', dtype=torch.float32)
    overlap = iou3d_nms_utils.boxes_iou3d_gpu(boxes, boxes).item()
    reference = rotate_iou_gpu_eval(np.array([[0, 0, 2, 2, 0]], dtype=np.float32),
                                    np.array([[0, 0, 2, 2, 0]], dtype=np.float32))[0, 0]
    if abs(overlap - 1) > 1e-5 or abs(float(reference) - 1) > 1e-5:
        raise RuntimeError('CUDA/reference overlap smoke failed.')
    info = runtime_info()
    write_json(output / 'environment.json', info)
    print(json.dumps(info, indent=2), flush=True)


def checkpoint_path(settings, name):
    override = settings.get('checkpoint_overrides', {}).get(name)
    return Path(override).expanduser().resolve() if override else paths_for(settings)[0] / 'checkpoints' / MODELS[name]['filename']


def validate_checkpoint_metadata(checkpoint, name):
    spec = MODELS[name]
    if (not isinstance(checkpoint, dict) or checkpoint.get('epoch') != spec['epoch']
            or checkpoint.get('it') != spec['it'] or not isinstance(checkpoint.get('model_state'), dict)):
        raise ValueError(f'{name}: checkpoint metadata differs from the audited official file.')


def download_checkpoints(settings):
    import gdown
    import torch
    work, repo, output = paths_for(settings)
    for name in settings['models']:
        spec = MODELS[name]
        path = checkpoint_path(settings, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = path
        if not path.exists():
            if settings.get('checkpoint_overrides', {}).get(name):
                raise FileNotFoundError(path)
            temporary = path.with_suffix('.download')
            if gdown.download(id=spec['drive_id'], output=str(temporary), quiet=False) is None:
                raise RuntimeError(f'Download failed for {name}. Download the official model manually and set CHECKPOINT_OVERRIDES.')
            candidate = temporary
        checkpoint = torch.load(candidate, map_location='cpu', weights_only=True)
        validate_checkpoint_metadata(checkpoint, name)
        if candidate != path:
            candidate.replace(path)
        write_json(output / 'checkpoints' / (name + '.json'), dict(
            path=str(path), sha256=sha256(path), bytes=path.stat().st_size,
            epoch=checkpoint['epoch'], iterations=checkpoint['it'],
            source=f"https://drive.google.com/file/d/{spec['drive_id']}/view",
            provenance='Official KITTI validation model-zoo checkpoint, with audited epoch/iteration header. No training-ID manifest is embedded; metadata is supporting evidence, not proof of exact training membership.'))
        print(f'{name}: epoch={checkpoint["epoch"]}, iterations={checkpoint["it"]}, {path.stat().st_size / 1e6:.1f} MB', flush=True)
        del checkpoint


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def make_dataset(repo, root, infos, name):
    import numpy as np
    from easydict import EasyDict
    from pcdet.config import cfg_from_yaml_file
    from pcdet.datasets.kitti.kitti_dataset import KittiDataset
    cfg = EasyDict()
    cfg_from_yaml_file(str(repo / 'tools/cfgs/kitti_models' / MODELS[name]['config']), cfg)
    if tuple(cfg.CLASS_NAMES) != CLASSES:
        raise ValueError('Checkpoint class mapping changed.')
    original = copy.deepcopy(cfg)
    cfg.DATA_CONFIG.INFO_PATH = {'train': [], 'test': []}
    dataset = KittiDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, training=False,
                          root_path=repo / 'data/kitti', logger=None)
    dataset.root_split_path = root
    # No GT in inference samples: avoid target preparation and recall work in timing.
    dataset.kitti_infos = copy.deepcopy(infos)
    for info in dataset.kitti_infos:
        info['image']['image_shape'] = np.asarray(info['image']['image_shape'], dtype=np.int32)
    return cfg, original, dataset


def timed_cuda(function):
    import torch
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    value = function()
    end.record()
    end.synchronize()
    return value, float(start.elapsed_time(end))


def evaluate_predictions(root, prediction_dir, frame_ids):
    import numpy as np
    from pcdet.datasets.kitti.kitti_object_eval_python import eval as reference
    gt = [array_annotation(read_annotation(root / 'label_2' / (identifier + '.txt'))) for identifier in frame_ids]
    predictions = [array_annotation(read_annotation(prediction_dir / (identifier + '.txt'), prediction=True)) for identifier in frame_ids]
    result = {}
    for name, metric in (('bev', 1), ('3d', 2)):
        curves = reference.eval_class(gt, predictions, [0, 1, 2], [0, 1, 2], metric,
            np.tile([.7, .5, .5], (1, 3, 1)), compute_aos=False, num_parts=50)
        result[name] = {}
        for recall, function in (('R11', reference.get_mAP), ('R40', reference.get_mAP_R40)):
            values = function(curves['precision'])[..., 0]
            if not np.isfinite(values).all():
                raise FloatingPointError('Non-finite reference AP.')
            result[name][recall] = dict(
                per_class={c: {d: float(values[i, j]) for j, d in enumerate(DIFFICULTIES)} for i, c in enumerate(CLASSES)},
                map_moderate_percent=float(values[:, 1].mean()), mean_ap_9_percent=float(values.mean()),
                valid_gt_counts={c: {d: sum(reference.clean_data(g, p, i, j)[0] for g, p in zip(gt, predictions))
                                      for j, d in enumerate(DIFFICULTIES)} for i, c in enumerate(CLASSES)})
    return result


def run_model(settings, name, phase):
    import torch
    work, repo, output = paths_for(settings)
    activate_repo(repo)
    configure_gpu()
    print(f'{name}: preparing {phase} inference on the selected GPU', flush=True)
    assets = json.loads((work / 'validation_assets.json').read_text())
    _, splits = audit_splits(repo)
    source_root = normalize_training_root(settings['kitti_root'], settings.get('image_sizes_json'))
    root = normalize_training_root(assets['training_root'], assets.get('image_sizes_json'))
    if assets['source_training_root'] != str(source_root) or assets['splits'] != splits:
        raise ValueError('Validation asset cache changed; rerun the data audit.')
    infos = assets['infos']
    maximum = settings.get('max_frames') if phase == 'full' else settings.get('smoke_frames', 3)
    if maximum is not None:
        infos = infos[:maximum]
    frame_ids = [info['point_cloud']['lidar_idx'] for info in infos]
    warmup = settings['warmup_frames'] if phase == 'full' else min(1, len(infos) - 1)
    if len(infos) <= warmup:
        raise ValueError('Need more frames than warmup_frames.')
    checkpoint = checkpoint_path(settings, name)
    cfg, original_cfg, dataset = make_dataset(repo, root, infos, name)
    environment = runtime_info()
    identity = dict(name=name, phase=phase, frame_ids=frame_ids, warmup_frames=warmup,
                    checkpoint_sha256=sha256(checkpoint), native_config=plain(original_cfg),
                    runner_sha256=sha256(__file__), revision=REVISION, asset_signature=assets['asset_signature'],
                    environment=environment['packages'], gpu=environment['gpu'], capability=environment['capability'],
                    driver_version=environment['driver_version'], torch_cuda=environment['torch_cuda'])
    run_fingerprint = fingerprint(identity)
    destination = output / phase / name
    report_path = destination / 'report.json'
    if report_path.exists():
        existing = json.loads(report_path.read_text())
        if existing.get('status') == 'ok' and existing.get('fingerprint') == run_fingerprint:
            predictions_path = destination / 'kitti_predictions'
            recorded = existing.get('prediction_sha256', {})
            if set(recorded) != set(frame_ids) or any(
                not (predictions_path / (identifier + '.txt')).is_file()
                or sha256(predictions_path / (identifier + '.txt')) != recorded[identifier]
                for identifier in frame_ids
            ):
                raise ValueError('Cached prediction files changed or are missing; choose a new RUN_NAME.')
            print(f'{name}: reusing completed identical run {report_path}', flush=True)
            return
        if existing.get('fingerprint') not in (None, run_fingerprint):
            raise ValueError('Existing run has different settings. Choose a new RUN_NAME.')
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'config.resolved.json', plain(original_cfg))
    write_json(destination / 'identity.json', identity)
    from pcdet.models import build_network, load_data_to_gpu
    model = build_network(cfg.MODEL, len(CLASSES), dataset)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validate_checkpoint_metadata(state, name)
    model_state, updated = model._load_state_dict(state['model_state'], strict=False)
    missing = [key for key in model_state if key not in updated and not key.endswith('num_batches_tracked')]
    if missing:
        raise ValueError(f'Incomplete checkpoint loading: {missing[:20]}')
    model = model.cuda().float().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    del state, model_state, updated
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = {stage: [] for stage in STAGES}
    prediction_dir = destination / 'kitti_predictions'
    prediction_dir.mkdir(exist_ok=True)
    per_frame_path = destination / 'latency_per_frame.csv'
    fields = ['frame_id', 'warmup'] + [stage + '_ms' for stage in STAGES]
    with per_frame_path.open('w', newline='') as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, identifier in enumerate(frame_ids):
            torch.cuda.synchronize()
            total_start = time.perf_counter()
            sample = dataset[index]
            batch = dataset.collate_batch([sample])
            if 'gt_boxes' in batch:
                raise RuntimeError('Ground truth must not enter timed inference.')
            preprocess_ms = (time.perf_counter() - total_start) * 1000
            _, transfer_ms = timed_cuda(lambda: load_data_to_gpu(batch))
            def forward_network():
                value = batch
                for module in model.module_list:
                    value = module(value)
                return value
            batch, network_ms = timed_cuda(forward_network)
            decode_start = time.perf_counter()
            predictions, _ = model.post_processing(batch)
            torch.cuda.synchronize()
            decode_ms = (time.perf_counter() - decode_start) * 1000
            total_ms = (time.perf_counter() - total_start) * 1000
            measured = dict(zip(STAGES, (preprocess_ms, transfer_ms, network_ms, decode_ms, total_ms)))
            writer.writerow(dict(frame_id=identifier, warmup=index < warmup,
                                 **{stage + '_ms': value for stage, value in measured.items()}))
            if index >= warmup:
                for stage, value in measured.items():
                    timings[stage].append(value)
            # KITTI calibration/projection and text writing are outside latency.
            dataset.generate_prediction_dicts(batch, predictions, list(CLASSES), output_path=prediction_dir)
            del predictions, batch, sample
            if (index + 1) % 100 == 0 or index + 1 == len(infos):
                print(f'{name} [{phase}]: {index + 1}/{len(infos)} frames', flush=True)
    peak_memory = torch.cuda.max_memory_allocated() / 1048576
    del model
    torch.cuda.empty_cache()
    accuracy = evaluate_predictions(root, prediction_dir, frame_ids)
    complete = phase == 'full' and len(frame_ids) == 3769 and frame_ids == assets['frame_ids']
    report = dict(name=name, status='ok', phase=phase, benchmark_complete=complete,
        fingerprint=run_fingerprint, frames=len(frame_ids), timed_frames=len(frame_ids) - warmup,
        model=dict(parameters=parameters, checkpoint=str(checkpoint), checkpoint_sha256=identity['checkpoint_sha256'],
                   checkpoint_epoch=MODELS[name]['epoch'], checkpoint_iterations=MODELS[name]['it']),
        protocol=dict(evaluator=verify_evaluator(repo), classes=CLASSES, difficulties=DIFFICULTIES,
            iou_thresholds=dict(zip(CLASSES, (.7, .5, .5))), domain='full-frame original GT; metric-specific pinned ignore rules',
            macro='unweighted three-class Moderate mean; missing classes retained', split=splits['val'],
            native_preprocessing=True, point_cloud_range=plain(original_cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            fov_points_only=original_cfg.DATA_CONFIG.FOV_POINTS_ONLY,
            postprocessing=plain(original_cfg.MODEL.POST_PROCESSING), warmup_frames_excluded=warmup,
            latency_boundary='Sequential raw bin/calibration read + native FOV/voxelization + collation + H2D + module_list + final post_processing; excludes GT recall, annotation projection/text export and AP.',
            network_boundary='module_list, including architecture-specific internal box decoding and ROI proposal NMS; final post_processing is separate.',
            timing_method='H2D/network: CUDA events with end synchronization; preprocessing/final NMS/total: perf_counter with explicit CUDA synchronization.',
            fps_definition='1000 / mean_ms at batch 1; sequential offline latency, not a pipelined throughput claim',
            ap_source='Exported KITTI text predictions (upstream 4-decimal formatting); no conversion from README R11'),
        accuracy=accuracy, latency={stage: timing_summary(values) for stage, values in timings.items()},
        prediction_sha256={identifier: sha256(prediction_dir / (identifier + '.txt')) for identifier in frame_ids},
        runtime={**environment, 'torch_peak_memory_mb': peak_memory},
        outputs=dict(predictions=str(prediction_dir), per_frame_latency=str(per_frame_path)))
    write_json(report_path, report)
    print(json.dumps(summary_row(report), ensure_ascii=False, indent=2), flush=True)


def summarize(settings):
    output = paths_for(settings)[2]
    rows = []
    for name in settings['models']:
        path = output / 'full' / name / 'report.json'
        fallback = (output / 'smoke' / name / 'failure.json', output / 'download' / name / 'failure.json')
        if not path.exists():
            path = next((candidate for candidate in fallback if candidate.exists()), path)
        report = json.loads(path.read_text()) if path.exists() else dict(name=name, status='not_run', phase='full')
        rows.append(summary_row(report))
    destination = output / 'comparison.csv'
    fields = list(rows[0])
    with destination.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'comparison.json', rows)
    columns = ('model', 'status', 'benchmark_complete', 'parameters', 'ap3d_r40_moderate',
               'apbev_r40_moderate', 'network_mean_ms', 'pipeline_mean_ms', 'pipeline_fps')
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join('—' if row.get(key) is None else str(round(row[key], 3))
                      if isinstance(row[key], float) else str(row[key]) for key in columns) + ' |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    print(f'Results: {output}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('setup', 'gpu-check', 'prepare', 'download', 'run', 'summarize'))
    parser.add_argument('--settings', required=True, type=Path)
    parser.add_argument('--model', choices=tuple(MODELS))
    parser.add_argument('--phase', choices=('smoke', 'full'), default='full')
    args = parser.parse_args(argv)
    settings = load_settings(args.settings)
    if args.command == 'run' and args.model is None:
        parser.error('--model is required for run')
    try:
        if args.command == 'run':
            run_model(settings, args.model, args.phase)
        elif args.command == 'download' and args.model:
            download_checkpoints({**settings, 'models': [args.model]})
        else:
            {'setup': setup, 'gpu-check': gpu_check, 'prepare': prepare_data,
             'download': download_checkpoints, 'summarize': summarize}[args.command](settings)
    except Exception as exc:
        if args.command == 'run' or (args.command == 'download' and args.model):
            phase = args.phase if args.command == 'run' else 'download'
            destination = paths_for(settings)[2] / phase / args.model
            error_path = destination / 'failure.json'
            failure = dict(name=args.model, status='error', phase=phase, benchmark_complete=False,
                           error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
            write_json(error_path, failure)
            # A stale successful report must never hide a failed rerun.
            if args.command == 'run':
                write_json(destination / 'report.json', failure)
        raise


if __name__ == '__main__':
    main()
