#!/usr/bin/env python3
"""Resolve raw KITTI or component RAR/ZIP/TAR archives into a local training root."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from zipfile import ZipFile
import zlib

from kitti_training_pipeline.image_metadata import load_image_sizes

COMPONENTS = {'velodyne': '.bin', 'calib': '.txt', 'label_2': '.txt'}
ALIASES = {'velodyne': ('velodyne',), 'calib': ('calib', 'calibration'),
           'label_2': ('label_2', 'labels', 'label'), 'image_2': ('image_2', 'images', 'image')}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def component_for(name, archive_name=''):
    path = PurePosixPath(name.replace('\\', '/'))
    parts = [part.lower() for part in path.parts]
    if any(part in ('testing', 'test', '__macosx', 'image_0', 'image_1', 'image_3',
                    'image_00', 'image_01', 'image_02', 'image_03') for part in parts):
        return None
    if not re.fullmatch(r'[0-9]{6}', path.stem):
        return None
    if '..' in parts or path.is_absolute() or re.match(r'^[A-Za-z]:', str(path)):
        raise ValueError(f'Unsafe archive path: {name}')
    if path.suffix.lower() == '.bin':
        return 'velodyne'
    choices = [key for key, aliases in ALIASES.items()
               if any(alias in parts[:-1] for alias in aliases)]
    if not choices:
        choices = [key for key, aliases in ALIASES.items()
                   if any(alias in archive_name.lower() for alias in aliases)]
    choices = [key for key in choices if path.suffix.lower() == ('.png' if key == 'image_2' else COMPONENTS[key])]
    if len(choices) > 1:
        raise ValueError(f'Ambiguous component: {name} in {archive_name}')
    return choices[0] if choices else None


def is_archive(path):
    name = path.name.lower()
    part = re.search(r'\.part(\d+)\.rar$', name)
    if part and int(part[1]) > 1:
        return False
    return name.endswith(('.rar', '.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tar.xz'))


def walk_files(root, max_depth=5):
    root = Path(root)
    for parent, directories, filenames in os.walk(root):
        depth = len(Path(parent).relative_to(root).parts)
        directories[:] = [name for name in sorted(directories)
                          if name.lower() not in ('testing', 'test', '__macosx', '.git')
                          and not name.startswith('.') and depth < max_depth]
        for name in sorted(filenames):
            yield Path(parent) / name


def rar_support(destination, allow_install):
    support = Path(destination).parent / '.kitti_rar_support'
    if support.is_dir():
        sys.path.insert(0, str(support))
    try:
        module = importlib.import_module('rarfile')
    except ImportError:
        if not allow_install:
            raise RuntimeError('RAR input requires rarfile and unrar. Install rarfile and unrar, or use Colab automatic setup.')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--target', str(support),
                        'rarfile>=4.2,<5'], check=True)
        sys.path.insert(0, str(support))
        module = importlib.import_module('rarfile')
    executable = shutil.which('unrar')
    if executable is None and allow_install and hasattr(os, 'geteuid') and os.geteuid() == 0 and shutil.which('apt-get'):
        print('Cài unrar để đọc RAR trên Colab...', flush=True)
        subprocess.run(['apt-get', 'update', '-qq'], check=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'unrar'], check=True)
        executable = shutil.which('unrar')
    if executable is None:
        raise RuntimeError('Missing unrar. Install the unrar command (e.g. apt-get install unrar), then rerun the data cell.')
    return module, executable


def crc32(path):
    value = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value = zlib.crc32(block, value)
    return value & 0xffffffff


def valid_file(path, component):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or (component != 'label_2' and path.stat().st_size == 0):
        return False
    return component != 'velodyne' or path.stat().st_size % 16 == 0


def prepare_dataset(source, destination, frame_ids, *, metadata_path=None,
                    fallback_metadata=None, allow_install=False):
    """Select exact IDs, exclude hidden test, flatten components and reuse safely.

    Image dimensions may come from the same per-frame JSON as the Robust notebook.
    RAR extraction runs once per archive (including solid RAR), not once per frame.
    """
    source, destination = Path(source).expanduser().resolve(), Path(destination).expanduser().resolve()
    identifiers = list(frame_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers) or any(not re.fullmatch(r'[0-9]{6}', i) for i in identifiers):
        raise ValueError('Expected a nonempty list of unique six-digit KITTI frame IDs')
    wanted = set(identifiers)
    if not source.exists():
        raise FileNotFoundError(f'KITTI source does not exist: {source}. Set KITTI_ROOT to a RAR/ZIP/TAR file or its containing folder.')
    if source.is_dir() and (source == destination or source in destination.parents):
        # A complete source may be reused, but never recursively scan an extraction tree inside an archive source.
        if not all((source / name).is_dir() for name in COMPONENTS):
            raise ValueError('EXTRACT_DIR must be outside the archive source directory')
    files = list(walk_files(source)) if source.is_dir() else [source]
    archives = [path for path in files if is_archive(path)]
    print('KITTI source:', source, flush=True)
    print('Archives:', ', '.join(path.name for path in archives) or '(unpacked files)', flush=True)
    metadata = Path(metadata_path).expanduser().resolve() if metadata_path else None
    if metadata is None:
        candidates = [path for path in files if path.name == 'kitti_image_sizes.json']
        if len(candidates) > 1:
            raise ValueError('Multiple image metadata files found; set IMAGE_SIZES_JSON explicitly')
        metadata = candidates[0] if candidates else None
    # Explicit/provided metadata is authoritative. The bundled metadata is only fallback.
    if metadata is None and fallback_metadata:
        metadata = Path(fallback_metadata).expanduser().resolve()

    records = {}
    def add(component, identifier, record):
        if identifier not in wanted:
            return
        key = component + '/' + identifier
        if key in records:
            existing = records[key]
            if existing['kind'] == 'raw' and record['kind'] != 'raw':
                return
            raise ValueError(f'Duplicate KITTI input for {key}: {existing["source"]} and {record["source"]}')
        records[key] = record

    # Prefer existing unpacked components over their matching archives.
    for path in files:
        if is_archive(path):
            continue
        component = component_for(str(path.relative_to(source)) if source.is_dir() else path.name)
        if component:
            add(component, path.stem, dict(kind='raw', source=str(path), size=path.stat().st_size, member=None, crc=None))
    # Only parse archive members if a required file is not available unpacked.
    required_components = dict(COMPONENTS)
    if metadata is None:
        required_components['image_2'] = '.png'
    needs_archives = any(component + '/' + identifier not in records
                         for component in required_components for identifier in identifiers)
    unrar = None
    if needs_archives:
        for path in archives:
            if path.suffix.lower() == '.zip':
                with ZipFile(path) as archive:
                    for info in archive.infolist():
                        if info.is_dir():
                            continue
                        component = component_for(info.filename, path.name)
                        if component in required_components:
                            add(component, PurePosixPath(info.filename.replace('\\', '/')).stem,
                                dict(kind='zip', source=str(path), member=info.filename, size=info.file_size, crc=info.CRC))
            elif path.suffix.lower() == '.rar':
                module, unrar = rar_support(destination, allow_install)
                try:
                    with module.RarFile(path, errors='strict') as archive:
                        if archive.needs_password():
                            raise ValueError(f'Password-protected RAR is not supported: {path}')
                        for info in archive.infolist():
                            if info.is_dir():
                                continue
                            component = component_for(info.filename, path.name)
                            if component in required_components:
                                if info.is_symlink() or getattr(info, 'file_redir', None):
                                    raise ValueError(f'Archive links are not supported: {info.filename}')
                                add(component, PurePosixPath(info.filename.replace('\\', '/')).stem,
                                    dict(kind='rar', source=str(path), member=info.filename, size=info.file_size, crc=info.CRC))
                except module.Error as error:
                    raise RuntimeError(f'Cannot read RAR {path}; keep all multipart volumes together and include part1. {error}') from error
            else:
                with tarfile.open(path, 'r:*') as archive:
                    for info in archive:
                        if info.isdir():
                            continue
                        component = component_for(info.name, path.name)
                        if component in required_components:
                            if not info.isfile():
                                raise ValueError(f'Archive links/special files are not supported: {info.name}')
                            add(component, PurePosixPath(info.name).stem,
                                dict(kind='tar', source=str(path), member=info.name, size=info.size, crc=None))
    missing = {component: [i for i in identifiers if component + '/' + i not in records] for component in required_components}
    missing = {component: ids for component, ids in missing.items() if ids}
    if missing:
        details = '; '.join(f'{name}: missing {len(ids)} IDs, e.g. {ids[:5]}' for name, ids in missing.items())
        raise FileNotFoundError(f'Incomplete KITTI input ({details}). For images, supply image_2 or kitti_image_sizes.json. '
                                f'Searched {source}; found archives: {[p.name for p in archives]}')
    # Validate metadata coverage before extracting GBs of LiDAR.
    if metadata is not None:
        load_image_sizes(source if source.is_dir() else source.parent, identifiers, metadata_path=metadata)

    selected = {key: value for key, value in records.items() if key.split('/')[0] in required_components}
    raw_parents = {Path(record['source']).parent.parent for record in selected.values() if record['kind'] == 'raw'}
    direct = (all(record['kind'] == 'raw' and Path(record['source']).parent.name == key.split('/')[0]
                  for key, record in selected.items()) and len(raw_parents) == 1)
    root = next(iter(raw_parents)) if direct else destination / 'training'
    if direct:
        for key, record in selected.items():
            if not valid_file(record['source'], key.split('/')[0]):
                raise ValueError('Invalid raw KITTI file: ' + record['source'])
        load_image_sizes(root, identifiers, metadata_path=metadata)
        return dict(training_root=str(root), image_sizes_json=str(metadata) if metadata else None,
                    staged=False, counts={name: len(identifiers) for name in COMPONENTS}, reused_files=len(selected))

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / 'archive_preparation.json'
    old = json.loads(manifest_path.read_text()).get('files', {}) if manifest_path.is_file() else {}
    pending, evidence, reused = [], {}, 0
    for key, record in selected.items():
        component, identifier = key.split('/')
        suffix = '.png' if component == 'image_2' else COMPONENTS[component]
        target = root / component / (identifier + suffix)
        source_stat = Path(record['source']).stat()
        identity = dict(**record, source_size=source_stat.st_size, source_mtime_ns=source_stat.st_mtime_ns)
        cached = old.get(key, {})
        matches = valid_file(target, component) and target.stat().st_size == record['size']
        if matches and cached.get('identity') == identity and cached.get('mtime_ns') == target.stat().st_mtime_ns:
            reused += 1
        elif matches and record['crc'] is not None and crc32(target) == record['crc']:
            reused += 1
        else:
            pending.append((key, record, target))
        evidence[key] = dict(identity=identity)
    required = sum(record['size'] for _, record, _ in pending)
    if pending and shutil.disk_usage(destination).free < required + 512 * 1024 * 1024:
        raise RuntimeError(f'Insufficient local disk space: need {required / 1e9:.2f} GB + 0.5 GB reserve. '
                           'Change EXTRACT_DIR or free local disk; archives stay on Drive.')
    print(f'Local KITTI: {root}; extract/copy {len(pending)} files, reuse {reused}; selected size {required / 1e9:.2f} GB', flush=True)

    def finish(key, record, target, temporary):
        if not valid_file(temporary, key.split('/')[0]) or temporary.stat().st_size != record['size']:
            raise ValueError('Invalid extracted KITTI file: ' + key)
        if record['crc'] is not None and crc32(temporary) != record['crc']:
            raise ValueError('Archive CRC mismatch: ' + key)
        temporary.replace(target)

    groups = {}
    for item in pending:
        groups.setdefault((item[1]['kind'], item[1]['source']), []).append(item)
    copied_raw = 0
    for (kind, original), items in groups.items():
        if kind != 'raw':
            print(f'Preparing {Path(original).name}: {len(items)} files', flush=True)
        if kind == 'rar':
            # Batch decompression is essential for solid/multipart RAR performance.
            with tempfile.TemporaryDirectory(prefix='.kitti_rar_', dir=destination) as temporary_dir:
                staging = Path(temporary_dir)
                members = staging / 'members.txt'
                members.write_text('\n'.join(record['member'] for _, record, _ in items) + '\n')
                subprocess.run([unrar, 'x', '-idq', '-o+', '-p-', original, '@' + str(members), str(staging) + os.sep], check=True)
                for key, record, target in items:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    extracted = staging / record['member'].replace('\\', '/')
                    if staging not in extracted.resolve().parents:
                        raise ValueError('Unsafe extracted archive path')
                    finish(key, record, target, extracted)
        else:
            archive = ZipFile(original) if kind == 'zip' else (tarfile.open(original, 'r:*') if kind == 'tar' else None)
            try:
                for key, record, target in items:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.extracting')
                    if kind == 'raw':
                        shutil.copy2(original, temporary)
                        copied_raw += 1
                        if copied_raw % 500 == 0:
                            print(f'Copied {copied_raw} raw files to local SSD', flush=True)
                    else:
                        reader = archive.open(record['member']) if kind == 'zip' else archive.extractfile(record['member'])
                        with reader as stream, temporary.open('wb') as writer:
                            shutil.copyfileobj(stream, writer, length=1024 * 1024)
                    finish(key, record, target, temporary)
            finally:
                if archive is not None:
                    archive.close()
    for key, record in evidence.items():
        component, identifier = key.split('/')
        target = root / component / (identifier + ('.png' if component == 'image_2' else COMPONENTS[component]))
        record['mtime_ns'] = target.stat().st_mtime_ns
    # Also checks PNG headers when JSON is absent; never fabricate image sizes.
    load_image_sizes(root, identifiers, metadata_path=metadata)
    result = dict(training_root=str(root), image_sizes_json=str(metadata) if metadata else None,
                  staged=True, counts={name: len(identifiers) for name in COMPONENTS}, reused_files=reused,
                  source=str(source), selected_ids=identifiers, files=evidence)
    save_json(manifest_path, result)
    print('KITTI ready:', result['counts'], '| image dimensions:', metadata or root / 'image_2', flush=True)
    return {key: value for key, value in result.items() if key != 'files'}
