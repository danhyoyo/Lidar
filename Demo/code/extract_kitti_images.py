#!/usr/bin/env python3
"""Find kitti_images.zip on Drive and extract exact training/image_2 PNGs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import struct
from zipfile import ZipFile


def find_images_zip(search_roots, *, explicit=None):
    """Prefer direct matches, then search subfolders by the user's exact filename."""
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    roots = list(dict.fromkeys(Path(root).expanduser().resolve() for root in search_roots))
    direct = {root / 'kitti_images.zip' for root in roots if (root / 'kitti_images.zip').is_file()}
    matches = direct or {path.resolve() for root in roots if root.is_dir()
                        for path in root.rglob('kitti_images.zip') if path.is_file()}
    if len(matches) > 1:
        raise ValueError('Multiple kitti_images.zip files found; set DRIVE_IMAGES_ZIP or --archive:\n'
                         + '\n'.join(str(path) for path in sorted(matches)))
    return next(iter(matches)) if matches else None


def image_size(stream, member):
    header = stream.read(24)
    if len(header) != 24 or header[:8] != b'\x89PNG\r\n\x1a\n' or header[12:16] != b'IHDR':
        raise ValueError(f'Invalid PNG header: {member}')
    width, height = struct.unpack('>II', header[16:24])
    if width < 1 or height < 1:
        raise ValueError(f'Invalid PNG dimensions: {member}')
    return [width, height]


def extract_training_images(archive_path, target_folder, *, expected_count=7481):
    """Flatten selected training left images into image_2, without extracting other splits."""
    archive_path, target_folder = Path(archive_path), Path(target_folder)
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError('expected_count must be positive')
    selected, sizes = {}, {}
    with ZipFile(archive_path) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename.replace('\\', '/'))
            parts = [part.lower() for part in path.parts]
            identifier = path.stem
            if (info.is_dir() or path.suffix.lower() != '.png'
                    or 'testing' in parts or '__macosx' in parts
                    or any(camera in parts for camera in ('image_0', 'image_1', 'image_3',
                                                          'image_00', 'image_01', 'image_02', 'image_03'))
                    or len(identifier) != 6 or not identifier.isascii() or not identifier.isdigit()):
                continue
            if '..' in parts or path.is_absolute():
                raise ValueError(f'Unexpected path in image ZIP: {info.filename}')
            if identifier in selected:
                raise ValueError(f'Duplicate training image ID: {identifier}')
            selected[identifier] = info
        if len(selected) < expected_count:
            raise ValueError(f'{archive_path.name} contains {len(selected)} training/image_2 PNGs; '
                             f'expected at least {expected_count}. Check that this is the full training image ZIP.')
        # Read dimensions for every selected image before creating the target directory.
        for identifier, info in selected.items():
            with archive.open(info) as stream:
                sizes[identifier] = image_size(stream, info.filename)
        target_folder.mkdir(parents=True, exist_ok=True)
        for index, (identifier, info) in enumerate(sorted(selected.items()), 1):
            destination = target_folder / f'{identifier}.png'
            partial = destination.with_suffix('.png.partial')
            try:
                with archive.open(info) as stream, partial.open('wb') as output:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
                if partial.stat().st_size != info.file_size:
                    raise ValueError(f'Extracted image size mismatch: {info.filename}')
                partial.replace(destination)
            finally:
                partial.unlink(missing_ok=True)
            if index == 1 or index % 500 == 0 or index == len(selected):
                print(f'Extracted {index}/{len(selected)} images -> {target_folder}', flush=True)
    report = {'source_zip': str(archive_path.resolve()), 'source_zip_size_bytes': archive_path.stat().st_size,
              'target_folder': str(target_folder.resolve()), 'split': 'training', 'camera': 'image_2',
              'image_count': len(selected), 'zip_crc_checked_during_extraction': True,
              'image_sizes': sizes}
    (target_folder.parent / 'image_2_extraction.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Explicit path; otherwise find kitti_images.zip on MyDrive')
    parser.add_argument('--mydrive-root', type=Path, default=Path('/content/drive/MyDrive'))
    parser.add_argument('--output-dir', type=Path, default=Path('/content/kitti_raw/training/image_2'))
    parser.add_argument('--expected-count', type=int, default=7481)
    args = parser.parse_args(argv)
    source = find_images_zip([args.mydrive_root], explicit=args.archive)
    if source is None:
        raise FileNotFoundError(f'Cannot find kitti_images.zip under {args.mydrive_root}; mount Drive or use --archive')
    print(f'Using image archive: {source}', flush=True)
    report = extract_training_images(source, args.output_dir, expected_count=args.expected_count)
    print(f"Ready: {report['image_count']} training images; dimensions saved to image_2_extraction.json")


if __name__ == '__main__':
    main()
