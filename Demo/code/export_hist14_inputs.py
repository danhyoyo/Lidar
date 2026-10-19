#!/usr/bin/env python3
"""Export measured KITTI Hist14 inputs as reusable scientific figure assets.

The encoder runs once on the full configured ROI. Focus figures are slices of
that tensor, never a recomputed encoding. Ground truth only enters references.
"""
from __future__ import annotations

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'Code'), str(ROOT / 'Code/detector')]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.utils_1.bev_backend import encode_hist14_numpy
from tools.visualization.export_kitti_overview import (
    camera_basis, canvas_limits, dataset_paths, points_in_object, projected,
    read_objects, read_points, setup_matplotlib, sha256,
)

TITLES = [
    'Height-bin density 0', 'Height-bin density 1',
    'Height-bin density 2', 'Height-bin density 3',
    'Maximum height', 'Mean height', 'Height span', 'Height standard deviation',
    'Maximum intensity', 'Mean intensity', 'Log point density', 'Occupancy',
    'Mean point offset X', 'Mean point offset Y',
]
COLORS = {'Car': '#dc3c45', 'Pedestrian': '#059669', 'Cyclist': '#c08108'}


def save_figure(fig, destination, *, svg=True):
    fig.savefig(destination.with_suffix('.png'), dpi=240, facecolor='white', bbox_inches='tight')
    if svg:
        fig.savefig(destination.with_suffix('.svg'), facecolor='white', bbox_inches='tight')


def configure_axis(ax, bounds):
    x0, x1, y0, y1 = bounds
    ax.set(xlim=(y1, y0), ylim=(x0, x1), xlabel='Y lateral (m)', ylabel='X forward (m)')
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7)
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    for spine in ax.spines.values():
        spine.set_color('#cbd5e1')


def channel_style(index):
    if index >= 12:
        return 'coolwarm', -.5, .5
    if index == 11:
        return 'Blues', 0., 1.
    return 'viridis', 0., .5 if index == 7 else 1.


def rgba_map(values, occupied, index):
    plt = setup_matplotlib()
    cmap, low, high = channel_style(index)
    rgba = plt.get_cmap(cmap)(np.clip((values - low) / (high - low), 0, 1))
    # A true zero in an occupied cell stays visible, including offset=0.
    rgba[..., 3] = occupied
    # Tensor is [Y,X]; screen is [forward,lateral], with forward up and left +Y.
    return np.flip(np.transpose(rgba, (1, 0, 2)), axis=(0, 1))


def draw_map(ax, tensor, index, bounds):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    x0, x1, y0, y1 = bounds
    ax.imshow(rgba_map(tensor[..., index], tensor[..., 11] > 0, index),
              extent=(y1, y0, x0, x1), origin='upper', interpolation='nearest')
    configure_axis(ax, bounds)
    cmap, low, high = channel_style(index)
    return ScalarMappable(norm=Normalize(low, high), cmap=cmap)


def draw_reference(ax, cloud, objects, bounds, *, ground_truth):
    order = np.argsort(cloud[:, 2], kind='stable')
    ax.scatter(cloud[order, 1], cloud[order, 0], c=cloud[order, 2],
               cmap='turbo', vmin=-2.5, vmax=1., s=.23, linewidths=0, rasterized=True)
    if ground_truth:
        for obj in objects:
            footprint = np.r_[obj.corners[:4], obj.corners[:1]]
            ax.plot(footprint[:, 1], footprint[:, 0], color=COLORS[obj.name], lw=.8)
    configure_axis(ax, bounds)


def write_readme(output, report):
    edges = report['schema']['bin_edges']
    lines = [
        '# Hist14 input visualization — KITTI ' + report['frame'], '',
        'Mở `index.html` để xem toàn bộ; `hist14_overview.png` là bảng tổng hợp.', '',
        '- `channels/`: 14 panel có tên, trục và thang màu (PNG/SVG).',
        '- `transparent/`: 14 ảnh PNG không chữ, nền trong suốt, vùng focus.',
        '- `full_roi/`: 14 ảnh PNG nền trong suốt của toàn bộ input ROI.',
        '- `pointcloud_3d_transparent.png`, `pointcloud_bev_transparent.png`: cloud để ghép diagram.',
        '- `hist14_tensor.npz`: tensor float32 nguyên gốc [14,Y,X], không resize.',
        '- `metadata.json`: nguồn, hash, geometry, thông số hiển thị và nhãn đối chiếu.', '',
        '## Nguồn và quy ước', '',
        f"Scan gốc có {report['source_points']:,} điểm; encoder giữ {report['encoded_points']:,} điểm.",
        f"Geometry đầy đủ: {report['schema']['geometry']}. Vùng focus XY: {report['focus_xy_m']} m.",
        'Mọi kênh đều được tính một lần bằng encoder Hist14 v1 của dự án trên scan gốc, không augmentation.',
        'Nhãn GT chỉ xuất hiện trên ảnh tham chiếu; không đi vào encoder. Đây không phải detection của model.',
        'Hướng nhìn BEV: X tiến lên trên, +Y sang trái. Không nội suy hay smoothing.',
        'Ảnh PNG nền trong suốt lặp mỗi ô 4×4 pixel để ghép diagram; tensor không thay đổi.',
        'SVG giữ chữ/trục dạng vector và nhúng raster cho feature map/point cloud.', '',
        'Đặt h=(z-z_min)/(z_max-z_min), I=clip(raw_intensity×intensity_scale,0,1),',
        'D(n)=min(1,log(1+n)/log(1+32)). N là số điểm trong pillar XY 0.1×0.1 m.', '',
        '| Kênh (tensor index) | Tên | Đại lượng | Thang màu |',
        '|---|---|---|---|',
    ]
    formulas = [f'D(n_b{i}); z ∈ [{edges[i]:g}, {edges[i+1]:g}) m' for i in range(4)] + [
        'max(h)', 'mean(h)', 'max(h)−min(h)', 'std(h), population',
        'max(I)', 'mean(I)', 'D(N)', '1 nếu N>0, còn lại 0',
        'mean((x−x_cell_center)/x_res)', 'mean((y−y_cell_center)/y_res)',
    ]
    for channel, formula in zip(report['channels'], formulas):
        lines.append(f"| C{channel['index']:02d} | {channel['name']} | {formula} | {channel['display_range']} |")
    lines += ['', 'Tất cả ô trống trong tensor đều bằng 0. Khi vẽ, ô trống được đặt alpha=0;',
              'giá trị 0 ở ô có điểm vẫn được tô màu. Height là giá trị chuẩn hóa, không phải mét.',
              'Thang màu cố định theo miền giá trị; không cân bằng histogram hay chuẩn hóa riêng từng ảnh.', '',
              '## Chạy lại', '', '```sh',
              '/home/duyennh/miniconda3/envs/AI_env/bin/python Demo/code/export_hist14_inputs.py --frame ' + report['frame'],
              '```', '']
    (output / 'README.md').write_text('\n'.join(lines))


def write_viewer(output, report):
    cards = []
    for channel in report['channels']:
        name = channel['stem']
        cards.append(f'<article><h3>C{channel["index"]:02d} · {html.escape(channel["name"])}</h3>'
                     f'<a href="channels/{name}.png"><img src="channels/{name}.png"></a>'
                     f'<p><a href="transparent/{name}.png">PNG trong suốt</a> · '
                     f'<a href="channels/{name}.svg">SVG có chú thích</a> · '
                     f'<a href="full_roi/{name}.png">Toàn ROI</a></p></article>')
    document = '''<!doctype html><html lang="vi"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Hist14 input assets</title>
<style>body{font:16px system-ui;margin:30px auto;max-width:1500px;padding:0 20px;color:#17243a;background:#f3f6fa}
h1{font-size:30px}a{color:#225db2}img{width:100%;display:block}article{background:white;padding:12px;border-radius:10px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:20px}h3{font-size:15px}
p{line-height:1.6}button{padding:9px 16px;margin:10px 0;cursor:pointer} .checker{background:#102030}</style>
<h1>Hist14 · KITTI FRAME_ID</h1>
<p>Scan thật · không augmentation · COUNTS. GT chỉ dùng cho hình tham chiếu.<br>
Input đầy đủ: 14 × 800 × 704; focus: X=[0,45], Y=[−20,20] m. X hướng lên trên, +Y sang trái.</p>
<p><a href="README.md">Công thức và quy ước</a> · <a href="metadata.json">Metadata</a> ·
<a href="hist14_tensor.npz">Tensor gốc</a> · <a href="hist14_overview.svg">Overview SVG</a></p>
<a href="hist14_overview.png"><img src="hist14_overview.png"></a>
<h2>Point cloud để ghép diagram</h2>
<button onclick="document.querySelectorAll('.cloud').forEach(e=>e.classList.toggle('checker'))">Đổi nền cloud</button>
<div class="grid"><article class="cloud"><a href="pointcloud_3d_transparent.png"><img src="pointcloud_3d_transparent.png"></a></article>
<article class="cloud"><a href="pointcloud_bev_transparent.png"><img src="pointcloud_bev_transparent.png"></a></article></div>
<h2>Từng kênh input</h2><div class="grid">CARDS</div></html>'''
    document = document.replace('FRAME_ID', report['frame']).replace(
        'COUNTS', html.escape(str(report['focus_ground_truth_counts']))).replace('CARDS', ''.join(cards))
    (output / 'index.html').write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', default='005663')
    parser.add_argument('--kitti-root', type=Path, default=ROOT / 'Code/Kitti_Raw')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    if len(args.frame) != 6 or not args.frame.isascii() or not args.frame.isdigit():
        parser.error('--frame must be a six-digit KITTI ID')
    output = args.output_dir or ROOT / 'Demo/outputs/hist14_inputs' / args.frame
    for folder in ('channels', 'transparent', 'full_roi'):
        (output / folder).mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(args.kitti_root)
    source = paths.velodyne / f'{args.frame}.bin'
    label = paths.labels / f'{args.frame}.txt'
    calib = paths.calib / f'{args.frame}.txt'
    source_hash = sha256(source)
    points = read_points(source)
    config = json.loads((ROOT / 'Code/configs/config.json').read_text())
    geometry = config['data']['kitti']['geometry']
    schema = resolve_bev_encoding({'name': 'hist14', 'backend': 'numpy',
                                   'density_norm': 32, 'intensity_scale': 1}, geometry)
    tensor = encode_hist14_numpy(points, schema)
    assert tensor.shape == (800, 704, 14) and tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    keep = np.isfinite(points[:, :4]).all(axis=1)
    for col, axis in enumerate('xyz'):
        keep &= (points[:, col] > geometry[f'{axis}_min'] + .001)
        keep &= (points[:, col] < geometry[f'{axis}_max'] - .001)
    selected = points[keep]
    focus = (0., 45., -20., 20.)
    cloud = selected[(selected[:, 0] < 45) & (np.abs(selected[:, 1]) < 20)]
    # Slice the existing full-grid tensor at exact cell boundaries.
    ix0, ix1 = [round((value - geometry['x_min']) / geometry['x_res']) for value in focus[:2]]
    iy0, iy1 = [round((value - geometry['y_min']) / geometry['y_res']) for value in focus[2:]]
    cropped = tensor[iy0:iy1, ix0:ix1]
    objects = read_objects(label, calib)
    objects = [obj for obj in objects if 0 < obj.center[0] < 45 and abs(obj.center[1]) < 20]
    counts = dict(Counter(obj.name for obj in objects))
    report = {'frame': args.frame, 'source_bin': str(source.resolve()), 'source_sha256': source_hash,
              'label_sha256': sha256(label), 'calibration_sha256': sha256(calib),
              'schema': schema.semantic_metadata(), 'source_points': len(points),
              'encoded_points': len(selected), 'focus_points': len(cloud), 'focus_xy_m': list(focus),
              'tensor_shape_cyx': [14, *tensor.shape[:2]], 'augmentation': False,
              'focus_ground_truth_counts': counts, 'ground_truth_is_model_input': False,
              'display': {'empty_cells': 'transparent; occupied zeros retained',
                          'bev_orientation': 'X up; +Y left', 'interpolation': 'nearest',
                          'clean_png_cell_upscale': 4, 'percentile_color_scaling': False},
              'objects': [{'class': obj.name, 'label_index': obj.label_index,
                           'points_in_box': int(points_in_object(selected, obj).sum()),
                           'occlusion': obj.occluded, 'truncation': obj.truncated}
                          for obj in objects], 'channels': []}
    plt = setup_matplotlib()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    for index, name in enumerate(schema.channel_names):
        stem = f'c{index:02d}_{name}'
        cmap, low, high = channel_style(index)
        for folder, value in (('transparent', cropped), ('full_roi', tensor)):
            rgba = rgba_map(value[..., index], value[..., 11] > 0, index)
            plt.imsave(output / folder / f'{stem}.png', np.repeat(np.repeat(rgba, 4, axis=0), 4, axis=1))
        fig, ax = plt.subplots(figsize=(4.8, 5.4))
        image = draw_map(ax, cropped, index, focus)
        ax.set_title(f'C{index:02d} · {TITLES[index]}\n{name}', fontsize=10)
        fig.colorbar(image, ax=ax, fraction=.04, pad=.035, label='Normalized feature value')
        save_figure(fig, output / 'channels' / stem)
        plt.close(fig)
        occupied_values = tensor[..., index][tensor[..., 11] > 0]
        report['channels'].append({'index': index, 'name': name, 'stem': stem,
                                   'display_range': [low, high], 'colormap': cmap,
                                   'actual_occupied_min': float(occupied_values.min()),
                                   'actual_occupied_max': float(occupied_values.max())})
    print('Exported 14 channels', flush=True)
    # References are separate assets; no annotations are burned into input maps.
    fig, ax = plt.subplots(figsize=(6, 6.7))
    draw_reference(ax, cloud, objects, focus, ground_truth=True)
    ax.set_title(f'KITTI {args.frame} · ground-truth reference\n'
                 f"{counts.get('Car',0)} Car · {counts.get('Pedestrian',0)} Pedestrian · {counts.get('Cyclist',0)} Cyclist")
    save_figure(fig, output / 'bev_ground_truth')
    plt.close(fig)
    for view in ('bev', '3d'):
        fig = plt.figure(figsize=(8, 9 if view == 'bev' else 6), dpi=240)
        ax = fig.add_axes([0, 0, 1, 1])
        if view == 'bev':
            draw_reference(ax, cloud, objects, focus, ground_truth=False)
        else:
            basis = camera_basis(28., -145.)
            uvz = projected(cloud, basis)
            order = np.argsort(uvz[:, 2], kind='stable')
            ax.scatter(uvz[order, 0], uvz[order, 1], c=cloud[order, 2], cmap='turbo',
                       vmin=-2.5, vmax=1., s=.55, linewidths=0, rasterized=True)
            limits = canvas_limits(cloud, basis, aspect=8/6, margin=.025)
            ax.set(xlim=limits[:2], ylim=limits[2:])
            ax.set_aspect('equal')
        ax.axis('off')
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'pointcloud_{view}_transparent.{suffix}', transparent=True, dpi=240)
        if view == '3d':
            fig.savefig(output / 'pointcloud_3d_dark.png', facecolor='#101b2c', dpi=240)
        plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(16, 18), layout='constrained')
    axes[0, 0].imshow(plt.imread(output / 'pointcloud_3d_transparent.png'))
    axes[0, 0].set_title('(a) Measured point cloud · height color', fontsize=10)
    axes[0, 0].axis('off')
    draw_reference(axes[0, 1], cloud, objects, focus, ground_truth=True)
    axes[0, 1].set_title('(b) Ground truth · 4 Car / 13 Pedestrian / 4 Cyclist' if args.frame == '005663'
                         else '(b) Ground-truth reference', fontsize=9)
    for index, ax in enumerate(list(axes.flat)[2:]):
        image = draw_map(ax, cropped, index, focus)
        ax.set_title(f'C{index:02d} · {TITLES[index]}', fontsize=10)
        fig.colorbar(image, ax=ax, fraction=.038, pad=.025, shrink=.8).ax.tick_params(labelsize=7)
    fig.suptitle(f'Hist14 input representation · KITTI {args.frame}', fontsize=20)
    save_figure(fig, output / 'hist14_overview')
    plt.close(fig)
    np.savez_compressed(output / 'hist14_tensor.npz', hist14_cyx=np.ascontiguousarray(tensor.transpose(2, 0, 1)),
                        channel_names=np.asarray(schema.channel_names))
    if sha256(source) != source_hash:
        raise RuntimeError('Source point cloud changed during export')
    (output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    write_readme(output, report)
    write_viewer(output, report)
    print(json.dumps({'output': str(output), 'source_points': len(points), 'encoded_points': len(selected),
                      'ground_truth_counts': counts, 'tensor_cyx': report['tensor_shape_cyx']}, indent=2))


if __name__ == '__main__':
    main()
