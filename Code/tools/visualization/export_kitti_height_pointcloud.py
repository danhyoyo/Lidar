#!/usr/bin/env python3
"""Render measured KITTI points in BEV and 3D with a shared height color scale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    camera_basis, canvas_limits, dataset_paths, projected, read_points,
    setup_matplotlib, sha256,
)


def render_height_views(points, output, *, frame, roi, height_range=None):
    """Use real Z values; percentile clipping affects only the display colors."""
    plt = setup_matplotlib()
    from matplotlib.colors import Normalize
    keep = np.isfinite(points[:, :3]).all(axis=1)
    for axis in range(3):
        keep &= (points[:, axis] >= roi[axis * 2]) & (points[:, axis] <= roi[axis * 2 + 1])
    cloud = points[keep]
    if not len(cloud):
        raise ValueError('No finite points inside the display ROI')
    low, high = height_range if height_range is not None else np.percentile(cloud[:, 2], [1, 99])
    if not np.isfinite([low, high]).all() or low >= high:
        raise ValueError('Height color range must be finite and increasing')
    norm = Normalize(vmin=low, vmax=high, clip=True)
    output.mkdir(parents=True, exist_ok=True)
    basis = camera_basis(28., -145.)
    uvz = projected(cloud, basis)
    limits = canvas_limits(cloud, basis, aspect=1.5, margin=.02)
    layouts = {}
    for view in ('bev', '3d'):
        size = (1600, round(1600 * (roi[1]-roi[0]) / (roi[3]-roi[2]))) if view == 'bev' else (1800, 1200)
        dpi = 200
        fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi, facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
        if view == 'bev':
            # Higher returns remain visible where top-down points overlap.
            order = np.argsort(cloud[:, 2], kind='stable')
            horizontal, vertical = cloud[:, 1], cloud[:, 0]
            ax.set(xlim=(roi[3], roi[2]), ylim=(roi[0], roi[1]))
        else:
            # Orthographic projection with actual XYZ, ordered back-to-front.
            order = np.argsort(uvz[:, 2], kind='stable')
            horizontal, vertical = uvz[:, 0], uvz[:, 1]
            ax.set(xlim=limits[:2], ylim=limits[2:])
        ax.scatter(horizontal[order], vertical[order], c=cloud[order, 2],
                   cmap='turbo', norm=norm, s=.9 if view == 'bev' else 1.1,
                   alpha=1., linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.axis('off')
        layouts[view] = {'size_pixels':list(size), 'axes_count':len(fig.axes),
                         'visible_text_count':len(ax.texts)+len(fig.texts),
                         'title':ax.get_title(),'axes_hidden':not ax.axison,
                         'legend_present':ax.get_legend() is not None}
        for suffix in ('png', 'svg'):
            fig.savefig(output / f'pointcloud_height_{view}.{suffix}', dpi=dpi,
                        facecolor='black', pad_inches=0)
        plt.close(fig)
    return {
        'frame': frame, 'source_point_count': len(points), 'display_point_count': len(cloud),
        'roi_xyz': list(roi), 'color_by': 'measured LiDAR Z (metres)', 'colormap': 'turbo',
        'shared_height_range_m': [float(low), float(high)],
        'height_range_policy': 'explicit range' if height_range is not None else 'display ROI Z percentiles 1 and 99',
        'display_clipping_only': True, 'point_subsampling': False,
        'bev_overlap_order': 'low Z before high Z', '3d_camera_elevation_azimuth_deg': [28, -145],
        'image_style':'full canvas, black background, pointcloud only',
        'view_layouts':layouts, '3d_projected_limits':list(limits),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', default='005663')
    parser.add_argument('--kitti-root', type=Path, default=ROOT / 'Kitti_Raw')
    parser.add_argument('--bin', type=Path, help='Optional direct path to a real XYZI .bin')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--roi', nargs=6, type=float, default=[-2, 45, -24, 24, -2.8, 4])
    parser.add_argument('--height-range', nargs=2, type=float, help='Shared min/max Z color limits, in metres')
    args = parser.parse_args(argv)
    if not np.isfinite(args.roi).all() or any(args.roi[i] >= args.roi[i + 1] for i in (0, 2, 4)):
        parser.error('ROI must contain finite increasing XYZ bounds')
    if len(args.frame) != 6 or not args.frame.isascii() or not args.frame.isdigit():
        parser.error('Frame must be a six-digit KITTI ID')
    source = (args.bin or dataset_paths(args.kitti_root).velodyne / f'{args.frame}.bin').resolve()
    output = (args.output_dir or ROOT.parent / 'Demo/outputs/pointcloud_height' / args.frame).resolve()
    source_hash = sha256(source)
    report = render_height_views(read_points(source), output, frame=args.frame, roi=args.roi,
                                 height_range=args.height_range)
    if sha256(source) != source_hash:
        raise RuntimeError('Source pointcloud changed during rendering')
    report.update(source_bin=str(source), source_sha256=source_hash)
    (output / 'metadata.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
