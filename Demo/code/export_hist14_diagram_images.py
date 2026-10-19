#!/usr/bin/env python3
"""Render plain black-background images from Hist14 and verified BEV detections."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'Code'), str(ROOT / 'Code/detector')]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.utils_1.bev_backend import encode_hist14_numpy
from tools.visualization.export_kitti_overview import (
    CLASS_COLORS, camera_basis, canvas_limits, points_in_object,
    projected, setup_matplotlib, sha256,
)
from tools.visualization.export_kitti_height_detections import lift_all, verified_cache


def main():
    plt = setup_matplotlib()
    from matplotlib.colors import Normalize

    prediction_dir = ROOT / 'Result/artifacts/kitti_overview_model'
    source, points, rows, objects, provenance = verified_cache(prediction_dir)
    assert source['frame_id'] == '005663'
    checkpoint = Path(source['inference']['checkpoint'])
    assert sha256(checkpoint) == source['inference']['checkpoint_sha256']
    output = ROOT / 'Demo/outputs/hist14_diagram/005663'
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((prediction_dir / 'config.inference.json').read_text())
    geometry = config['data']['kitti']['geometry']
    encoding = dict(config['data']['bev_encoding'], backend='numpy')
    schema = resolve_bev_encoding(encoding, geometry)
    full = encode_hist14_numpy(points, schema)
    # Same exact XY region for all 14 channels and the BEV scene.
    tensor = full[200:600, :450]
    roi = (0., 45., -20., 20., -2.8, 4.)
    keep = ((points[:, :3] >= np.array(roi)[::2])
            & (points[:, :3] <= np.array(roi)[1::2])).all(axis=1)
    cloud = points[keep]
    height_range = (-2.5, 1.)
    height_norm = Normalize(*height_range, clip=True)
    audits = {}

    def canvas(size):
        dpi = 200
        fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor='black')
        ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
        ax.set_aspect('equal')
        ax.axis('off')
        return fig, ax

    def finish(fig, ax, name):
        assert not ax.axison and not ax.texts and not fig.texts
        assert not ax.get_title() and not ax.get_xlabel() and not ax.get_ylabel()
        assert len(fig.axes) == 1 and ax.get_legend() is None
        fig.savefig(output / name, dpi=200, facecolor='black', pad_inches=0)
        audits[name] = {'axes_visible': False, 'text_count': 0, 'black_background': True}
        plt.close(fig)

    # 00: actual measured XYZ; no detection overlays.
    basis = camera_basis(28., -145.)
    uvz = projected(cloud, basis)
    order = np.argsort(uvz[:, 2], kind='stable')
    fig, ax = canvas((2000, 1500))
    limits = canvas_limits(cloud, basis, 4 / 3, margin=.025)
    ax.set(xlim=limits[:2], ylim=limits[2:])
    ax.scatter(uvz[order, 0], uvz[order, 1], c=cloud[order, 2], cmap='turbo',
               norm=height_norm, s=.8, linewidths=0)
    finish(fig, ax, '00_pointcloud_height.png')

    # 01–14: render feature values directly, without axes or interpolation.
    # A feature value of zero maps to black; offsets retain signed neutral zero
    # in occupied cells. Empty pillars always map to black.
    occupied = tensor[..., 11] > 0
    for index, name in enumerate(schema.channel_names):
        values = tensor[..., index]
        if index >= 12:
            colors = plt.get_cmap('coolwarm')(np.clip(values + .5, 0, 1))[..., :3]
            colors[~occupied] = 0
        elif index == 11:
            colors = np.zeros((*values.shape, 3), dtype=float)
            colors[occupied] = [0.15, .85, 1.]
        else:
            high = .5 if index == 7 else 1.
            colors = plt.get_cmap('turbo')(np.clip(values / high, 0, 1))[..., :3]
            colors[(values == 0) | ~occupied] = 0
        image = np.flip(colors.transpose(1, 0, 2), axis=(0, 1))
        image = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)
        filename = f'{index + 1:02d}_{name}.png'
        plt.imsave(output / filename, image)
        audits[filename] = {'axes_visible': False, 'text_count': 0, 'black_background': True,
                            'tensor_index': index, 'zero_feature_is_black': index < 12}

    # 15: only checkpoint-predicted oriented BEV boxes; no ground-truth input.
    visible = [obj for obj in objects if 0 < obj.row[2] < 45 and abs(obj.row[3]) < 20]
    fig, ax = canvas((1600, 1800))
    ax.set(xlim=(20, -20), ylim=(0, 45))
    order = np.argsort(cloud[:, 2], kind='stable')
    ax.scatter(cloud[order, 1], cloud[order, 0], c=cloud[order, 2], cmap='turbo',
               norm=height_norm, s=.8, linewidths=0)
    for obj in visible:
        footprint = np.r_[obj.footprint, obj.footprint[:1]]
        ax.plot(footprint[:, 1], footprint[:, 0], color=CLASS_COLORS[obj.name], lw=1.35)
    finish(fig, ax, '15_detection_bev.png')

    # Measured support is used only to isolate the object points for zooms.
    # Every displayed detection is a flat BEV footprint.
    cuboids, reports = lift_all(points, visible, roi[4:])
    # Detection scene shares the exact camera, crop, colors and canvas of 00.
    basis = camera_basis(28., -145.)
    uvz = projected(cloud, basis)
    order = np.argsort(uvz[:, 2], kind='stable')
    limits = canvas_limits(cloud, basis, 4 / 3, margin=.025)
    fig, ax = canvas((2000, 1500))
    ax.set(xlim=limits[:2], ylim=limits[2:])
    ax.scatter(uvz[order, 0], uvz[order, 1], c=cloud[order, 2], cmap='turbo',
               norm=height_norm, s=.8, linewidths=0)
    footprints = []
    display_rectangles = {}
    for obj in visible:
        # BEV predictions contain XY, length, width and yaw only. Place one
        # flat rectangle on measured local ground purely for perspective display.
        ground_plane = reports[obj.index].get('ground', {}).get('ground_plane')
        display_z = (float(np.r_[obj.row[2:4], 1.] @ np.asarray(ground_plane))
                     if ground_plane is not None else geometry['z_min'])
        rectangle = np.c_[obj.footprint, np.full(4, display_z)]
        display_rectangles[obj.index] = rectangle
        corners = projected(rectangle, basis)
        outline = corners[[0, 1, 2, 3, 0], :2]
        ax.plot(outline[:, 0], outline[:, 1], color=CLASS_COLORS[obj.name], lw=1.6)
        footprints.append({'prediction_index': obj.index, 'vertices': 4, 'edges': 4,
                           'display_z_m': display_z, 'predicted_height_used': False})
    finish(fig, ax, '15_detection_pointcloud.png')
    matched_camera = {'elevation_azimuth_degrees': [28., -145.],
                      'projected_limits': list(limits), 'canvas_pixels': [2000, 1500]}
    for filename in ('00_pointcloud_height.png', '15_detection_pointcloud.png'):
        audits[filename]['camera'] = matched_camera.copy()
    audits['15_detection_pointcloud.png'].update(
        bev_footprint_count=len(footprints), footprints=footprints, cuboid_count=0,
        box_mode='BEV rectangles in the pointcloud camera',
        vertical_source='flat local-ground display placement only; no box height')
    zooms = []
    for name in ('Car', 'Pedestrian', 'Cyclist'):
        candidates = []
        for obj in visible:
            if obj.name != name or obj.index not in cuboids:
                continue
            indices = np.flatnonzero(points_in_object(points, cuboids[obj.index]))
            if len(indices) >= 8:
                candidates.append((len(indices) * obj.score, obj, indices))
        chosen = sorted(candidates, key=lambda item: item[0], reverse=True)[:2]
        assert len(chosen) == 2, f'Need two supported {name} detections'
        for _, obj, indices in chosen:
            obj_cloud = points[indices]
            basis = camera_basis(28., -145.)
            uvz = projected(obj_cloud, basis)
            order = np.argsort(uvz[:, 2], kind='stable')
            rectangle = display_rectangles[obj.index]
            corners = projected(rectangle, basis)
            limits = canvas_limits(np.r_[obj_cloud[:, :3], rectangle], basis, 1., margin=.13)
            fig, ax = canvas((1200, 1200))
            ax.set(xlim=limits[:2], ylim=limits[2:])
            ax.scatter(uvz[order, 0], uvz[order, 1], c=obj_cloud[order, 2],
                       cmap='turbo', norm=height_norm, s=5., linewidths=0)
            outline = corners[[0, 1, 2, 3, 0], :2]
            ax.plot(outline[:, 0], outline[:, 1], color=CLASS_COLORS[obj.name], lw=1.6)
            filename = f'{16 + len(zooms):02d}_zoom_{name.lower()}_{obj.index:02d}.png'
            finish(fig, ax, filename)
            audits[filename].update(
                bev_footprint_count=1, cuboid_count=0, box_vertices=4, box_edges=4,
                box_mode='BEV rectangle in the pointcloud camera',
                elevation_azimuth_degrees=[28., -145.])
            zooms.append({'file': filename, 'prediction_index': obj.index, 'class': obj.name,
                          'score': obj.score, 'source_point_count': len(indices),
                          'box_mode': 'BEV', 'cuboid_count': 0,
                          'vertical_source': 'flat local-ground display placement only; no box height'})

    # Provenance stays outside the image-only delivery folder.
    audit = {'frame': source['frame_id'], 'output': str(output), 'provenance': provenance,
             'checkpoint_sha256': sha256(checkpoint), 'ground_truth_used': False,
             'height_range_m': height_range, 'roi_xyz_m': roi,
             'channel_order': list(schema.channel_names), 'channel_zero_display': 'black',
             'visible_detections': dict(Counter(obj.name for obj in visible)),
             'zooms': zooms, 'images': audits}
    audit_dir = ROOT / 'artifacts/hist14_diagram_005663'
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / 'render_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    assert sha256(provenance['source_bin']['path']) == provenance['source_bin']['sha256']
    print(json.dumps({'output': str(output), 'images': len(audits),
                      'detections': len(visible), 'zooms': len(zooms)}, indent=2))


if __name__ == '__main__':
    main()
