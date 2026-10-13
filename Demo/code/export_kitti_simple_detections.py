#!/usr/bin/env python3
"""Export bare BEV/3D detections and one isolated zoom PNG per object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    CLASS_COLORS, EDGES, camera_basis, canvas_limits, points_in_object, projected,
    roi_vertices, setup_matplotlib, sha256,
)
from tools.visualization.export_kitti_height_detections import (
    crop_indices, lift_all, point_owners, verified_cache,
)


def render_simple_panel(points, indices, owners, objects, cuboids, output, *,
                        view, roi, height_range, zoom=False):
    plt = setup_matplotlib()
    from matplotlib.colors import Normalize, to_rgba
    from matplotlib.collections import LineCollection
    from matplotlib.transforms import Bbox

    if zoom:
        obj = objects[0]
        basis = camera_basis(14., -150. + np.rad2deg(obj.row[6]))
        limits = canvas_limits(cuboids[obj.index].corners, basis, 1., margin=.15)
        size = (1000, 1000)
    else:
        basis = camera_basis(28., -145.)
        limits = canvas_limits(roi_vertices(roi), basis, 1.5, margin=.03)
        size = (1600, 1500) if view == 'bev' else (1800, 1200)
    dpi = 200
    fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi, facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='black')
    cloud = points[indices]
    uvz = projected(cloud, basis)
    colors = plt.get_cmap('turbo')(Normalize(*height_range, clip=True)(cloud[:, 2]))
    annotations = []
    for obj in objects:
        colors[owners[indices] == obj.index] = to_rgba(CLASS_COLORS[obj.name])
    if view == 'bev':
        order = np.argsort(cloud[:, 2], kind='stable')
        xy = cloud[:, [1, 0]]
        ax.set(xlim=(roi[3], roi[2]), ylim=(roi[0], roi[1]))
    else:
        order = np.argsort(uvz[:, 2], kind='stable')
        xy = uvz[:, :2]
        ax.set(xlim=limits[:2], ylim=limits[2:])
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=2.2 if zoom else .7,
               linewidths=0, alpha=1., rasterized=True)
    for obj in objects:
        color = CLASS_COLORS[obj.name]
        if view == 'bev':
            footprint = np.r_[obj.footprint, obj.footprint[:1]][:, [1, 0]]
            ax.plot(footprint[:, 0], footprint[:, 1], color=color, lw=1.)
            anchor = footprint[np.argmax(footprint[:, 1])]
        elif obj.index in cuboids:
            corners = projected(cuboids[obj.index].corners, basis)
            ax.add_collection(LineCollection([corners[[a, b], :2] for a, b in EDGES],
                                             colors=color, linewidths=1.4 if zoom else 1.))
            anchor = corners[4:, :2].mean(axis=0)
            if zoom:
                anchor = corners[np.argmax(corners[:, 1]), :2]
        else:
            continue
        annotations.append(ax.annotate(f'{obj.score:.2f}', anchor, xytext=(0, 5),
                            textcoords='offset points', color='white',
                            fontsize=10 if zoom else 7, ha='center', va='bottom'))
    ax.set_aspect('equal')
    ax.axis('off')
    if not zoom:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        occupied = []
        for text in annotations:
            candidates = [(0, 5)] + [(dx, dy) for dy in range(13, 70, 8) for dx in (0, -16, 16, -32, 32)]
            for offset in candidates:
                text.set_position(offset)
                box = text.get_window_extent(renderer)
                padded = Bbox.from_extents(box.x0-2, box.y0-2, box.x1+2, box.y1+2)
                if (ax.bbox.contains(padded.x0,padded.y0) and ax.bbox.contains(padded.x1,padded.y1)
                        and not any(padded.overlaps(previous) for previous in occupied)):
                    occupied.append(padded)
                    break
    fig.savefig(output, dpi=dpi, facecolor='black', pad_inches=0)
    audit = {'text_labels':[text.get_text() for text in ax.texts],
             'axes_count':len(fig.axes),'axes_hidden':not ax.axison,
             'legend_present':ax.get_legend() is not None, 'figure_text_count':len(fig.texts),
             'object_colors':{str(obj.index):CLASS_COLORS[obj.name] for obj in objects}}
    plt.close(fig)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', required=True, type=Path)
    parser.add_argument('--height-metadata', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--metadata-output', type=Path)
    args = parser.parse_args(argv)
    source, points, rows, objects, provenance = verified_cache(args.prediction_dir.expanduser().resolve())
    frame = source['frame_id']
    output = (args.output_dir or ROOT.parent/'Demo/outputs/detection_simple'/frame).resolve()
    metadata_path = (args.metadata_output or ROOT.parent/'Demo/data/simple_detections'/f'{frame}.json').resolve()
    height = json.loads(args.height_metadata.read_text())
    roi = source['roi_xyz']
    if height['source_sha256'] != provenance['source_bin']['sha256'] or height['roi_xyz'] != roi:
        raise ValueError('Height scale metadata must match this raw scan and ROI')
    height_range = height['shared_height_range_m']
    cuboids, reports = lift_all(points, objects, (roi[4], roi[5]))
    owners = point_owners(len(points), objects, reports)
    visible = [obj for obj in objects if ((obj.footprint.max(axis=0) >= np.array(roi)[[0, 2]])
               & (obj.footprint.min(axis=0) <= np.array(roi)[[1, 3]])).all()]
    output.mkdir(parents=True, exist_ok=True)
    zoom_dir = output/'zooms'
    zoom_dir.mkdir(exist_ok=True)
    audits = {}
    indices = crop_indices(points, roi)
    for view in ('bev', '3d'):
        name = f'detection_{view}.png'
        audits[name] = render_simple_panel(points, indices, owners, visible, cuboids, output/name,
                                           view=view,roi=roi,height_range=height_range)
    zooms = []
    for obj in visible:
        if obj.index not in cuboids:
            continue
        cuboid = cuboids[obj.index]
        indices = np.flatnonzero(points_in_object(points, cuboid))
        name = f'zooms/{obj.index:02d}_{obj.name.lower()}.png'
        audits[name] = render_simple_panel(points, indices, owners, [obj], cuboids, output/name,
                                           view='3d',roi=roi,height_range=height_range,zoom=True)
        zooms.append({'prediction_index':obj.index,'class':obj.name,'score':obj.score,
                      'color_hex':CLASS_COLORS[obj.name],'path':name,
                      'source_point_indices':indices.tolist(),'point_count':len(indices)})
    for value in provenance.values():
        if sha256(Path(value['path'])) != value['sha256']:
            raise RuntimeError('Upstream input changed during rendering')
    metadata = {'frame_id':frame,'output_dir':str(output),'provenance':provenance,
                'inference':source['inference'],'roi_xyz':roi,'shared_height_range_m':height_range,
                'detection_count':len(objects),'visible_detection_count':len(visible),
                'supported_cuboid_count':len(cuboids),'zooms':zooms,'render_audits':audits,
                'bev_rows':rows.tolist(),'vertical_source':'BEV boxes + measured-LiDAR Z/height',
                'ground_truth_used':False,'class_colors':dict(CLASS_COLORS),
                'image_style':'black background; class colors; score only; one 3D object per zoom'}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print(f'Exported BEV, 3D and {len(zooms)} separate object zooms to {output}')
    return metadata


if __name__ == '__main__':
    main()
