#!/usr/bin/env python3
"""Render verified raw KITTI detections with height background and instance zooms.

Boxes and scores come from the BEV checkpoint cache. Vertical geometry and
foreground support are measured-LiDAR postprocessing, not learned 3D outputs.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    EDGES, camera_basis, canvas_limits, projected, read_points, roi_vertices,
    setup_matplotlib, sha256,
)
from tools.visualization.predict_kitti_overview import bev_detections, points_in_bev
from tools.visualization.lift_kitti_bev_to_3d import local_ground, lift_object


def instance_color(index):
    """Distinct, stable colors keyed by source prediction ID, independent of class."""
    if index < 0 or int(index) != index:
        raise ValueError('Expected a nonnegative integer prediction ID')
    rgb = colorsys.hsv_to_rgb((.04 + index * .618033988749895) % 1., .72, 1.)
    return '#' + ''.join(f'{round(channel * 255):02x}' for channel in rgb)


def verified_cache(directory):
    metadata_path = directory / 'metadata.json'
    source = json.loads(metadata_path.read_text())
    if (source.get('box_source') != 'fresh checkpoint inference'
            or source.get('ground_truth_boxes_used', True)
            or source['inference']['box_mode'] != 'bev'):
        raise ValueError('Expected verified BEV checkpoint inference without GT boxes')
    bin_path = Path(source['source_bin']['path'])
    if sha256(bin_path) != source['source_bin']['sha256']:
        raise ValueError('Raw LiDAR hash does not match cached inference')
    row_path = directory / f"{source['frame_id']}_bev_predictions.npy"
    rows = np.load(row_path, allow_pickle=False)
    original = np.asarray([item['bev_row'] for item in source['predictions']], dtype=rows.dtype).reshape(-1, 7)
    if not np.array_equal(rows, original):
        raise ValueError('Prediction rows do not match inference metadata')
    config_path = directory / 'config.inference.json'
    config = json.loads(config_path.read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != source['inference']['config_sha256']:
        raise ValueError('Config hash does not match inference metadata')
    objects = bev_detections(rows, config['data']['kitti']['objects'])
    provenance = {
        'inference_metadata': {'path': str(metadata_path), 'sha256': sha256(metadata_path)},
        'predictions': {'path': str(row_path), 'sha256': sha256(row_path)},
        'config': {'path': str(config_path), 'sha256': sha256(config_path)},
        'source_bin': {'path': str(bin_path), 'sha256': sha256(bin_path)},
    }
    return source, read_points(bin_path), rows, objects, provenance


def crop_indices(points, roi):
    roi = np.asarray(roi)
    return np.flatnonzero(np.isfinite(points[:, :3]).all(axis=1)
        & ((points[:, :3] >= roi[::2]) & (points[:, :3] <= roi[1::2])).all(axis=1))


def lift_all(points, objects, z_bounds):
    occupied = np.zeros(len(points), dtype=bool)
    for obj in objects:
        occupied |= points_in_bev(points, obj)
    cuboids, reports = {}, {}
    for obj in objects:
        try:
            plane, ground = local_ground(points, obj, occupied, z_bounds=z_bounds)
            cuboid, report = lift_object(points, obj, plane, z_bounds=z_bounds)
            report['ground'] = ground
            if cuboid is not None:
                cuboids[obj.index] = cuboid
        except ValueError as error:
            report = {'prediction_index': obj.index, 'status': 'unsupported', 'reason': str(error)}
        reports[obj.index] = report
    return cuboids, reports


def point_owners(point_count, objects, reports):
    """Assign measured foreground support to the highest-score overlapping box."""
    owner = np.full(point_count, -1, dtype=np.int64)
    for obj in sorted(objects, key=lambda item: (item.score, -item.index)):
        support = np.asarray(reports[obj.index].get('support_indices', []), dtype=np.int64)
        if support.size:
            if support.min() < 0 or support.max() >= point_count:
                raise ValueError('Foreground support index outside source cloud')
            owner[support] = obj.index
    return owner


def place_labels(ax, labels):
    """Place scene labels without covering each other; leaders identify boxes."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(size=7)
    scale = ax.figure.dpi / 72.
    occupied = []
    for text, anchor, color in labels:
        width, height, _ = renderer.get_text_width_height_descent(text, font, ismath=False)
        center = ax.transData.transform(anchor)
        candidates = [(0, 14)]
        for distance in range(28, 225, 14):
            candidates.extend([(0, distance), (0, -distance), (distance, distance/2),
                               (-distance, distance/2), (distance, -distance/2), (-distance, -distance/2)])
        chosen = candidates[-1]
        for offset in candidates:
            x, y = center + np.asarray(offset)*scale
            box = Bbox.from_bounds(x-width/2-4, y-height/2-4, width+8, height+8)
            if (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                    and not any(box.overlaps(previous) for previous in occupied)):
                occupied.append(box); chosen = offset
                break
        ax.annotate(text, anchor, xytext=chosen, textcoords='offset points',
                    ha='center', va='center', color=color, fontsize=7,
                    bbox={'facecolor':'#080c12','edgecolor':color,'alpha':.9,'pad':1.4},
                    arrowprops={'arrowstyle':'-','color':color,'lw':.65,'shrinkA':2,'shrinkB':2})


def render_view(points, source_indices, owners, objects, cuboids, output, *, view,
                roi, height_range, frame, zoom=False):
    plt = setup_matplotlib()
    from matplotlib.colors import Normalize
    from matplotlib.collections import LineCollection

    cloud = points[source_indices]
    basis = camera_basis(28., -145.)
    uvz = projected(cloud, basis)
    fig, ax = plt.subplots(figsize=((8, 8) if zoom else (10, 9)) if view == 'bev'
                           else ((9, 8) if zoom else (13, 8)), facecolor='#080c12')
    ax.set_facecolor('#080c12')
    norm = Normalize(*height_range, clip=True)
    colors = plt.get_cmap('turbo')(norm(cloud[:, 2]))
    labels = []
    for obj in objects:
        colors[owners[source_indices] == obj.index] = plt.matplotlib.colors.to_rgba(instance_color(obj.index))
    if view == 'bev':
        order = np.argsort(cloud[:, 2], kind='stable')
        xy = cloud[:, [1, 0]]
        ax.set(xlim=(roi[3], roi[2]), ylim=(roi[0], roi[1]),
               xlabel='Y lateral (m)', ylabel='X forward (m)')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#75808b')
    else:
        order = np.argsort(uvz[:, 2], kind='stable')
        xy = uvz[:, :2]
        limits = canvas_limits(roi_vertices(roi), basis, aspect=1.12 if zoom else 1.5, margin=.03)
        ax.set(xlim=limits[:2], ylim=limits[2:]); ax.axis('off')
    ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=6 if zoom else 1.1,
               linewidths=0, alpha=1., rasterized=True)
    for obj in objects:
        color = instance_color(obj.index)
        if view == 'bev':
            footprint = np.r_[obj.footprint, obj.footprint[:1]][:, [1, 0]]
            ax.plot(footprint[:, 0], footprint[:, 1], color=color, lw=1.8 if zoom else 1.15)
            anchor = footprint[:-1].mean(axis=0)
        elif obj.index in cuboids:
            corners = projected(cuboids[obj.index].corners, basis)
            ax.add_collection(LineCollection([corners[[a, b], :2] for a, b in EDGES],
                                             colors=color, linewidths=1.8 if zoom else 1.15))
            anchor = corners[4:, :2].mean(axis=0)
        else:
            continue
        text = f'#{obj.index:02d} {obj.name} {obj.score:.2f}'
        if zoom:
            ax.annotate(text, anchor, xytext=(0, 6), textcoords='offset points', ha='center', va='bottom',
                        color=color, fontsize=10,
                        bbox={'facecolor':'#080c12','edgecolor':color,'alpha':.85,'pad':1.4})
        else:
            labels.append((text, anchor, color))
    ax.set_aspect('equal')
    if zoom:
        obj = objects[0]
        title = f'#{obj.index:02d} {obj.name} | {view.upper()} | score {obj.score:.3f}'
        if view == '3d' and obj.index not in cuboids:
            title += '\nInsufficient LiDAR support for Z / height'
    else:
        title = f'KITTI {frame} | {view.upper()} detections | {len(objects)} objects'
    ax.set_title(title, color='white', fontsize=14, pad=14)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='turbo'), ax=ax,
                       fraction=.033, pad=.035, extend='both')
    bar.set_label('Background Z height (m)', color='white')
    bar.ax.tick_params(colors='white'); bar.outline.set_edgecolor('#75808b')
    fig.text(.5, .025, 'Background: height colors  |  Box + foreground points: object ID color',
             color='#c8d2df', ha='center', fontsize=9)
    if view == '3d':
        fig.text(.5, .005, 'BEV checkpoint boxes + LiDAR-estimated Z / height',
                 color='#a7b6c8', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    if labels:
        fig.canvas.draw()
        place_labels(ax, labels)
    for suffix in ('png', 'svg'):
        fig.savefig(output.with_suffix('.' + suffix), dpi=200,
                    facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close(fig)


def joined_pair(left, right, output, *, width=1100):
    from PIL import Image
    result = Image.new('RGB', (width * 2, 1000), '#080c12')
    for x, path in ((0, left), (width, right)):
        with Image.open(path) as im:
            image = im.convert('RGB')
            image.thumbnail((width, 1000), Image.Resampling.LANCZOS)
            result.paste(image, (x + (width-image.width)//2, (1000-image.height)//2))
    result.save(output)


def contact_sheet(entries, output, *, columns=4, thumbnail_width=600, thumbnail_height=300):
    from PIL import Image
    if not entries:
        return
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new('RGB', (columns * thumbnail_width, rows * thumbnail_height), '#080c12')
    for index, path in enumerate(entries):
        with Image.open(path) as image:
            image = image.convert('RGB')
            image.thumbnail((thumbnail_width, thumbnail_height), Image.Resampling.LANCZOS)
            sheet.paste(image, ((index % columns) * thumbnail_width + (thumbnail_width-image.width)//2,
                               (index // columns) * thumbnail_height + (thumbnail_height-image.height)//2))
    sheet.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--height-metadata', type=Path,
                        help='Use the shared height limits and ROI from the corresponding raw render')
    args = parser.parse_args(argv)
    prediction_dir = args.prediction_dir.expanduser().resolve()
    source, points, rows, objects, provenance = verified_cache(prediction_dir)
    frame = source['frame_id']
    output = (args.output_dir or ROOT.parent / 'Demo/outputs/height_detections' / frame).resolve()
    if output == prediction_dir:
        parser.error('Use a separate output directory to preserve cached predictions')
    output.mkdir(parents=True, exist_ok=True)
    roi = source['roi_xyz']
    scene_indices = crop_indices(points, roi)
    if not len(scene_indices):
        raise ValueError('Empty display ROI')
    height_range = np.percentile(points[scene_indices, 2], [1, 99]).tolist()
    if args.height_metadata:
        height_metadata = json.loads(args.height_metadata.read_text())
        if (height_metadata['source_sha256'] != provenance['source_bin']['sha256']
                or height_metadata['roi_xyz'] != roi):
            raise ValueError('Height metadata does not match the raw scan and ROI')
        height_range = height_metadata['shared_height_range_m']
    cuboids, reports = lift_all(points, objects, (roi[4], roi[5]))
    owners = point_owners(len(points), objects, reports)
    # Render predictions whose XY extent intersects the original raw display ROI.
    visible = [obj for obj in objects if ((obj.footprint.max(axis=0) >= np.array(roi)[[0, 2]])
               & (obj.footprint.min(axis=0) <= np.array(roi)[[1, 3]])).all()]
    print(f'{len(objects)} cached raw-frame detections, {len(cuboids)} LiDAR-supported cuboids, {len(visible)} visible', flush=True)
    for view in ('bev', '3d'):
        render_view(points, scene_indices, owners, visible, cuboids, output / f'detection_height_{view}',
                    view=view, roi=roi, height_range=height_range, frame=frame)
    joined_pair(output/'detection_height_bev.png', output/'detection_height_3d.png', output/'detections_overview.png')
    np.save(output / f'{frame}_bev_predictions.npy', rows)
    lifted_ids = np.array(sorted(cuboids), dtype=np.int64)
    np.savez_compressed(output/'lifted_detections_3d.npz',
                        rows=np.asarray([cuboids[index].row for index in lifted_ids], dtype=rows.dtype).reshape(-1, 9),
                        source_prediction_indices=lifted_ids, source_bev_rows=rows)
    zooms, preview_paths = [], []
    for obj in visible:
        padding = max(1., .3 * max(obj.row[4:6]))
        low, high = obj.footprint.min(axis=0)-padding, obj.footprint.max(axis=0)+padding
        if obj.index in cuboids:
            z_low, z_high = cuboids[obj.index].corners[:, 2].min()-.45, cuboids[obj.index].corners[:, 2].max()+.45
        else:
            z_low, z_high = roi[4:6]
        zoom_roi = [float(low[0]), float(high[0]), float(low[1]), float(high[1]), float(z_low), float(z_high)]
        indices = crop_indices(points, zoom_roi)
        folder = output / 'zooms' / f'{obj.index:02d}_{obj.name.lower()}'
        folder.mkdir(parents=True, exist_ok=True)
        for view in ('bev', '3d'):
            render_view(points, indices, owners, [obj], cuboids, folder / view, view=view,
                        roi=zoom_roi, height_range=height_range, frame=frame, zoom=True)
        joined_pair(folder/'bev.png', folder/'3d.png', folder/'pair.png')
        support = np.asarray(reports[obj.index].get('support_indices', []), dtype=np.int64)
        np.savez_compressed(folder/'points.npz', context_points=points[indices], context_source_indices=indices,
                            foreground_points=points[support], foreground_source_indices=support)
        zooms.append({'prediction_index':obj.index,'class':obj.name,'score':obj.score,
                      'color_hex':instance_color(obj.index),'bev_row':obj.row.tolist(),
                      'roi_xyz':zoom_roi,'context_point_count':len(indices),'foreground_point_count':len(support),
                      'cuboid_supported':obj.index in cuboids,'path':str(folder.relative_to(output))})
        preview_paths.append(folder/'pair.png')
        print(f'Zoom #{obj.index:02d} {obj.name}: {len(indices)} measured context points, {len(support)} foreground points', flush=True)
    contact_sheet(preview_paths, output/'object_zooms.png')
    examples = []
    for name in ('Car', 'Cyclist', 'Pedestrian'):
        candidates = [item for item in zooms if item['class']==name and item['cuboid_supported']]
        if candidates:
            best = max(candidates, key=lambda item: (item['foreground_point_count'], item['score']))
            examples.append({'class':name,'prediction_index':best['prediction_index'],'path':best['path']+'/pair.png'})
    contact_sheet([output/item['path'] for item in examples], output/'class_zoom_examples.png',
                  columns=1, thumbnail_width=2200, thumbnail_height=1000)
    metadata = {'frame_id':frame,'box_source':'verified cached raw-frame checkpoint BEV inference',
        'inference':source['inference'],'provenance':provenance,'ground_truth_used':False,
        'source_point_count':len(points),'display_point_count':len(scene_indices),'roi_xyz':roi,
        'shared_height_range_m':height_range,'height_color':'turbo, measured LiDAR Z in metres',
        'instance_color':'golden-angle HSV keyed by source prediction index; not tracking IDs',
        'foreground_assignment':'measured above-ground support; highest score wins overlap',
        'vertical_source':'BEV boxes + measured local ground / foreground Z; not learned 3D',
        'detection_count':len(objects),'visible_detection_count':len(visible),'supported_cuboid_count':len(cuboids),
        'per_class':{name:sum(obj.name==name for obj in objects) for name in ('Car','Cyclist','Pedestrian')},
        'objects':[{'prediction_index':obj.index,'class':obj.name,'score':obj.score,
                    'color_hex':instance_color(obj.index),'bev_row':obj.row.tolist(),
                    'cuboid_row':cuboids[obj.index].row.tolist() if obj.index in cuboids else None}
                   for obj in objects],
        'vertical_reports':list(reports.values()),'zooms':zooms,'class_examples':examples}
    # Confirm all upstream inputs remained unchanged.
    for value in provenance.values():
        if sha256(Path(value['path'])) != value['sha256']:
            raise RuntimeError('Upstream input changed during rendering')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    (output/'README.md').write_text(
        f'# KITTI {frame}: height background + instance detections\n\n'
        'Open `detection_height_bev.png`, `detection_height_3d.png`, `object_zooms.png` '
        'or `class_zoom_examples.png`. Each `zooms/<ID>_<class>/` has BEV/3D PNG/SVG, '
        'a paired preview and actual measured point crops with original scan indices.\n\n'
        'Background color is measured Z; box and foreground support color identifies '
        'the same prediction in every image. IDs are frame-local, not track IDs. '
        'The original raw height scale is shared by the scenes and all zooms.\n\n'
        'BEV geometry and score come from verified checkpoint inference on the raw scan. '
        'Z/height come from existing measured-LiDAR lifting, without GT or class height priors. '
        'These are model predictions, not ground truth or an AP evaluation.\n')
    print(f'Complete: {output}',flush=True)
    return metadata


if __name__ == '__main__':
    main()
