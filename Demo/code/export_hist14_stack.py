#!/usr/bin/env python3
"""Draw all fourteen Hist14 planes in the supplied representation.png layout."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'Code'), str(ROOT / 'Code/detector')]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.utils_1.bev_backend import encode_hist14_numpy
from tools.visualization.export_kitti_height_detections import verified_cache
from tools.visualization.export_kitti_overview import setup_matplotlib


def main():
    plt = setup_matplotlib()
    from matplotlib.collections import PolyCollection
    from matplotlib.patches import Polygon

    source_dir = ROOT / 'Result/artifacts/kitti_overview_model'
    source, points, _, _, provenance = verified_cache(source_dir)
    config = json.loads((source_dir / 'config.inference.json').read_text())
    schema = resolve_bev_encoding(dict(config['data']['bev_encoding'], backend='numpy'),
                                  config['data']['kitti']['geometry'])
    tensor = encode_hist14_numpy(points, schema)[200:600, :450]
    occupied = tensor[..., 11] > 0
    # Reference layout: upright parallel sheets, vertical side edges, slanted
    # top/bottom edges, and horizontal channel separation (no perspective taper).
    quad = np.array([[0., 0.], [240., 150.], [240., 750.], [0., 600.]])
    step = np.array([62., 0.])
    fig = plt.figure(figsize=(12, 9), dpi=300, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], facecolor='white')
    ax.set_aspect('equal')
    ax.axis('off')
    records = []
    for index in range(14):
        shift = index * step
        plane = quad + shift
        # Opaque planes keep the semantics of separate feature channels.
        ax.add_patch(Polygon(plane, closed=True, facecolor='white',
                             edgecolor='none', zorder=1 + 3 * index))
        values = tensor[..., index]
        active = occupied if index >= 12 else occupied & (values != 0)
        iy, ix = np.nonzero(active)
        cmap = 'Oranges' if index == 0 else 'Blues'
        normalized = (values[active] + .5 if index >= 12 else
                      values[active] / (.5 if index == 7 else 1.))
        colors = plt.get_cmap(cmap)(np.clip(normalized, 0, 1))
        # Every cell is placed by the same affine transform as its parent plane. No
        # interpolation, generated texture, or changes to the feature values.
        # Forward range runs across each narrow sheet, matching the reference's
        # sideways map orientation. All channels share this same orientation.
        left, right = ix / 450, (ix + 1) / 450
        near, far = iy / 400, (iy + 1) / 400
        vertices = np.stack((np.c_[left, near], np.c_[right, near],
                             np.c_[right, far], np.c_[left, far]), axis=1)
        projected = (quad[0] + vertices[..., :1] * (quad[1] - quad[0])
                     + vertices[..., 1:] * (quad[3] - quad[0]) + shift)
        ax.add_collection(PolyCollection(projected, facecolors=colors, edgecolors='none',
                                         antialiased=False, rasterized=True,
                                         zorder=2 + 3 * index))
        # Thin outlines distinguish the fourteen channel planes on white.
        ax.add_patch(Polygon(plane, closed=True, fill=False, edgecolor='#333333',
                             linewidth=.65, zorder=3 + 3 * index))
        records.append({'tensor_index': index, 'name': schema.channel_names[index],
                        'projected_quad': plane.tolist(), 'rendered_nonzero_cells': len(iy),
                        'colormap': cmap})
    all_corners = np.concatenate([quad + index * step for index in range(14)])
    low, high = all_corners.min(axis=0), all_corners.max(axis=0)
    center = (low + high) / 2
    span = high - low
    aspect = 12 / 9
    width = max(span[0] * 1.08, span[1] * 1.08 * aspect)
    height = width / aspect
    ax.set(xlim=(center[0] - width/2, center[0] + width/2),
           ylim=(center[1] - height/2, center[1] + height/2))
    assert not ax.axison and not ax.texts and not fig.texts
    output = ROOT / 'Demo/outputs/hist14_diagram/005663'
    output.mkdir(parents=True, exist_ok=True)
    destination = output / '22_hist14_channel_stack.png'
    fig.savefig(destination, dpi=300, facecolor='white', pad_inches=0)
    audit_dir = ROOT / 'artifacts/hist14_diagram_005663'
    audit_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(audit_dir / 'hist14_channel_stack.svg', facecolor='white', pad_inches=0)
    plt.close(fig)
    report = {'frame': source['frame_id'], 'source': provenance,
              'channel_count': 14, 'source_shape_yxc': list(tensor.shape),
              'left_to_right_channel_indices': list(range(14)),
              'front_to_back_channel_indices': list(range(13, -1, -1)),
              'projection': 'upright parallel planes, affine shear, horizontal channel spacing',
              'map_orientation': 'forward X across sheet width; lateral Y up sheet height',
              'layout_reference': str(output / 'representation.png'),
              'layer_step': step.tolist(), 'planes': records,
              'background': 'white', 'plane_background': 'white', 'text_count': 0, 'axes_visible': False,
              'canvas_pixels': [3600, 2700], 'output': str(destination)}
    (audit_dir / 'channel_stack_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    with zipfile.ZipFile(output.parent / '005663_images.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.glob('*.png')):
            if path.name != 'representation.png':
                archive.write(path, path.name)
    print(json.dumps({'output': str(destination), 'channels': 14, 'size': [3600, 2700]}, indent=2))


if __name__ == '__main__':
    main()
