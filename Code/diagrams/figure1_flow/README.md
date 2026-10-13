# Figure 1 — middle flow panel

Compact flow matching the layout, serif font, rounded boxes, dashed frame and
filled orange arrows of `Code/diagrams/overview.png`. The original reference is
preserved. The revised panel has two upper boxes, five aligned prediction heads,
the existing heatmap, and one OGA loss bar; no separate postprocessing blocks.

- `overview_flow.svg`: editable text and vector shapes.
- `overview_flow.pdf`: vector text/shapes plus the existing PNG heatmap.
- `overview_flow_no_heatmap.svg`: vector source with an empty heatmap slot.
- `overview_flow_no_heatmap.pdf`: entirely vector version with an empty slot.
- `overview_flow.png`: white background, 600 DPI.
- `overview_flow_transparent.png`: transparent background, 600 DPI, for assembly.
- `provenance.json`: source metadata, checkpoint hash and preset validation.

Place the point cloud to the left of the input arrow and the BEV detections to
the right of the output arrow. Decoding and NMS are implicit in the output arrow.
Any associated 3D cuboid panel uses subsequent LiDAR-based vertical lifting,
which should be identified in the figure caption. The heatmap is the existing
model prediction from KITTI frame 005663, embedded byte-for-byte without edits.

The two head groups are **Car** and **Pedestrian + Cyclist**, each with the five
illustrated output branches. Offsets are metric residuals in meters. The final
3D stage estimates vertical extent using local ground and measured LiDAR points;
it is a postprocessing step after BEV detection.

Orange arrows show inference flow. The five downward black arrows denote loss
supervision. The OGA objective includes classification focal loss,
offset/size/yaw Smooth L1, oriented geometry and IoU-quality supervision with
adaptive task weighting. Focal Context, SG-FPN and the two head groups are
collapsed into the detector/prediction structure to keep Figure 1 compact.

Regenerate from the repository root:

```bash
python3 Code/tools/visualization/draw_overview_flow.py
```

Requires Python 3 and `rsvg-convert` (librsvg); no model inference is rerun.
