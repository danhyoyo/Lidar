# MobilePixorNeXt paper diagrams

The [loss and attention gallery](loss_attention/index.html) contains OGA,
Q-OGA, Focal Context, SimAM, Gaussian focal loss and adaptive task weighting.
See the [six-page vector PDF](loss_attention/loss_attention_atlas.pdf) or
[source audit and figure links](loss_attention/README.md).

The [current vs proposed grouped-head comparison](head_current_vs_proposed/README.md)
shows both groups with every Conv3×3 and output Conv1×1 drawn explicitly.
The proposed architecture shares the first Conv3×3 within each group and is
implemented via `model.header_architecture="shared"`; `independent` remains the
default. Separate and side-by-side PNG/SVG/PDF/TikZ figures are included.

The single BEV → backbone → **two explicit grouped heads** figure with SimAM
at C3 is in [`hist14_focal_simam_f32_s42/`](hist14_focal_simam_f32_s42/README.md).
This is a requested variant derived from the original artifact's
`local_attention=none` configuration, with its own config and forward audit.

The artifact-specific atlas for **hist14 + Focal Context, local attention none,
SG-FPN fusion 32, seed 42** is in
[`hist14_focal_none_f32_s42/`](hist14_focal_none_f32_s42/README.md).
Its [gallery](hist14_focal_none_f32_s42/index.html) contains 17 separate figures,
including the backbone, stages, NeXt blocks, Focal Context, both scale gates and
independent grouped heads, with PNG/SVG/PDF/PlotNeuralNet TikZ exports.

The figures follow `../other_paper.png` and `../other_paper_backbone.png`:
pastel operation colors, serif labels, dashed panels, a downward encoder and
upward decoder. Insets show the actual NeXt block and scale-gated fusion.

| Figure | Configuration | Backbone including neck | Output (B,C,H,W) |
| --- | --- | ---: | --- |
| `mobilepixornext_rich8` | `configs/config.json` | 616,080 | (1,16,200,176) |
| `mobilepixornext_hist14_focal_eca` | `configs/experiments/under1m/hist14_local3_focal_eca_grouped_oga_iqa.json` | 660,532 | (1,32,200,176) |

Each figure has a 300-DPI PNG (6480×4680), editable SVG, vector PDF,
PlotNeuralNet/TikZ `.tex`, and an architecture audit `.json`.

The generator constructs the production backbone and captures its actual
full-resolution CPU forward shapes. It validates every displayed feature level,
finite output, and text bounds/overlaps before exporting. The JSON records the
config hash, stage depths, feature shapes and actual parameter count. These
figures do not claim measured AP or latency.

The detection head panel is a branch template. In the grouped preset it is
repeated independently for Car (K=1) and Pedestrian/Cyclist (K=2), including
independent regression and IoU branches. Head parameters are excluded from the
backbone parameter number. C3/C4/C5 are semantic FPN levels at strides 4/8/16;
the Python temporary variable `c2` corresponds to the diagram's C3.

## Regenerate

Run from the repository root with PyTorch and Matplotlib installed:

```bash
python3 tools/visualization/draw_mobilepixornext_paper.py
```

To draw a supported custom configuration:

```bash
python3 tools/visualization/draw_mobilepixornext_paper.py \
  --config configs/config.json \
  --output-dir diagrams/mobilepixornext \
  --dpi 300
```

The layout supports the standard SG-FPN, optional Focal Context/ECA, Gaussian
BEV heads with BN/SiLU, and excludes RepConv, RC necks, detail paths and SimAM.
PNG, SVG and PDF are rendered directly by Matplotlib from the shared scene.
No LaTeX installation is needed for those three formats.

## PlotNeuralNet / Overleaf source

The `.tex` export calls the vendored
`third_party/PlotNeuralNet/PlotNeuralNet/PyCore/TikzGen.py::ToConv` for cuboids.
The `layers/` directory contains the required original layer resources and MIT
license. Upload a `.tex` file and that directory together to Overleaf, or run:

```bash
cd diagrams/mobilepixornext
pdflatex -halt-on-error mobilepixornext_hist14_focal_eca.tex
```

**The TikZ source has not been compiled locally:** this machine has no LaTeX
compiler. The delivered PDFs are the verified Matplotlib vector exports.
Compile into a separate output directory if you want to preserve those PDFs.
Cuboid sizes are schematic; tensor labels specify the actual feature dimensions.

The existing reference images, visualization scripts, model files and configs
were preserved.
