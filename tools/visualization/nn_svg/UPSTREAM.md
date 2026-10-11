# NN-SVG upstream snapshot

Source: https://github.com/alexlenail/NN-SVG

Pinned commit: `d72cf1248e15a4c48b3715cd53535153285be306` (MIT, Alexander Lenail).
Runtime HTML/JavaScript and local font assets are included. Paper and example
figures are omitted. Existing styles still use their original CDN dependencies.

Local changes:
- A Backbone tab in index.html, LeNet.html and AlexNet.html.
- Backbone.html / Backbone.js / backbone-ui.js / backbone.css: an offline SVG
  graph editor for MobilePixorNeXt, including branched FPN and operation views.
- Dimension scaling in Backbone.js adapts the depthFn / widthFn functions from
  AlexNet.js. The new renderer uses native SVG to keep labels as vector text and
  support branches, residuals and operators without Three.js or D3 dependencies.
- Editor.js / Reference.js: transactional editable graph documents, preset layout
  data, generic reference-style SVG, free block/edge/group customization, undo/redo
  and portable JSON. Reference.js renders arbitrary IDs and topology.
- Paper.js: separate native 2D renderer with backbone/FPN, block and gate
  panels for a two-column paper figure; the original 3D renderer stays available.
- export-svg.cjs and mobilepixornext.js / .json: local generation and preset.

Original Git blob hashes (before the navigation additions):

```json
{
  "AlexNet.html": "bb186adc61a263477d8b86b19923f5ee048db5c8",
  "AlexNet.js": "bd0355ad2dc773c37401e1acdc34489865e8eb5f",
  "FCNN.js": "6bd2c75ea0dd4ecc82659c90ebe2cf8605d59e83",
  "LICENSE": "dbee165fd6aa4cdf115e7a03e248abb00868f285",
  "LeNet.html": "080894b8388f3ea8f4f616e9f27563932e3238a4",
  "LeNet.js": "66ae49d007ed2ce9fa4b23d59a31c990cede8faf",
  "OrbitControls.js": "72b915d91972e693615025c13b445386872ee29c",
  "Projector.js": "ebb915ecf6955cae003bdfe48a60f9924e9131ee",
  "SVGRenderer.js": "05e4aa542f61e0882f4dc9a50ec26ef36152f6e3",
  "about.html": "1dfb23f9721812c089218bbd300ddcc3c1bce77b",
  "fonts/LICENSE": "65c6407257f1ecd3179ad9005643a24e10db13a5",
  "fonts/README": "8708f607706b13c2df480d08310906f533c59a31",
  "fonts/helvetiker_regular.typeface.json": "d19293b47e8dacf779740ed961f130e8049d1b16",
  "index.html": "5fb47c12138efc73de62fec048dd33f17f74c424",
  "util.js": "ad1f6af3eb9fcfa2183e7e60d1e0406768f5ffa0"
}
```

