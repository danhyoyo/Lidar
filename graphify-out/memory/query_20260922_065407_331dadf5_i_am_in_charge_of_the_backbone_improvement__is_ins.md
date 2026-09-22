---
type: "query"
date: "2026-09-22T06:54:07.817792+00:00"
question: "I am in charge of the Backbone improvement. Is inserting YOLOv26 C2PSA after C5 doable and advisable on mathematical pros and cons, and what else must change in the prior backbone?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["MobilePixorBackBone", "InvertedResidual", "C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)", "Head"]
---

# Q: I am in charge of the Backbone improvement. Is inserting YOLOv26 C2PSA after C5 doable and advisable on mathematical pros and cons, and what else must change in the prior backbone?

## Answer

Expanded via graph vocab: mobilepixor, backbone, attention, inverted, residual, fusion, stride, channels, head, fpn, feature, block. Insertion is shape-compatible at C5 if C2PSA maps 96 channels to 96 channels and preserves stride 16. Baseline path is block5 -> latlayer1 -> top-down FPN, so the isolated variant should be block5 -> C2PSA -> latlayer1 without changing FPN, head, loss, or BEV encoding. At configured 800x704 BEV, C5 is 50x44, N=2200; official C2PSA with e=0.5 uses hidden width 48 and one head. Global attention is quadratic and forms 4.84M scores per sample, approximately 348M attention MACs plus 77M convolution MACs for one PSABlock. It offers global context and CSP/residual gradient paths but is much more expensive than CoordAtt and may dilute small-object/local geometry. Implement under a distinct mobilepixor_c2psa backbone/config, update model dispatch and supported-backbone validation, add shape/gradient/export/profile tests, and keep B0 untouched. Treat this as a cited ablation; because C5 attention placement was inspired by the senior thesis, it is not strictly thesis-independent.

## Outcome

- Signal: useful

## Source Nodes

- MobilePixorBackBone
- InvertedResidual
- C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)
- Head