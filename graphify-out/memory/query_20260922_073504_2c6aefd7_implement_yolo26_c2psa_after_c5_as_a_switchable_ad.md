---
type: "query"
date: "2026-09-22T07:35:04.733402+00:00"
question: "Implement YOLO26 C2PSA after C5 as a switchable addition and clean the Google Colab training notebook of unusable thesis-derived variants."
contributor: "graphify"
outcome: "useful"
source_nodes: ["MobilePixorBackBone", "CustomModel", "test_standard_training_notebook.py", "C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)"]
---

# Q: Implement YOLO26 C2PSA after C5 as a switchable addition and clean the Google Colab training notebook of unusable thesis-derived variants.

## Answer

Expanded via graph vocab: mobilepixor, backbone, attention, config, notebook, training, model, augmentation, fpn, feature, block. Implemented model.c5_attention as none or c2psa inside the existing MobilePIXOR backbone, preserving B0 by default and inserting a local dependency-free C2PSA only between block5 and latlayer1. Added configurable repeats, expansion, and attention ratio; a clean B1 JSON differing from B0 only in model attention metadata; lazy-loaded legacy CoordAtt; made augmentation flags explicitly false; updated training names/logs; converted the Colab notebook to B0 and B1_C2PSA only with automatic BF16/FP16 selection; and added clean C2PSA/notebook tests. Verified syntax, gradients, checkpoint compatibility, config factory construction, full repository regressions, actual 96x50x44 C2PSA output, and a complete 35x800x704 forward producing 200x176 heads.

## Outcome

- Signal: useful

## Source Nodes

- MobilePixorBackBone
- CustomModel
- test_standard_training_notebook.py
- C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)