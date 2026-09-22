---
type: "query"
date: "2026-09-22T15:56:48.528030+00:00"
question: "Does the current codebase have runnable RichBEV-8 or RichBEV-11 support, and how can current Legacy35 B0/B1 configs use RichBEV-8?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["encode_bev()", "voxelize()", "input_shape()", "build_model()", "Dataset", "MobilePixorBackBone", "RichBEV-8", "test_mobile_bev.py"]
---

# Q: Does the current codebase have runnable RichBEV-8 or RichBEV-11 support, and how can current Legacy35 B0/B1 configs use RichBEV-8?

## Answer

Expanded from the user's question via graph vocabulary: [rich, bev, legacy, encode, encoder, channels, voxel, voxelize, config, mobilepixor]. RichBEV-8 is implemented and full MobileBEV tests pass. data.bev_encoding.name=rich8 produces eight channels; common.input_shape propagates 8 into CustomModel and the MobilePIXOR stem. Synthetic B0 and B1_C2PSA forward passes both succeeded. RichBEV-11 is not implemented; only binary_slices and rich8 are accepted. Existing mobilebev configs also enable thesis-derived CoordAtt/UWAG, so current clean experiments should copy only the Rich8 data field into separate factorial-ablation configs rather than reuse those configs wholesale. Existing 35-channel checkpoints are incompatible with the new 8-channel stem. The current repo also has stale README/test references to b0_b1 while configs and notebook point to backbone_branch.

## Outcome

- Signal: useful

## Source Nodes

- encode_bev()
- voxelize()
- input_shape()
- build_model()
- Dataset
- MobilePixorBackBone
- RichBEV-8
- test_mobile_bev.py