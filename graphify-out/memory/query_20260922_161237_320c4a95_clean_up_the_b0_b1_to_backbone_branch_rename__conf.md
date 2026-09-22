---
type: "query"
date: "2026-09-22T16:12:37.981122+00:00"
question: "Clean up the b0_b1 to backbone_branch rename, confirm binary_slices means Legacy35, and add the missing BEV encoding fields to both configs without changing architecture."
contributor: "graphify"
outcome: "useful"
source_nodes: ["encode_bev()", "input_shape()", "Dataset", "MobilePixorBackBone", "test_standard_training_notebook.py", "test_mobile_bev.py"]
---

# Q: Clean up the b0_b1 to backbone_branch rename, confirm binary_slices means Legacy35, and add the missing BEV encoding fields to both configs without changing architecture.

## Answer

Expanded through graph vocabulary: [config, notebook, legacy, encode, encoder, dataset, test, mobilepixor]. Updated all stale README and notebook-contract references from configs/kitti/b0_b1 to configs/kitti/backbone_branch. Added identical data.bev_encoding objects to B0 and B1 with name=binary_slices, density_norm=32, intensity_scale=1. Did not copy deprecated CoordAtt, UWAG, or scale_gated_fpn fields. Updated the notebook contract to accept either supported selected variant. Verified binary_slices resolves to input shape (1,35,800,704), both stems remain 35 channels, parameter counts remain 597817 and 633865, notebook tests pass 6/6, and MobileBEV regression checks pass 11/11.

## Outcome

- Signal: useful

## Source Nodes

- encode_bev()
- input_shape()
- Dataset
- MobilePixorBackBone
- test_standard_training_notebook.py
- test_mobile_bev.py