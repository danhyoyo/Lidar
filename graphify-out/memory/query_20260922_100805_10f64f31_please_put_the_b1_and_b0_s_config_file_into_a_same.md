---
type: "query"
date: "2026-09-22T10:08:05.769647+00:00"
question: "please put the B1 and B0's config file into a same folder please, and edit the notebook accordingly, also enable data augmentation in these configs"
contributor: "graphify"
outcome: "useful"
source_nodes: ["test_standard_training_notebook.py", "Dataset", "MobilePixorBackBone"]
---

# Q: please put the B1 and B0's config file into a same folder please, and edit the notebook accordingly, also enable data augmentation in these configs

## Answer

Expanded from original query via graph vocabulary: [config, notebook, augmentation, baseline, kitti, mobilepixor, training]. Moved B0 and B1_C2PSA configs into configs/kitti/b0_b1, updated every notebook/README/test reference, enabled identical OneOf augmentation with overall p=0.5 and rotation/scaling/translation switches enabled, and set the notebook's safe default to B0. Verified notebook contracts, clean-backbone tests, JSON parity, and model construction from both moved paths.

## Outcome

- Signal: useful

## Source Nodes

- test_standard_training_notebook.py
- Dataset
- MobilePixorBackBone