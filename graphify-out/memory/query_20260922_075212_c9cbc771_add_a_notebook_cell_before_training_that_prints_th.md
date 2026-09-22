---
type: "query"
date: "2026-09-22T07:52:12.659349+00:00"
question: "Add a notebook cell before training that prints the model architecture and differences compared with the baseline, without needing to print the model name."
contributor: "graphify"
outcome: "useful"
source_nodes: ["test_standard_training_notebook.py", "build_model()", "CustomModel", "MobilePixorBackBone"]
---

# Q: Add a notebook cell before training that prints the model architecture and differences compared with the baseline, without needing to print the model name.

## Answer

Expanded via graph vocab: notebook, model, config, backbone, baseline, training, build, architecture, mobilepixor. Added a CPU-only architecture inspection cell immediately after config selection and before KITTI preparation/smoke/full training. It builds the selected and B0 models through the real build_model path, prints the selected nn.Module tree, compares module types rather than random weights, reports changed/added/removed module roots, reports baseline and selected trainable parameters plus delta, and deletes both temporary models. Added a contract test that checks placement, required comparison logic, cleanup, and Python compilation. Verified locally with B1_C2PSA: Identity -> C2PSA and +36,048 parameters.

## Outcome

- Signal: useful

## Source Nodes

- test_standard_training_notebook.py
- build_model()
- CustomModel
- MobilePixorBackBone