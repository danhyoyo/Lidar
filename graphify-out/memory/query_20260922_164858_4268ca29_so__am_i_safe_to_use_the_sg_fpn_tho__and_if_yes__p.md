---
type: "query"
date: "2026-09-22T16:48:58.353045+00:00"
question: "so, am I safe to use the SG-FPN tho? And if yes, please also help me to get the SG-FPN from the depricated code and make it work with a switch from the config file surely"
contributor: "graphify"
outcome: "useful"
source_nodes: ["MobilePixorBackBone", "SG-FPN", "CustomModel", "check_gates()", "test_standard_training_notebook.py"]
---

# Q: so, am I safe to use the SG-FPN tho? And if yes, please also help me to get the SG-FPN from the depricated code and make it work with a switch from the config file surely

## Answer

Expanded from original query via graph vocab: [fpn, gated, gate, fusion, mobilepixor, backbone, feature, scaled]. The clean MobilePixorBackBone now supports model.scale_gated_fpn. False preserves sum-FPN and the existing B0/B1 parameter counts; true adds zero-initialized depthwise gates at C4 and C3, starts exactly equivalent to sum-FPN, adds 480 parameters, receives finite nonzero gradients, and is routed by CustomModel. Both backbone_branch configs keep it false. The Colab notebook supports a copied true config through CONFIG_OVERRIDE and reports the structural/parameter delta. The local senior thesis contains the ordinary FPN plus CoordAtt/UWAG work but no SG-FPN or scale-gated FPN phrase; Git history places this gate in a later MobileBEV commit. It may be used as cited inherited repository work, not presented as a new contribution. The acronym SG-FPN is ambiguous in published literature, so define the local equation explicitly.

## Outcome

- Signal: useful

## Source Nodes

- MobilePixorBackBone
- SG-FPN
- CustomModel
- check_gates()
- test_standard_training_notebook.py