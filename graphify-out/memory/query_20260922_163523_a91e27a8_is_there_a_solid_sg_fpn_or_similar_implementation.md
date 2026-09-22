---
type: "query"
date: "2026-09-22T16:35:23.147401+00:00"
question: "Is there a solid SG-FPN or similar implementation in the current codebase, and is it runnable?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["SG-FPN", "C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)", "C4→C3 Fusion (Sum-FPN or SG-FPN Gate C3)", "MobilePixorBackBone", "CustomModel", "check_gates()"]
---

# Q: Is there a solid SG-FPN or similar implementation in the current codebase, and is it runnable?

## Answer

Expanded through graph vocabulary: [fpn, fusion, gate, gated, gates, backbone, mobilepixor, config, test]. An SG-FPN implementation exists and is runnable in mobilepixor_coordinate_attention.py through model.scale_gated_fpn. It uses zero-initialized depthwise 3x3 gates at C4 and C3, adds 480 parameters, and starts exactly equivalent to sum-FPN. The dedicated gate test passes and the deprecated A3 config builds and forwards successfully with Legacy35. However, CustomModel only routes this flag for backbone=mobilepixor_coordatt; the current B0/B1 backbone=mobilepixor path ignores it, so adding the JSON field to B0/B1 would not activate SG-FPN. No BiFPN, PANet/PAN-FPN, or ASFF implementation was found. Porting SG-FPN to clean B0/B1 requires a separate switch in mobilepixor.py and model.py plus new ablation configs; its thesis-derived provenance must also be addressed.

## Outcome

- Signal: useful

## Source Nodes

- SG-FPN
- C5→C4 Fusion (Sum-FPN or SG-FPN Gate C4)
- C4→C3 Fusion (Sum-FPN or SG-FPN Gate C3)
- MobilePixorBackBone
- CustomModel
- check_gates()