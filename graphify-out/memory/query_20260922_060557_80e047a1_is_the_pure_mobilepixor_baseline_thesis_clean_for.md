---
type: "query"
date: "2026-09-22T06:05:57.552622+00:00"
question: "Is the pure MobilePIXOR baseline thesis-clean for independent improvements, and is the Legacy35 voxel input safe to use as inherited prior work?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["MobilePixorBackBone", "voxelize()", "encode_bev()", "LossFunction"]
---

# Q: Is the pure MobilePIXOR baseline thesis-clean for independent improvements, and is the Legacy35 voxel input safe to use as inherited prior work?

## Answer

Yes, conditionally. The pure kitti_mobilepixor_baseline configuration disables augmentation globally and selects the baseline MobilePIXOR backbone and baseline loss, avoiding the senior thesis-specific CoordAtt, UWAG, and augmentation contributions. Legacy35 is the default binary_slices BEV voxel encoder and may be used as inherited baseline machinery, provided it is defined and cited rather than claimed as novel. Academic cleanliness also requires proper code licensing/permission and independent novelty evidence.

## Outcome

- Signal: useful

## Source Nodes

- MobilePixorBackBone
- voxelize()
- encode_bev()
- LossFunction