---
type: "architecture"
date: "2026-09-22T09:10:05.773415+00:00"
question: "What is the implemented B1_C2PSA end-to-end architecture for a Mermaid figure?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["raw", "lidar", "encoder", "mobilepixor", "stem", "feature", "attention", "fpn", "head", "loss"]
---

# Q: What is the implemented B1_C2PSA end-to-end architecture for a Mermaid figure?

## Answer

Legacy35 BEV input [B,35,800,704] passes through the MobilePIXOR stem and C2-C5 stages. A switchable C2PSA block processes C5 [B,96,50,44] without changing its shape, before the original sum-FPN fuses C5/C4/C3 into P4 [B,16,200,176]. Four dense heads predict class heatmaps, offsets, log-size, and doubled-yaw vectors. Training uses modified focal plus three masked L1 losses; inference uses sigmoid, local peaks, thresholding, decoding, and optional rotated NMS.

## Outcome

- Signal: useful

## Source Nodes

- raw
- lidar
- encoder
- mobilepixor
- stem
- feature
- attention
- fpn
- head
- loss
- training
- target