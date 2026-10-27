Execution: sequential single-agent.

| Producer | Output | Transform / consumer |
|---|---|---|
| Raw LiDAR | xyz/intensity points | ROI filter; PC-GTA before rasterization during training only |
| PC-GTA | augmented points + adjusted labels | Strata-BEV14; raw points bypass this step at inference |
| Strata-BEV14 | B×14×800×704, row=v(y), col=u(x) | ConvNeXt stem and backbone |
| Backbone including Block-SimAM | stride 4/8/16 features | SG-FPN lateral projections and gated top-down fusion |
| SG-FPN | B×32×200×176 | independent class-group stems, shared tasks within each group |
| Grouped heads | classification / offset / size / yaw / quality | train: decoded geometry and detached projection-quality targets; infer: score fusion, filtering, decoding, NMS |
| OGA/IQA/T-SBUW | group objectives and task weights | shared-backbone gradient sum, not a conserved gradient norm |
| Decoding / NMS | class/score/x/y/l/w/orientation modulo π | final BEV footprints; no vertical extent or velocity |

The document contains depth-2 subsections but no depth-3 numbered units; subsection_context_polish is not applicable. Adjacent paragraphs were reviewed directly.
