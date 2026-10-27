from pathlib import Path
P=Path('/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment')
f=P/'sections/introduction.tex';s=f.read_text()
s=s.replace('In on-road autonomous driving, horizontal clearance margins and Time-to-Collision (TTC) along the ground plane dictate the vast majority of critical control decisions.','Horizontal clearance is one geometric input to on-road planning; estimating Time-to-Collision (TTC) additionally requires temporal or velocity information.')
s=s.replace('Can an ultra-compact LiDAR detector operating on a parameter-free BEV tensor approach the localization accuracy of heavy volumetric and pillar architectures while achieving higher end-to-end throughput on edge hardware?','What BEV detection accuracy and offline pipeline latency can a LiDAR detector achieve using a parameter-free input representation and a sub-million-parameter network?')
s=s.replace('we recognize that compact backbones cannot succeed through isolated architectural scaling alone; rather, they require an end-to-end co-design between sensor physical modeling and geometric loss supervision.','we combine statistical input encoding with compact feature extraction, augmentation, and geometric supervision.')
s=s.replace('Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) predictions are decoupled into independent \\textbf{GroupedHeader} towers sharing an initial $3 \\times 3$ Conv-BN-SiLU block.','Car ($K_1=1$) and Pedestrian/Cyclist ($K_2=2$) use separate \\textbf{GroupedHeader} towers. Within each group, the task branches share an initial $3 \\times 3$ Conv-BN-SiLU block.')
# Expose the representation -> architecture -> supervision chain in readable paragraphs.
s=s.replace(' Point-based direct architectures such as','\n\nPoint-based direct architectures such as').replace(' Proposal-free BEV architectures such as','\n\nProposal-free BEV architectures such as')
f.write_text(s)

f=P/'sections/methodology.tex';s=f.read_text()
s=s.replace('Figure~\\ref{fig:r37_overview} shows the data flow from the raw 3D point cloud to decoded oriented BEV bounding footprints.','Figure~\\ref{fig:r37_overview} shows the data flow from the raw 3D point cloud to decoded oriented BEV bounding footprints. During training, PC-GTA operates on points before Strata-BEV14 rasterization; at inference, the unaugmented scan is rasterized directly. Block-SimAM operates within the backbone, SG-FPN fuses its outputs, and the grouped heads feed geometry/quality supervision during training and decoding/NMS during inference. Here, 3D SimAM attention refers to modulation across feature channels and two spatial axes, rather than volumetric 3D convolution.')
s=s.replace('anchor point $P$','grid reference point $P$').replace('($753{,}569$ parameters total)','($753{,}569$ deployable parameters total)')
s=s.replace('and incorporates Range and Density-Adaptive (RDA) weighting','and incorporates the range-adaptive weighting denoted RDA in the implementation')
s=s.replace('The sampler provides five consistency mechanisms','The sampler provides five consistency mechanisms')
s=s.replace('the resulting physically consistent point cloud','the resulting augmented point cloud')
f.write_text(s)

f=P/'sections/results.tex';s=f.read_text()
repl={
'3. Balanced Multi-Class Detection.':'3. Per-Class Results.',
'Multi-Seed Statistical Robustness and Variance Analysis.':'Summary of Archived Seed Runs.',
'Systematic Architectural Progression.':'Historical Configuration Inventory.',
'1. Backbone Architecture (A0 $\\to$ A1).':'1. Historical Backbone Configurations (A0, A1).',
'2. BEV Input Representation (A1 $\\to$ A2 $\\to$ A3).':'2. Historical BEV Representations (A1--A3).',
'3. Decoupled Grouped Prediction Towers (A3 $\\to$ A4 $\\to$ A5).':'3. Historical Head and Sampler Configurations (A3--A5).',
'4. Physics-Consistent Ground-Truth Augmentation (A5 $\\to$ A6).':'4. Historical Sampler Recipes (A5--A7).',
'5. Multiscale Neck and Scale Gating (A6 $\\to$ A7 $\\to$ A8).':'5. Neck and Context Configurations (A7, A8, A8F).',
'6. Geometric Supervision and Localization Quality (A8 $\\to$ A9).':'6. Historical Geometric Supervision (A8F, A9).',
'7. Block-SimAM Energy Attention (A9 $\\to$ A10).':'7. Final Block-SimAM Configuration (A10).',
'Training Schedule Fairness and Convergence Validation.':'Checkpoint Selection and Timing Sensitivity.',
'Near-Range Safety Envelope':'Near-Range Detection',
'MobilePIXORNeXt (converged MobilePIXORNeXt deployment model)':'MobilePIXORNeXt',
'This qualitative comparison':'This qualitative comparison',
}
for a,b in repl.items(): s=s.replace(a,b)
old=next(l for l in s.splitlines() if l.startswith('\\captionof{figure}{Qualitative comparison'))
s=s.replace(old,r'\captionof{figure}{Selected BEV detections and foreground heatmaps for MobilePIXOR (Row A) and MobilePIXORNeXt (Row B) in two driving scenes. The panels permit qualitative inspection of response concentration and predicted footprints; they do not quantify dataset-level recall or isolate a component effect.}')
f.write_text(s)
print('Aligned overview, transitions, headings, and qualitative captions.')
