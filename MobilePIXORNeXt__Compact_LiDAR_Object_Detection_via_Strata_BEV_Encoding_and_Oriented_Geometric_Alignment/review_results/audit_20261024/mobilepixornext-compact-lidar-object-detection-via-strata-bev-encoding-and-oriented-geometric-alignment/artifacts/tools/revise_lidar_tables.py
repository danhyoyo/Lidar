from pathlib import Path
import re,csv,json
P=Path('/home/duyennh/AI_projects/research_lidar/Lidar/MobilePIXORNeXt__Compact_LiDAR_Object_Detection_via_Strata_BEV_Encoding_and_Oriented_Geometric_Alignment')
def strip_highlights(s):
    return re.sub(r'\\(?:textbf|underline)\{([0-9.]+)\}',r'\1',s)
def caption(s,new):
    # All target captions are single source lines.
    return re.sub(r'^\\caption\{.*\}$',lambda m:r'\caption{'+new+'}',s,flags=re.M)
def note(s,new):
    return re.sub(r'^\{\\footnotesize \\textit\{Note\.\}.*$',lambda m:r'{\footnotesize \textit{Note.} '+new+'}',s,flags=re.M)
f=P/'tables/comparison.tex';s=strip_highlights(f.read_text())
s=caption(s,r'Archived Moderate BEV AP (\%) on KITTI validation. Superscripts L and R identify the local loader-aligned evaluator and the pinned reference evaluator, respectively; both use 40-point interpolation with nominal IoU thresholds 0.70 for Car and 0.50 for Pedestrian/Cyclist. Protocol differences preclude a shared AP ranking. Parameter counts describe the evaluated implementations.')
s=s.replace('PIXOR (Car only)',r'PIXOR (Car only)$^{\mathrm{R}}$').replace('MobilePIXOR~',r'MobilePIXOR$^{\mathrm{L}}$~').replace('PointPillars~',r'PointPillars$^{\mathrm{R}}$~').replace('SECOND~',r'SECOND$^{\mathrm{R}}$~').replace('MobilePIXORNeXt (Ours)',r'MobilePIXORNeXt (Ours)$^{\mathrm{L}}$').replace('PointRCNN~',r'PointRCNN$^{\mathrm{R}}$~').replace('Part-A$^2$-Anchor~',r'Part-A$^2$-Anchor$^{\mathrm{R}}$~').replace('PV-RCNN~',r'PV-RCNN$^{\mathrm{R}}$~')
s=note(s,r'$\text{mAP}_{\mathrm{M}}$ averages the three Moderate class AP values; $\text{mAP}_9$ averages all nine class--difficulty cells. Dashes indicate unavailable classes. The MobilePIXORNeXt row is from the L4 seed-42 report (selected epoch 130); the separate T4 snapshot gives 71.46\% and 73.42\%, respectively. MobilePIXOR is a historical epoch-30 checkpoint from a run archived at 38 of 100 planned epochs. Reference models retain native preprocessing, ROI, checkpoint selection, and evaluation rules. The rows are descriptive and do not establish superiority across evaluators. Timing is reported separately in Table~\ref{tab:efficiency}.')
f.write_text(s)

f=P/'tables/efficiency.tex';s=strip_highlights(f.read_text())
s=caption(s,r'Archived sequential offline pipeline latency on KITTI validation (3,759 timed frames after 10 warmup frames), FP32 and batch size 1. MobilePIXOR uses L4; the remaining rows use T4. GPU labels are retained per row. Network-stage boundaries and native preprocessing differ by implementation.')
lines=s.splitlines()
for i,l in enumerate(lines):
    if l.startswith('MobilePIXOR~'):
        lines[i]=r'MobilePIXOR (L4)~\cite{nguyen2024indoor} & 0.598 & 11.05 & 90.50 & 85.92 & 11.64 & 84.11 & 112.52 & 124.94 \\'
    elif l.startswith('PointRCNN~'):
        lines[i]=l.replace('191.24','194.77').replace('235.10','222.37')
    elif l.startswith('Part-A$^2$-Anchor~'):
        lines[i]=l.replace('131.50','132.40').replace('172.80','169.13')
    elif l.startswith('PV-RCNN~'):
        lines[i]=l.replace('186.40','187.35').replace('224.15','219.07')
s='\n'.join(lines)+'\n'
s=note(s,r'Except for the explicitly marked L4 row, timing comes from the archived T4 snapshot. The offline boundary covers input preparation, H2D transfer, network execution, and final decoding/NMS; it excludes sensor acquisition, tracking, control, and AP computation. Learned encoding or internal proposal processing may lie inside the network stage of reference detectors. The 8.67\,ms / 115.32\,FPS model measurement belongs to a separate L4 report, rather than an isolated T4 benchmark. CPU/IO conditions and checkpoint identity must be matched before attributing timing differences to individual modules.')
f.write_text(s)

f=P/'tables/encoding.tex';s=strip_highlights(f.read_text())
s=caption(s,r'Archived BEV representation results on KITTI validation using the local evaluator with $R_{40}$ interpolation. The early configurations use a MobilePIXORNeXt backbone, F24 neck, and single H16 head. Training budgets and checkpoint-selection criteria differ, so these are descriptive representation comparisons. Payload is the FP32 tensor size for an $800\times704$ grid.')
s=note(s,r'Binary35 is the historical B1 run (100-epoch budget, loss-selected epoch 13); Strata-BEV8 and Strata-BEV14 are E8 and E14 (50-epoch budgets, AP-selected epochs 20 and 30). The reported forward rates are archived measurements, rather than a new common-T4 benchmark. The 14-channel tensor is 60.0\% smaller than Binary35. The AP gap does not isolate the effects of individual height statistics or centroid offsets. TriBand-BEV denotes the local three-channel adaptation used in this experiment, rather than a reproduction of the complete cited detector.')
s=s.replace('Standalone','Archived model')
f.write_text(s)

f=P/'tables/loss.tex';s=strip_highlights(f.read_text())
s=caption(s,r'Historical OGA--IQA factorial study (P1--P4) and Q-OGA alternative (P11), evaluated with local Moderate BEV AP $R_{40}$. These runs retain focal context and earlier independent task towers; they are distinct from the final shared-head detector. IQA uses $S_{\mathrm{det}}=\sqrt{P_{\mathrm{cls}}\hat{q}}$ for ranking.')
lines=s.splitlines()
for i,l in enumerate(lines):
    if l.startswith('Baseline Smooth-L1 & No'):
        lines[i]=r'P1: Baseline Smooth-L1 & No & 0.816 & 82.97 & 56.99 & 62.17 & 67.38 \\'
    elif l.startswith('OGA (MGIoU'):
        lines[i]=r'P2: OGA (MGIoU + $\pi$-Corner) & No & 0.816 & 84.09 & 57.93 & 64.88 & 68.97 \\'
    elif l.startswith('Baseline Smooth-L1 & Yes'):
        lines[i]=r'P3: Baseline Smooth-L1 & Yes & 0.853 & 84.13 & 59.45 & 67.34 & 70.31 \\'
    elif l.startswith('\\textbf{OGA (MGIoU'):
        lines[i]=r'P4: OGA (MGIoU + $\pi$-Corner) & Yes & 0.853 & 84.87 & 59.63 & 67.24 & 70.58 \\'
    elif l.startswith('Q-OGA (Quality'):
        lines[i]=r'P11: Q-OGA & No & 0.816 & 84.89 & 57.15 & 65.56 & 69.20 \\'
s='\n'.join(lines)+'\n'
s=note(s,r'P1--P4 used 50-epoch budgets and AP-selected epoch-50 checkpoints; P11 used a 50-epoch budget and an AP-selected epoch-40 checkpoint. Archived detector counts are 815,743 without IQA and 852,929 with IQA, a 37,186-parameter difference for these older towers. Q-OGA also changes corner smoothing and range weighting, so P11--P2 is a comparison of objective packages. No AP value in this table measures the proposed final 753,569-parameter architecture. Local per-class values and counts are taken from the archived P1--P4/P11 records.')
f.write_text(s)

f=P/'tables/attention.tex';s=strip_highlights(f.read_text())
s=caption(s,r'Historical attention-related configurations and final-detector snapshot on KITTI validation (local BEV AP $R_{40}$). Neck width, focal context, prediction towers, schedule, and checkpoint identity are not held fixed across all rows; this table is not a controlled placement ablation.')
s=s.replace('None (No attention)','None (P4, focal context)').replace('Stage-3 ($C_3$) & $F_{32}$','Stage-3 ($C_3$, R25) & $F_{24}$').replace('Standalone','Archived model')
s=note(s,r'The no-attention row is historical P4 (852,929 parameters, focal context); the final Block-SimAM row is the seed-42 shared-head detector (753,569 parameters, no focal context). The archived R25 stage-attention row uses F24, unlike the F32 final detector. The final 8.67\,ms model time comes from the L4 report; a common GPU/timing boundary has not been established for every row. Accordingly, differences in AP or latency cannot be assigned solely to attention placement. The separate T4 pipeline timing is reported in Table~\ref{tab:efficiency}.')
f.write_text(s)

# Replace the invented monotone construction with a source-linked inventory.
rows=[
('A0','B0','MobilePIXOR / Binary35 / legacy head','38/100; 30','41.40','42.27','11.05','90.50'),
('A1','B1','Binary35 / single H16 / F24 SumFPN','100; 13','47.60','49.08','5.88','169.97'),
('A2','E8','Strata-BEV8 / single H16 / F24','50; 20','47.37','48.95','6.06','165.02'),
('A3','E14','Strata-BEV14 / single H16 / F24','50; 30','50.14','52.00','5.92','169.01'),
('A4','G16','Grouped H16 / global augmentation / F24','50; 30','51.30','53.96','6.92','144.58'),
('A5','O32','Grouped H32 / simplified GT V1 / F32 SumFPN','50; 50','60.22','61.11','6.70','149.15'),
('A6','O32G16','Grouped H16 / simplified GT V2 / F32 SG-FPN','50; 50','66.25','68.45','7.45','134.25'),
('A7','H32','Grouped H32 / hybrid GT (6/8/10) / SumFPN','50; 50','68.12','70.05','6.72','148.82'),
('A8','S32','Grouped H32 / hybrid GT (15/10/10) / SG-FPN','50; 50','68.46','70.28','6.94','144.16'),
('A8F','P1','Focal context / baseline loss / no IQA','50; 50','67.38','69.46','7.77','128.66'),
('A9','P4','Focal context / OGA + IQA / independent towers','50; 50','70.58','72.66','8.19','122.16'),
('A10','Final S42','No focal / Block-SimAM / shared grouped towers','160; 130','71.50','73.45','8.67','115.32'),
]
s=r'''\begin{table*}[t]
\centering\footnotesize
\caption{Source-linked inventory of historical configurations and the final detector. Labels A0--A10 are retained for navigation; they do not denote a sequence of experiments changing one factor at a time. AP is local BEV $R_{40}$ on KITTI validation. The schedule column records archived/completed epochs (planned total when incomplete), followed by selected checkpoint epoch.}
\label{tab:components}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll>{\raggedright\arraybackslash}p{0.31\textwidth}rrrrr@{}}
\toprule
\textbf{Label} & \textbf{Run} & \textbf{Configuration} & \textbf{\shortstack{Epochs;\\selected}} & \textbf{\shortstack{mAP M\\(\%)}} & \textbf{\shortstack{mAP$_9$\\(\%)}} & \textbf{\shortstack{Model\\(ms)}} & \textbf{FPS} \\
\midrule
'''
for lab,run,desc,epochs,m,a,t,fps in rows:
    s+=' & '.join([lab,run,desc,epochs,m,a,t,fps])+r' \\'+'\n'
s+=r'''\bottomrule
\end{tabular*}
\par\vspace{2pt}
{\footnotesize \textit{Note.} B1 uses loss-based selection; the other listed AP checkpoints follow their archived selection records. A5 changes both head width and sampler relative to A4. A6 is a simplified-sampler H16 run, rather than the claimed PC-GTA H32/F24 addition. A7--A8 also change insertion quotas. A8F identifies the focal-context P1 result previously conflated with A8. A9 and A10 have different context modules, head architectures, and schedules. Historical model timings are archived values; the final model timing is from L4. These rows do not isolate backbone, head, PC-GTA, neck, or Block-SimAM gains under a common budget.}
\end{table*}
'''
(P/'tables/components.tex').write_text(s)

f=P/'tables/seed_robustness.tex';s=f.read_text()
s=caption(s,r'Descriptive summary of archived runs labeled by seeds 42, 43, and 44 (local BEV $R_{40}$, L4 evaluation). Means and sample standard deviations summarize these runs; their settings differ, so the dispersion is not an estimate of initialization variance under a fixed training protocol.')
s=s.replace(r'\end{table*}',r'''\par\vspace{2pt}
{\footnotesize \textit{Note.} Final MobilePIXORNeXt seed 42 completed 160 epochs (selected epoch 130), whereas seeds 43/44 completed 100 (selected epoch 100). MobilePIXOR seed 42 uses Binary35 and a run archived at 38/100 epochs (selected epoch 30); seeds 43/44 use Hist14 and 100-epoch runs (selected epochs 90/100). The numerical summary therefore combines changes in representation, training duration, and initialization. Gains are descriptive mean differences, not a matched-seed significance test.}
\end{table*}''')
f.write_text(s)

f=P/'tables/setup.tex';s=f.read_text()
s=caption(s,r'Final detector architecture and configured optimizer settings. The seed-42 run completed 160 epochs and selected epoch 130; seeds 43/44 completed 100 epochs. The L4 accuracy reports and separate T4 timing snapshot are distinguished in the text.')
s=s.replace(r'Grid resolution ($\Delta_x, \Delta_y, \Delta_z$) & $0.10\,\text{m} \times 0.10\,\text{m} \times 0.10\,\text{m}$',r'Horizontal grid / vertical strata & $0.10\,\text{m}\times0.10\,\text{m}$ / four uniform height bands')
s=s.replace('9.34\\,MB FP32','9.34\\,MB serialized checkpoint')
s=s.replace(r'100 epochs: 10 warmup + 90 cosine',r'Run-dependent; seed 42: 160 epochs')
s=s.replace('Converged deployment checkpoint evaluated at epoch 100','AP-selected seed-42 checkpoint at epoch 130')
s=s.replace('NVIDIA Tesla T4 GPU (16\\,GB GDDR6, 98.44\\,MiB peak)','L4 accuracy; separate T4 timing (98.44\\,MiB peak)')
f.write_text(s)
print('Updated nine tables with explicit provenance and source-backed corrections.')
