"""
Script to generate Figure 9: Multi-task training and validation convergence trajectories across 50 epochs
for progressive architectural ablation milestones A0 - A8.
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set matplotlib font and rendering defaults for Springer / academic papers
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.4

BASE_DIR = Path('/home/duyennh/Downloads/A0_A8_Milestones')

MODELS = [
    {
        'id': 'A0',
        'label': r'$A_0$ · MobilePIXOR (Baseline)',
        'color': '#7f7f7f',
        'ls': '--',
        'lw': 1.6
    },
    {
        'id': 'A1',
        'label': r'$A_1$ · + MobilePIXORNeXt Backbone',
        'color': '#1f77b4',
        'ls': '-',
        'lw': 1.4
    },
    {
        'id': 'A2',
        'label': r'$A_2$ · + Strata-BEV14',
        'color': '#2ca02c',
        'ls': '-',
        'lw': 1.4
    },
    {
        'id': 'A3',
        'label': r'$A_3$ · + Decoupled Grouped Head',
        'color': '#9467bd',
        'ls': '-',
        'lw': 1.4
    },
    {
        'id': 'A4',
        'label': r'$A_4$ · + Wide Head Capacity',
        'color': '#8c564b',
        'ls': '-',
        'lw': 1.4
    },
    {
        'id': 'A5',
        'label': r'$A_5$ · + PC-GTA Augmentation',
        'color': '#e377c2',
        'ls': '-',
        'lw': 1.4
    },
    {
        'id': 'A6',
        'label': r'$A_6$ · + SG-FPN Multi-Scale Fusion',
        'color': '#ff7f0e',
        'ls': '-',
        'lw': 1.5
    },
    {
        'id': 'A7',
        'label': r'$A_7$ · + OGA + IQA Objective',
        'color': '#17becf',
        'ls': '-',
        'lw': 1.6
    },
    {
        'id': 'A8',
        'label': r'$A_8$ · + Block-SimAM (Final Model)',
        'color': '#d62728',
        'ls': '-',
        'lw': 2.2
    },
]

# Load metrics for each model
data = {}
for m in MODELS:
    p = BASE_DIR / m['id'] / 'metrics.jsonl'
    lines = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    data[m['id']] = lines[:50]  # First 50 epochs

SUBPLOTS = [
    {
        'title': 'Training loss',
        'key': lambda d: d.get('train_objective'),
        'has_metric': lambda m, d: d.get('train_objective') is not None,
    },
    {
        'title': 'Validation loss',
        'key': lambda d: d.get('validation', {}).get('loss'),
        'has_metric': lambda m, d: d.get('validation', {}).get('loss') is not None,
    },
    {
        'title': 'Classification',
        'key': lambda d: d.get('validation', {}).get('cls'),
        'has_metric': lambda m, d: d.get('validation', {}).get('cls') is not None,
    },
    {
        'title': 'Geometry',
        'key': lambda d: d.get('validation', {}).get('geo'),
        # Geometry is only present in OGA models (A7, A8)
        'has_metric': lambda m, d: m['id'] in ('A7', 'A8') and d.get('validation', {}).get('geo', 0) > 0,
    },
    {
        'title': 'Localization Quality (IoU)',
        'key': lambda d: d.get('validation', {}).get('iou'),
        # IoU quality loss is only present in IQA models (A7, A8)
        'has_metric': lambda m, d: m['id'] in ('A7', 'A8') and d.get('validation', {}).get('iou', 0) > 0,
    },
    {
        'title': 'Coordinate Offset',
        'key': lambda d: d.get('validation', {}).get('offset'),
        'has_metric': lambda m, d: d.get('validation', {}).get('offset') is not None,
    },
    {
        'title': 'Bounding Box Size',
        'key': lambda d: d.get('validation', {}).get('size'),
        'has_metric': lambda m, d: d.get('validation', {}).get('size') is not None,
    },
    {
        'title': 'Yaw Angle',
        'key': lambda d: d.get('validation', {}).get('yaw'),
        'has_metric': lambda m, d: d.get('validation', {}).get('yaw') is not None,
    },
]

fig, axes = plt.subplots(4, 2, figsize=(13.0, 15.8))
axes_flat = axes.flatten()

for idx, sp in enumerate(SUBPLOTS):
    ax = axes_flat[idx]
    
    for m in MODELS:
        m_lines = data[m['id']]
        ep_list = []
        val_list = []
        for line in m_lines:
            ep = line['epoch']
            if sp['has_metric'](m, line):
                val = sp['key'](line)
                if val is not None and val > 0:
                    ep_list.append(ep)
                    val_list.append(val)
        
        if ep_list and val_list:
            ax.plot(
                ep_list, val_list,
                label=m['label'],
                color=m['color'],
                linestyle=m['ls'],
                linewidth=m['lw'],
                alpha=0.92
            )
            
    ax.set_title(sp['title'], fontsize=12, fontweight='bold', pad=8)
    ax.set_yscale('log')
    ax.set_xlim(0, 52)
    ax.set_xticks([0, 10, 20, 30, 40, 50])
    ax.set_xlabel('Epoch', fontsize=10.5)
    ax.set_ylabel('Loss (log scale)', fontsize=10.5)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.35)
    ax.tick_params(axis='both', which='major', labelsize=9.5)

legend_handles = []
legend_labels = []
for m in MODELS:
    h = plt.Line2D([0], [0], color=m['color'], linestyle=m['ls'], linewidth=m['lw'])
    legend_handles.append(h)
    legend_labels.append(m['label'])

fig.legend(
    legend_handles, legend_labels,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.012),
    ncol=3,
    frameon=True,
    facecolor='#f9f9f9',
    edgecolor='#d0d0d0',
    fontsize=9.8,
    borderpad=0.8,
    columnspacing=2.2,
    handlelength=2.4
)

plt.subplots_adjust(
    left=0.08,
    right=0.96,
    top=0.96,
    bottom=0.10,
    wspace=0.22,
    hspace=0.38
)

out_pdf = Path('paper/figures/training_curves_all_losses.pdf')
plt.savefig(out_pdf, format='pdf', dpi=300)
out_png = Path('paper/figures/training_curves_all_losses.png')
plt.savefig(out_png, format='png', dpi=300)
print(f'Successfully generated {out_pdf} ({out_pdf.stat().st_size} bytes)')
print(f'Successfully generated {out_png} ({out_png.stat().st_size} bytes)')
