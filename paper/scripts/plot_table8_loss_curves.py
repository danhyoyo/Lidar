"""
Script to generate learning curves for Table 8: Loss Function & Quality Supervision Ablation.
Panels:
(a) Training Loss Trajectory across 50 epochs (log scale)
(b) Validation Loss Trajectory across 50 epochs (log scale)
(c) Validation Moderate mAP Trajectory (%) across evaluation checkpoints
"""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 0.8
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.6

RUNS = [
    {
        'id': 'smoothl1_no_iqa',
        'label': 'Smooth-L1 (No IQA)',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-002/artifacts/mobilepixornext-pc_gta-smoothl1_no_iqa-hist14-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#7f7f7f',
        'ls': ':',
        'marker': 'o',
        'lw': 1.6
    },
    {
        'id': 'mgiou_no_iqa',
        'label': 'MGIoU (No IQA)',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-002/artifacts/mobilepixornext-pc_gta-mgiou_loss-hist14-no_iqa-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#9467bd',
        'ls': '--',
        'marker': 's',
        'lw': 1.6
    },
    {
        'id': 'oga_no_iqa',
        'label': 'OGA (No IQA)',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-005/artifacts/mobilepixornext-pc_gta-oga_no_iqa-hist14-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#1f77b4',
        'ls': '--',
        'marker': '^',
        'lw': 1.8
    },
    {
        'id': 'smoothl1_iqa',
        'label': 'Smooth-L1 (+ IQA)',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-001/artifacts/mobilepixornext-pc_gta-smoothl1_loss-hist14-iqa-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#2ca02c',
        'ls': '-.',
        'marker': 'v',
        'lw': 1.8
    },
    {
        'id': 'oga_iqa',
        'label': 'OGA (+ IQA)',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-005/artifacts/mobilepixornext-pc_gta-oga_loss-hist14-iqa-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#ff7f0e',
        'ls': '-',
        'marker': 'D',
        'lw': 2.0
    },
    {
        'id': 'q_oga',
        'label': 'Q-OGA',
        'path': '/home/duyennh/Downloads/RUNS/Result/artifacts-20261028T183038Z-1-004/artifacts/mobilepixornext-pc_gta-q_oga_loss-hist14-no_iqa-sgfpn-simam-depth342-f32-h32-s42-50ep',
        'color': '#d62728',
        'ls': '-',
        'marker': '*',
        'lw': 2.2
    },
]

# Load data
data = {}
for r in RUNS:
    p = Path(r['path']) / 'metrics.jsonl'
    lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    data[r['id']] = lines[:50]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.4))

for r in RUNS:
    lines = data[r['id']]
    epochs = [l['epoch'] for l in lines]
    train_loss = [l.get('train_objective', 0) for l in lines]
    val_loss = [l.get('validation', {}).get('loss', 0) for l in lines]
    
    # AP evaluations
    ap_epochs = []
    ap_scores = []
    for l in lines:
        if l.get('ap_validation') and l['ap_validation'].get('score') is not None:
            ap_epochs.append(l['epoch'])
            ap_scores.append(l['ap_validation']['score'])
            
    # Panel (a): Training loss
    ax1.plot(epochs, train_loss, label=r['label'], color=r['color'], linestyle=r['ls'], linewidth=r['lw'], alpha=0.9)
    
    # Panel (b): Validation loss
    ax2.plot(epochs, val_loss, label=r['label'], color=r['color'], linestyle=r['ls'], linewidth=r['lw'], alpha=0.9)
    
    # Panel (c): Validation mAP Moderate
    if ap_epochs:
        ax3.plot(ap_epochs, ap_scores, label=r['label'], color=r['color'], linestyle=r['ls'],
                 marker=r['marker'], markersize=5.5, linewidth=r['lw'], alpha=0.95)

# Style (a)
ax1.set_title('(a) Total Training Objective', fontsize=11.5, fontweight='bold', pad=8)
ax1.set_yscale('log')
ax1.set_xlim(0, 52)
ax1.set_xticks([0, 10, 20, 30, 40, 50])
ax1.set_xlabel('Epoch', fontsize=10.5)
ax1.set_ylabel('Objective (log scale)', fontsize=10.5)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.35)

# Style (b)
ax2.set_title('(b) Total Validation Loss', fontsize=11.5, fontweight='bold', pad=8)
ax2.set_yscale('log')
ax2.set_xlim(0, 52)
ax2.set_xticks([0, 10, 20, 30, 40, 50])
ax2.set_xlabel('Epoch', fontsize=10.5)
ax2.set_ylabel('Loss (log scale)', fontsize=10.5)
ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.35)

# Style (c)
ax3.set_title(r'(c) Validation Moderate mAP ($R_{40}$)', fontsize=11.5, fontweight='bold', pad=8)
ax3.set_xlim(0, 52)
ax3.set_ylim(25, 72)
ax3.set_xticks([0, 10, 20, 30, 40, 50])
ax3.set_xlabel('Epoch', fontsize=10.5)
ax3.set_ylabel('Moderate mAP (%) ↑', fontsize=10.5)
ax3.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.35)
ax3.axhline(70.0, color='#888888', linestyle=':', linewidth=0.9, alpha=0.7)
ax3.text(2, 70.3, '70% threshold', fontsize=8.5, color='#666666', style='italic')

# Unified legend at bottom
handles, labels = ax3.get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc='lower center',
    bbox_to_anchor=(0.5, -0.05),
    ncol=6,
    frameon=True,
    facecolor='#f9f9f9',
    edgecolor='#d0d0d0',
    fontsize=9.5,
    borderpad=0.6,
    columnspacing=1.4,
    handlelength=2.2
)

plt.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.90,
    bottom=0.20,
    wspace=0.22
)

out_pdf = Path('paper/figures/loss_ablation_curves.pdf')
out_png = Path('paper/figures/loss_ablation_curves.png')
plt.savefig(out_pdf, format='pdf', dpi=300, bbox_inches='tight')
plt.savefig(out_png, format='png', dpi=300, bbox_inches='tight')
print(f'Successfully generated {out_pdf} and {out_png}')
