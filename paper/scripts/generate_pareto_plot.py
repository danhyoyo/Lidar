#!/usr/bin/env python3
"""
Generate publication-quality Pareto Frontier plots for MobilePIXORNeXt paper.
Plots:
  (a) Moderate mAP (%) vs. Model Parameters (M) [log-scale]
  (b) Moderate mAP (%) vs. End-to-End Pipeline Latency (ms) on Tesla T4
Output:
  paper/figures/pareto_tradeoff.pdf
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.titlesize': 14,
    'mathtext.fontset': 'cm',
    'lines.linewidth': 1.8,
    'axes.linewidth': 1.0,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
})

# Model data
# (Name, params_M, map_mod, e2e_latency_ms, category, marker, color)
models = [
    {
        'name': 'MobilePIXORNeXt (Ours)',
        'params': 0.754,
        'map': 71.46,
        'latency': 60.96,
        'cat': 'ours',
        'marker': '*',
        'color': '#D9381E',
        'size': 220,
        'offset_p': (-15, 10),
        'offset_l': (10, 5),
    },
    {
        'name': 'MobilePIXOR',
        'params': 0.598,
        'map': 41.40,
        'latency': 85.92,
        'cat': 'bev_light',
        'marker': 'o',
        'color': '#7F8C8D',
        'size': 85,
        'offset_p': (10, -5),
        'offset_l': (10, -10),
    },
    {
        'name': 'PointPillars',
        'params': 4.835,
        'map': 70.13,
        'latency': 63.84,
        'cat': 'pillar',
        'marker': 's',
        'color': '#2980B9',
        'size': 95,
        'offset_p': (10, -12),
        'offset_l': (10, -12),
    },
    {
        'name': 'SECOND',
        'params': 5.326,
        'map': 72.09,
        'latency': 93.70,
        'cat': 'sparse3d',
        'marker': 'D',
        'color': '#E67E22',
        'size': 90,
        'offset_p': (10, 5),
        'offset_l': (10, 5),
    },
    {
        'name': 'PointRCNN',
        'params': 4.039,
        'map': 73.38,
        'latency': 194.50,
        'cat': 'heavy3d',
        'marker': '^',
        'color': '#8E44AD',
        'size': 95,
        'offset_p': (-70, 8),
        'offset_l': (-75, -8),
    },
    {
        'name': 'PV-RCNN',
        'params': 13.116,
        'map': 74.44,
        'latency': 188.60,
        'cat': 'heavy3d',
        'marker': 'p',
        'color': '#16A085',
        'size': 105,
        'offset_p': (10, -8),
        'offset_l': (-65, 8),
    },
    {
        'name': 'Part-A$^2$',
        'params': 63.815,
        'map': 76.28,
        'latency': 133.75,
        'cat': 'heavy3d',
        'marker': 'h',
        'color': '#27AE60',
        'size': 110,
        'offset_p': (-60, 5),
        'offset_l': (10, 5),
    },
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

# -------------------------------------------------------------
# Subplot (a): Accuracy vs. Model Size (Parameters)
# -------------------------------------------------------------
ax1.set_title('(a) Accuracy vs. Parameter Footprint', pad=10, fontweight='bold')
ax1.set_xlabel('Model Parameters (M) [log scale]')
ax1.set_ylabel('KITTI Moderate mAP (%)')
ax1.set_xscale('log')
ax1.set_ylim(35, 80)
ax1.set_xlim(0.4, 90)
ax1.grid(True, which='both', linestyle='--', alpha=0.4)

# Draw Pareto frontier line (Ours -> SECOND -> PointRCNN -> PV-RCNN -> Part-A2)
pareto_params_x = [0.754, 5.326, 63.815]
pareto_params_y = [71.46, 72.09, 76.28]
ax1.plot(pareto_params_x, pareto_params_y, '--', color='#D9381E', alpha=0.5, zorder=2, label='Pareto Frontier')
ax1.fill_between(pareto_params_x, pareto_params_y, 35, color='#D9381E', alpha=0.04)

for m in models:
    ax1.scatter(m['params'], m['map'], s=m['size'], c=m['color'], 
                marker=m['marker'], edgecolors='k', linewidth=0.8, zorder=5,
                label=m['name'])
    ax1.annotate(m['name'],
                 (m['params'], m['map']),
                 xytext=m['offset_p'],
                 textcoords='offset points',
                 fontsize=9,
                 fontweight='bold' if 'Ours' in m['name'] else 'normal',
                 color='#900C3F' if 'Ours' in m['name'] else '#2C3E50',
                 zorder=6)

# -------------------------------------------------------------
# Subplot (b): Accuracy vs. End-to-End Latency
# -------------------------------------------------------------
ax2.set_title('(b) Accuracy vs. End-to-End Pipeline Latency', pad=10, fontweight='bold')
ax2.set_xlabel('End-to-End Latency on Tesla T4 (ms) $\\downarrow$')
ax2.set_ylabel('KITTI Moderate mAP (%) $\\uparrow$')
ax2.set_ylim(35, 80)
ax2.set_xlim(45, 215)
ax2.grid(True, linestyle='--', alpha=0.4)

# Pareto frontier in latency vs mAP (lower latency is better, higher mAP is better)
# Non-dominated models: MobilePIXORNeXt (60.96ms, 71.46%), SECOND (93.70ms, 72.09%), Part-A2 (133.75ms, 76.28%)
pareto_lat_x = [60.96, 93.70, 133.75]
pareto_lat_y = [71.46, 72.09, 76.28]
ax2.plot(pareto_lat_x, pareto_lat_y, '--', color='#D9381E', alpha=0.5, zorder=2, label='Pareto Frontier')
ax2.fill_between(pareto_lat_x, pareto_lat_y, 35, color='#D9381E', alpha=0.04)

for m in models:
    ax2.scatter(m['latency'], m['map'], s=m['size'], c=m['color'], 
                marker=m['marker'], edgecolors='k', linewidth=0.8, zorder=5)
    ax2.annotate(m['name'],
                 (m['latency'], m['map']),
                 xytext=m['offset_l'],
                 textcoords='offset points',
                 fontsize=9,
                 fontweight='bold' if 'Ours' in m['name'] else 'normal',
                 color='#900C3F' if 'Ours' in m['name'] else '#2C3E50',
                 zorder=6)

# Real-time boundary line at 100 ms (10 FPS)
ax2.axvline(x=100.0, color='#C0392B', linestyle=':', linewidth=1.2, alpha=0.7)
ax2.text(102, 38, 'Real-Time Limit (10 FPS)', color='#C0392B', fontsize=8.5, rotation=90)

# Add custom unified legend at the bottom of the figure
handles, labels = ax1.get_legend_handles_labels()
# Remove Pareto Frontier from start if duplicated
fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.06),
           frameon=True, framealpha=0.9, edgecolor='#BDC3C7')

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)

output_path = 'paper/figures/pareto_tradeoff.pdf'
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved Pareto trade-off figure to {output_path}")
