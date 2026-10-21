#!/usr/bin/env python3
"""
Generate a clean, wide, publication-grade qualitative BEV detection figure.
Requirements from user:
- Pure BEV detection (flat footprints on road & BEV plane)
- Model checkpoint: Final-h32 (best_ap.pt)
- NO titles, NO text labels, NO descriptive annotations
- Wide, spacious, high-contrast, publication-grade aesthetics
Output:
- paper/figures/qualitative_detections.pdf
- paper/figures/qualitative_detections.png
"""

import os
from PIL import Image
import numpy as np

def generate_clean_qualitative_figure():
    src_path = 'Result/encoderBevviz/camera_bev_footprints_strata_bev14_3x2.png'
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Source image not found: {src_path}")
    
    im = Image.open(src_path)
    print(f"Loaded source image: {src_path} (size: {im.size})")
    
    # 1. Crop off the top title banner (which occupied rows 0 to 135)
    # The actual content panels begin at y=135
    content = im.crop((0, 135, im.width, im.height))
    
    # 2. Trim excess outer white border pixels while preserving clean inter-row spacing
    arr = np.array(content)[:, :, :3]
    mask = np.any(arr < 250, axis=2)
    y_indices, x_indices = np.where(mask)
    
    ymin, ymax = y_indices.min(), y_indices.max()
    xmin, xmax = x_indices.min(), x_indices.max()
    
    # Add a subtle 4px clean breathing margin
    padding = 4
    x0 = max(0, xmin - padding)
    y0 = max(0, ymin - padding)
    x1 = min(content.width, xmax + 1 + padding)
    y1 = min(content.height, ymax + 1 + padding)
    
    clean_fig = content.crop((x0, y0, x1, y1)).convert('RGB')
    print(f"Cleaned figure dimensions: {clean_fig.size}")
    
    # 3. Save as PNG and high-resolution PDF
    out_png = 'paper/figures/qualitative_detections.png'
    out_pdf = 'paper/figures/qualitative_detections.pdf'
    
    clean_fig.save(out_png, 'PNG', optimize=True)
    print(f"Saved {out_png} ({os.path.getsize(out_png) / 1024 / 1024:.2f} MB)")
    
    clean_fig.save(out_pdf, 'PDF', resolution=300.0)
    print(f"Saved {out_pdf} ({os.path.getsize(out_pdf) / 1024 / 1024:.2f} MB)")

if __name__ == '__main__':
    generate_clean_qualitative_figure()
