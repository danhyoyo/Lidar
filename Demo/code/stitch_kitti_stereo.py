#!/usr/bin/env python3
"""Align real KITTI stereo images and join them with a graph-cut seam.

Stereo images overlap almost completely. Their translation introduces depth
parallax, so a background homography provides only a small field-of-view gain.
No synthetic image content is generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def stitch(left_path, right_path, output_path):
    left, right = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (left_path, right_path)]
    if left is None or right is None or left.shape != right.shape:
        raise ValueError('Expected readable stereo RGB images with matching dimensions')
    sift = cv2.SIFT_create()
    features = [sift.detectAndCompute(image, None) for image in (left, right)]
    if any(descriptors is None for _, descriptors in features):
        raise ValueError('Insufficient image features')
    pairs = cv2.BFMatcher().knnMatch(features[1][1], features[0][1], k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance]
    if len(matches) < 30:
        raise ValueError('Insufficient matching features for stitching')
    source = np.float32([features[1][0][m.queryIdx].pt for m in matches])
    destination = np.float32([features[0][0][m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(42)
    transform, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 2.,
                                           maxIters=5000, confidence=.995)
    if transform is None or not np.isfinite(transform).all() or inliers.sum() < 30:
        raise ValueError('No reliable image alignment')
    error = np.linalg.norm(cv2.perspectiveTransform(source[:, None, :], transform)[:, 0, :]-destination, axis=1)
    height, width = left.shape[:2]
    corners = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    warped_corners = cv2.perspectiveTransform(corners[:, None, :], transform)[:, 0, :]
    canvas_width = max(width, int(np.ceil(warped_corners[:, 0].max())))
    if canvas_width > 2 * width:
        raise ValueError('Implausible stereo alignment extent')
    warped_right = cv2.warpPerspective(right, transform, (canvas_width, height))
    right_valid = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform,
                                     (canvas_width, height), flags=cv2.INTER_NEAREST)
    left_canvas = np.zeros_like(warped_right)
    left_canvas[:, :width] = left
    left_valid = np.zeros_like(right_valid)
    left_valid[:, :width] = 255
    union = (left_valid > 0) | (right_valid > 0)
    # Trim the uneven outer boundary to a filled rectangle rather than adding
    # content or leaving black corners. All original left-image rows are kept.
    last_valid = [int(np.flatnonzero(row)[-1]) for row in union]
    final_width = min(last_valid) + 1
    if not union[:, :final_width].all():
        raise ValueError('Aligned sources do not cover the full output rectangle')
    warped_right, right_valid = warped_right[:, :final_width], right_valid[:, :final_width]
    left_canvas, left_valid = left_canvas[:, :final_width], left_valid[:, :final_width]
    # Keep the reference view across foreground objects and place the seam in
    # the right-hand overlap. This avoids unnecessary stereo ghosting elsewhere.
    candidate_right = right_valid.copy()
    candidate_right[:, :max(0, width-180)] = 0
    masks = [cv2.UMat(left_valid.copy()), cv2.UMat(candidate_right)]
    cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD').find(
        [left_canvas.astype(np.float32), warped_right.astype(np.float32)], [(0, 0), (0, 0)], masks)
    right_seam = masks[1].get() > 0
    alpha = cv2.GaussianBlur(right_seam.astype(np.float32), (3, 3), .7)
    alpha[right_valid == 0] = 0
    alpha[left_valid == 0] = 1
    result = np.clip(left_canvas*(1-alpha[:, :, None]) + warped_right*alpha[:, :, None], 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise OSError('Could not write stitched image')
    return {
        'left_source':str(left_path), 'right_source':str(right_path),
        'left_sha256':hashlib.sha256(left_path.read_bytes()).hexdigest(),
        'right_sha256':hashlib.sha256(right_path.read_bytes()).hexdigest(),
        'method':'SIFT + RANSAC background homography + graph-cut seam + narrow feathering',
        'right_to_left_homography':transform.tolist(), 'matches':len(matches),
        'inliers':int(inliers.sum()), 'median_inlier_alignment_error_px':float(np.median(error[inliers.ravel()>0])),
        'input_size_pixels':[width, height], 'output_size_pixels':[final_width, height],
        'right_source_contributing_pixels':int((alpha > .5).sum()),
        'left_source_contributing_pixels':int((alpha < .5).sum()),
        'synthetic_content':False, 'annotations':False,
        'limitation':'Highly overlapping translated stereo views; foreground depth parallax cannot all be represented by one homography',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left', required=True, type=Path)
    parser.add_argument('--right', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--metadata', type=Path)
    args = parser.parse_args(argv)
    report = stitch(args.left.resolve(), args.right.resolve(), args.output.resolve())
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    main()
