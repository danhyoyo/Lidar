import torch


def filter_roi_points(points: torch.Tensor, geometry: dict) -> torch.Tensor:
    """Filter non-finite points and keep points within ROI geometry bounds."""
    eps = 0.001
    valid = torch.isfinite(points[:, :4]).all(dim=1)
    valid &= points[:, 0] > (geometry["x_min"] + eps)
    valid &= points[:, 0] < (geometry["x_max"] - eps)
    valid &= points[:, 1] > (geometry["y_min"] + eps)
    valid &= points[:, 1] < (geometry["y_max"] - eps)
    valid &= points[:, 2] > (geometry["z_min"] + eps)
    valid &= points[:, 2] < (geometry["z_max"] - eps)
    pts = points[valid].clone()
    if pts.shape[0] > 0:
        pts[:, 3] = torch.clamp(pts[:, 3], 0.0, 1.0)
    return pts


def group_and_sort_pillars(
    points: torch.Tensor,
    geometry: dict,
    max_points_per_pillar: int = 20,
    max_pillars: int = 32000,
):
    """
    Fully-vectorized, parallel grouping of points into BEV pillars with elevation (Z) sorting.
    Zero Python per-pillar loops: executes completely in parallel on GPU.

    Returns:
        pillar_features: (P, max_points_per_pillar, 8)
        pillar_indices: (P, 2) [y_idx, x_idx]
        num_pillars: int
    """
    pts = filter_roi_points(points, geometry)
    device = points.device

    if pts.shape[0] == 0:
        empty_feat = torch.zeros(
            (0, max_points_per_pillar, 8), dtype=torch.float32, device=device
        )
        empty_idx = torch.zeros((0, 2), dtype=torch.int64, device=device)
        return empty_feat, empty_idx, 0

    x_res = float(geometry["x_res"])
    y_res = float(geometry["y_res"])
    x_min = float(geometry["x_min"])
    y_min = float(geometry["y_min"])
    z_min = float(geometry["z_min"])
    z_max = float(geometry["z_max"])
    z_c = (z_min + z_max) / 2.0

    x_size = int(round((float(geometry["x_max"]) - x_min) / x_res))

    # Grid indices
    x_idx = ((pts[:, 0] - x_min) / x_res).long()
    y_idx = ((pts[:, 1] - y_min) / y_res).long()
    flat_ids = y_idx * x_size + x_idx

    # Step 1: Lexicographical sort (Z first, then stable sort by flat_id)
    # This guarantees points are grouped by pillar, and within each pillar are sorted ascending by Z
    z_order = torch.argsort(pts[:, 2])
    pts = pts[z_order]
    flat_ids = flat_ids[z_order]

    pillar_order = torch.argsort(flat_ids, stable=True)
    pts = pts[pillar_order]
    flat_ids = flat_ids[pillar_order]

    # Step 2: Parallel run-length indexing of contiguous pillars
    diff = flat_ids[1:] != flat_ids[:-1]
    starts = torch.cat([torch.tensor([True], device=device), diff])
    pillar_run_starts = torch.where(starts)[0]
    num_unique_pillars = int(pillar_run_starts.shape[0])
    num_pillars = min(num_unique_pillars, max_pillars)

    # Pillar ID for every point
    point_pillar_ids = torch.cumsum(starts.long(), dim=0) - 1
    # Intra-pillar index (0, 1, 2... for each point within its pillar)
    intra_pillar_indices = torch.arange(pts.shape[0], device=device) - pillar_run_starts[point_pillar_ids]

    # Step 3: Filter by capacity limits
    keep = (intra_pillar_indices < max_points_per_pillar) & (point_pillar_ids < num_pillars)
    pts_kept = pts[keep]
    pillar_id_kept = point_pillar_ids[keep]
    intra_idx_kept = intra_pillar_indices[keep]

    # Unique active pillar coordinates
    active_flat_ids = flat_ids[pillar_run_starts[:num_pillars]]
    pillar_indices = torch.zeros((num_pillars, 2), dtype=torch.int64, device=device)
    pillar_indices[:, 0] = active_flat_ids // x_size  # y_idx
    pillar_indices[:, 1] = active_flat_ids % x_size   # x_idx

    # Centers of active pillars
    xc = (pillar_indices[:, 1].float() + 0.5) * x_res + x_min
    yc = (pillar_indices[:, 0].float() + 0.5) * y_res + y_min

    # Step 4: Point feature enrichment: [dx, dy, dz, x, y, z, r, z - z_min]
    dx = pts_kept[:, 0] - xc[pillar_id_kept]
    dy = pts_kept[:, 1] - yc[pillar_id_kept]
    dz = pts_kept[:, 2] - z_c
    z_rel = pts_kept[:, 2] - z_min

    enriched = torch.stack(
        [dx, dy, dz, pts_kept[:, 0], pts_kept[:, 1], pts_kept[:, 2], pts_kept[:, 3], z_rel],
        dim=-1,
    )

    # Step 5: Direct parallel assignment into output tensor
    pillar_features = torch.zeros(
        (num_pillars, max_points_per_pillar, 8), dtype=torch.float32, device=device
    )
    pillar_features[pillar_id_kept, intra_idx_kept] = enriched

    return pillar_features, pillar_indices, num_pillars
