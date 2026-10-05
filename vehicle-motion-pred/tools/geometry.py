from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def to_local(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s], [-s, c]], dtype=np.float64)
    return ((np.asarray(xy, np.float64) - origin[:2]) @ rot.T).astype(np.float32)


def resample(poly: np.ndarray, n: int) -> np.ndarray:
    poly = np.asarray(poly, np.float64)[:, :2]
    if len(poly) == 1:
        return np.repeat(poly, n, axis=0)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] < 1e-6:
        return np.repeat(poly[:1], n, axis=0)
    q = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], axis=1)


def split_polyline(poly: np.ndarray, max_len_m: float) -> list[np.ndarray]:
    poly = np.asarray(poly, np.float64)[:, :2]
    if len(poly) < 2:
        return [poly]
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n_parts = max(1, int(np.ceil(s[-1] / max_len_m)))
    edges = np.linspace(0.0, s[-1], n_parts + 1)
    parts = []
    for a, b in zip(edges[:-1], edges[1:]):
        keep = (s >= a) & (s <= b)
        pts = poly[keep]
        if len(pts) < 2:
            q = np.array([a, b])
            pts = np.stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])], axis=1)
        parts.append(pts)
    return parts


def pick_lanes(polys_local: list[np.ndarray], n_lanes: int, n_points: int) -> tuple[np.ndarray, np.ndarray]:
    lanes = np.zeros((n_lanes, n_points, 2), np.float32)
    mask = np.zeros((n_lanes, n_points), bool)
    if not polys_local:
        return lanes, mask
    dist = [float(np.linalg.norm(p, axis=1).min()) for p in polys_local]
    for j, i in enumerate(np.argsort(dist)[:n_lanes]):
        lanes[j] = resample(polys_local[i], n_points)
        mask[j] = True
    return lanes, mask


def drivable_sdf(polygons: list[tuple[np.ndarray, list[np.ndarray]]], grid: dict) -> np.ndarray:
    from PIL import Image, ImageDraw

    n, res = int(grid["size"]), float(grid["res"])
    x0, y0 = float(grid["x_min"]), float(grid["y_min"])
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    pix = lambda a: np.column_stack([(a[:, 0] - x0) / res - 0.5, (a[:, 1] - y0) / res - 0.5]).ravel().tolist()
    for ext, holes in polygons:
        if len(ext) >= 3:
            draw.polygon(pix(np.asarray(ext)), fill=1)
    for ext, holes in polygons:
        for hole in holes:
            if len(hole) >= 3:
                draw.polygon(pix(np.asarray(hole)), fill=0)
    inside = np.asarray(img, dtype=bool)
    if not inside.any():
        return np.full((n, n), 127 * float(grid["scale"]), np.float32)
    if inside.all():
        return np.full((n, n), -127 * float(grid["scale"]), np.float32)
    d_out = distance_transform_edt(~inside) * res
    d_in = distance_transform_edt(inside) * res
    return (np.where(inside, -(d_in - 0.5 * res), d_out - 0.5 * res)).astype(np.float32)
