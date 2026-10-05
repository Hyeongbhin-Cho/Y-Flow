from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from vfm.config import load_config
from vfm.constraints import NuScenesConstraint
from vfm.data import VehicleTrajDataset, load_meta, load_split, to_device
from vfm.indep import boxes, headings, load_agents, sat_overlap

CLS = {0: "vehicle", 1: "pedestrian", 2: "cyclist", 3: "static"}


def overlap_steps(pred, fsize, ag_fut, ag_yaw, ag_size, ag_mask, chunk=64):
    n, k, t = pred.shape[:3]
    yaw = headings(pred)
    out = np.zeros((n, k, t, ag_fut.shape[1]), bool)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        fb = boxes(pred[s:e], yaw[s:e], fsize[s:e, 0, None, None], fsize[s:e, 1, None, None])
        ab = boxes(ag_fut[s:e], ag_yaw[s:e], ag_size[s:e, :, 0, None], ag_size[s:e, :, 1, None])
        ov = sat_overlap(fb[:, :, None], ab[:, None])
        ov &= ag_mask[s:e, None]
        ov &= (ag_size[s:e, :, 0] > 0)[:, None, :, None]
        out[s:e] = ov.transpose(0, 1, 3, 2)
    return out


def cv_agents(ag):
    f, m = ag["ag_fut"], ag["ag_mask"]
    v = np.where((m[..., 0] & m[..., 1])[..., None], f[:, :, 1] - f[:, :, 0], 0.0)
    t = np.arange(f.shape[2], dtype=np.float32)
    cv = f[:, :, :1] + v[:, :, None] * t[None, None, :, None]
    yaw = np.repeat(ag["ag_yaw"][:, :, :1], f.shape[2], 2)
    mask = np.repeat(m[:, :, :1], f.shape[2], 2)
    return cv.astype(np.float32), yaw, mask


def pct(x):
    return f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="datasets/nuscenes_v2")
    ap.add_argument("--config", default="configs/nuscenes_v2.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--pred", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    ccfg = dict(cfg.constraints)
    meta = load_meta(args.cache)
    a = load_split(args.cache, args.split, args.limit)
    n = len(a["fut"])
    ag = load_agents(args.cache, args.split, a["scene_id"])
    tol = float(ccfg.get("footprint_tol", 0.5))
    fpm = float(ccfg.get("fp_margin", 0.0))
    hz = float(meta["sample_hz"])
    hor = float(ccfg.get("coll_horizon_s", 1e9))
    fsize = a.get("focal_size")
    if fsize is None:
        fsize = np.tile([[4.5, 1.9]], (n, 1)).astype(np.float32)
    gt = a["fut"][:, None].copy()
    gt = np.where(a["fut_mask"][:, None, :, None], gt, gt[:, :, :1])

    def fp_max(p):
        ds = VehicleTrajDataset(a, optional=True)
        out = []
        for s in range(0, n, 256):
            b = to_device({k: v[s:s + 256] for k, v in ds.tensors.items()}, dev)
            c = NuScenesConstraint(b, meta, ccfg)
            out.append(c.footprint_offroad(torch.from_numpy(p[s:s + 256]).float().to(dev), c.focal_l, c.focal_w).cpu().numpy())
        return np.concatenate(out)

    g_fp = fp_max(gt)[:, 0]
    g_ov = overlap_steps(gt, fsize, ag["ag_fut"], ag["ag_yaw"], ag["ag_size"], ag["ag_mask"]).any((2, 3))[:, 0]
    cvf, cvy, cvm = cv_agents(ag)
    print(f"scenes={n}  footprint_tol={tol}  fp_margin={fpm}  coll_horizon_s={hor}  coll_margin={ccfg.get('coll_margin')}")
    report = {}
    for f in args.pred:
        p = np.load(f)[:n]
        name = f.split("/")[-1]
        r = {}
        ok = g_fp <= tol
        v = fp_max(p)[ok]
        fail = v > tol
        r["offroad_traj_gtok"] = float(fail.mean())
        r["offroad_scene_gtok"] = float(fail.any(1).mean())
        fv = v[fail]
        r["offroad_fail_sdf_in_(tol,fp_margin]"] = float(((fv > tol) & (fv <= fpm)).mean()) if fv.size else 0.0
        r["offroad_fail_sdf_in_(fp_margin,fp_margin+1]"] = float(((fv > fpm) & (fv <= fpm + 1)).mean()) if fv.size else 0.0
        r["offroad_fail_sdf_>fp_margin+1"] = float((fv > fpm + 1).mean()) if fv.size else 0.0
        okc = ~g_ov
        ov = overlap_steps(p, fsize, ag["ag_fut"], ag["ag_yaw"], ag["ag_size"], ag["ag_mask"])[okc]
        ovc = overlap_steps(p, fsize, cvf, cvy, ag["ag_size"], cvm & ag["ag_mask"][:, :, :1])[okc]
        hit = ov.any((2, 3))
        r["coll_traj_gtok"] = float(hit.mean())
        r["coll_scene_gtok"] = float(hit.any(1).mean())
        first = np.where(ov.any(3).any(2), ov.any(3).argmax(2), -1)[hit]
        tsec = (first + 1) / hz
        for lo, hi in ((0, hor), (hor, 3.0), (3.0, 99)):
            r[f"coll_first_t_in_({lo:g},{hi:g}]s"] = float(((tsec > lo) & (tsec <= hi)).mean()) if tsec.size else 0.0
        cls = ag["ag_cls"][okc]
        for c, nm in CLS.items():
            byc = (ov & (cls == c)[:, None, None, :]).any((2, 3))
            r[f"coll_fail_involves_{nm}"] = float(byc[hit].mean()) if hit.any() else 0.0
        seen = ovc.any((2, 3))
        r["coll_fail_also_hits_cv_copy"] = float(seen[hit].mean()) if hit.any() else 0.0
        seen_h = ovc[:, :, : int(round(hor * hz))].any((2, 3))
        r["coll_fail_hits_cv_copy_within_horizon"] = float(seen_h[hit].mean()) if hit.any() else 0.0
        report[name] = r
        print(f"\n== {name}")
        for k2, v2 in r.items():
            print(f"  {k2:<48}{pct(v2)}")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
