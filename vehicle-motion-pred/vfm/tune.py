from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from vfm.config import Cfg, dump_config, load_config
from vfm.evaluate import run

SPACE = {
    "yflow.t_on": ("uniform", 0.3, 0.8),
    "yflow.lambda_oc": ("log", 1.0, 100.0),
    "yflow.mu": ("choice", [0.0, 0.3, 1.0, 3.0]),
    "yflow.delta": ("choice", [0.05, 0.1, 0.3, 1e9]),
    "yflow.gamma_max": ("uniform", 0.5, 1.0),
    "yflow.max_iter": ("choice", [5, 10, 20]),
    "constraints.w_lane": ("choice", [0.0, 0.05, 0.1, 0.3]),
    "constraints.lane_half_width": ("uniform", 1.5, 4.0),
    "constraints.drivable_margin": ("uniform", 1.5, 2.5),
    "constraints.fp_margin": ("uniform", 0.2, 0.5),
    "constraints.coll_margin": ("uniform", 0.1, 0.3),
    "constraints.coll_horizon_s": ("choice", [1.0, 1.5, 2.0]),
}


def _get(cfg, key: str):
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def draw(rng: np.random.Generator, cfg=None) -> dict:
    out = {}
    for k, spec in SPACE.items():
        if spec[0] == "offset":
            base = _get(cfg, k) if cfg is not None else None
            if base is None:
                continue
            out[k] = float(base) + float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "uniform":
            out[k] = float(rng.uniform(spec[1], spec[2]))
        elif spec[0] == "log":
            out[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        else:
            v = spec[1][int(rng.integers(len(spec[1])))]
            out[k] = v
    return out


def apply(cfg: Cfg, params: dict) -> Cfg:
    c = copy.deepcopy(cfg)
    for k, v in params.items():
        node = c
        parts = k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, Cfg())
        node[parts[-1]] = v
    return c


def objective(m: dict, k: int, w_safe: float, w_mr: float, w_coll: float = 0.0, w_off: float = 0.0) -> float:
    thr = [kk for kk in m if kk.startswith(f"MR_{k}@")]
    v = m[f"minADE_{k}"] + w_safe * (1.0 - m.get("all_hard_safe", 0.0)) + w_mr * (m[thr[0]] if thr else 0.0)
    v += w_coll * m.get("indep_coll_scene_gtok", 0.0) + w_off * m.get("indep_offroad_scene_gtok", 0.0)
    return float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w_safe", type=float, default=5.0)
    ap.add_argument("--w_mr", type=float, default=1.0)
    ap.add_argument("--w_coll", type=float, default=2.0)
    ap.add_argument("--w_off", type=float, default=2.0)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg.data.limit_eval = int(cfg.data.get("limit_tune", 1000))
    cfg.eval.save_predictions = False
    k = int(cfg.eval.get("k", 10))
    out = Path(cfg.out_dir) / str(cfg.run_name) / "tune"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    run_dir = Path(cfg.out_dir) / str(cfg.run_name)
    ckpt = str(run_dir / "best.pt" if (run_dir / "best.pt").exists() else run_dir / "last.pt")
    base = run(cfg, "fm", ckpt_path=ckpt)
    rows = [{"trial": -1, "params": {}, "objective": objective(base, k, args.w_safe, args.w_mr, args.w_coll, args.w_off), "metrics": base}]
    best = None
    for t in range(args.trials):
        params = draw(rng, cfg)
        c = apply(cfg, params)
        c.run_name = f"{cfg.run_name}/tune/trial_{t:03d}"
        try:
            m = run(c, "yflow", ckpt_path=ckpt)
        except Exception as exc:
            rows.append({"trial": t, "params": params, "error": repr(exc)})
            continue
        obj = objective(m, k, args.w_safe, args.w_mr, args.w_coll, args.w_off)
        rows.append({"trial": t, "params": params, "objective": obj, "metrics": m})
        if best is None or obj < best[0]:
            best = (obj, params)
            dump_config(apply(cfg, params), out / "best.yaml")
        (out / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        print(f"trial {t}: obj={obj:.4f} minADE={m[f'minADE_{k}']:.3f} safe={m.get('all_hard_safe', 0):.3f} "
              f"coll_gtok={m.get('indep_coll_scene_gtok', float('nan')):.3f} "
              f"offroad_gtok={m.get('indep_offroad_scene_gtok', float('nan')):.3f} best={best[0]:.4f}", flush=True)
    (out / "trials.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(json.dumps({"baseline_objective": rows[0]["objective"], "best": best}, indent=2))


if __name__ == "__main__":
    main()
