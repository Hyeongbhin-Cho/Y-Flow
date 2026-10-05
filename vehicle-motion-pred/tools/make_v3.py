from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def patch(path, old, new):
    f = ROOT / path
    s = f.read_text()
    if new in s:
        print(f"{path}: already patched")
        return
    if old not in s:
        raise SystemExit(f"{path}: expected text not found:\n{old}")
    f.write_text(s.replace(old, new, 1))
    print(f"{path}: patched")


def main():
    src = ROOT / "configs/nuscenes_v2.yaml"
    cfg = yaml.safe_load(src.read_text())
    c = cfg["constraints"]
    print("v2:", {k: c.get(k) for k in ("fp_margin", "coll_margin", "coll_horizon_s", "footprint_tol", "hard")})
    c["fp_margin"] = 0.4
    c["coll_margin"] = 0.1
    c["coll_horizon_s"] = 1.5
    (ROOT / "configs/nuscenes_v3.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    print("v3:", {k: c.get(k) for k in ("fp_margin", "coll_margin", "coll_horizon_s", "footprint_tol", "hard")})
    print("wrote configs/nuscenes_v3.yaml")

    patch("vfm/tune.py",
          '    "constraints.fp_margin": ("offset", 0.0, 1.0),\n    "constraints.coll_margin": ("offset", 0.0, 0.5),\n',
          '    "constraints.fp_margin": ("uniform", 0.2, 0.5),\n    "constraints.coll_margin": ("uniform", 0.1, 0.3),\n'
          '    "constraints.coll_horizon_s": ("choice", [1.0, 1.5, 2.0]),\n')
    patch("vfm/evaluate.py",
          "        fp, gfp = [], []\n",
          "        fp, gfp, fpv = [], [], []\n")
    patch("vfm/evaluate.py",
          "            fp.append((cons.footprint_offroad(torch.from_numpy(pred[s:s + batch_size]).float().to(device), L, W) > tol).cpu().numpy())\n",
          "            v = cons.footprint_offroad(torch.from_numpy(pred[s:s + batch_size]).float().to(device), L, W).cpu().numpy()\n"
          "            fpv.append(v)\n"
          "            fp.append(v > tol)\n")
    patch("vfm/evaluate.py",
          "            out[f\"indep_offroad_{k}\"] = v\n",
          "            out[f\"indep_offroad_{k}\"] = v\n"
          "        gok = ~np.concatenate(gfp)[:, 0]\n"
          "        ex = np.clip(np.concatenate(fpv) - tol, 0.0, None)\n"
          "        out[\"indep_offroad_excess_gtok\"] = float(ex[gok].mean()) if gok.any() else float(\"nan\")\n")
    patch("tools/run_v2.py",
          '("indep_offroad_traj_gtok", "offroad traj %"), ("indep_offroad_scene_gtok", "offroad scene %")]',
          '("indep_offroad_traj_gtok", "offroad traj %"), ("indep_offroad_scene_gtok", "offroad scene %"),\n'
          '            ("indep_offroad_excess_gtok", "offroad excess m")]')
    patch("tools/run_v2.py",
          '"Y-Flow v2": run / "pred_yflow_val_seed0.npy"}',
          'f"Y-Flow {args.run}": run / "pred_yflow_val_seed0.npy"}')


if __name__ == "__main__":
    main()
