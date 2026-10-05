from pathlib import Path

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
    p = "vfm/evaluate.py"
    patch(p, 'VehicleTrajDataset(arrays, optional=method == "yflow")',
          'VehicleTrajDataset(arrays, optional=method in ("yflow", "proj"))')
    patch(p, "            else:\n                x = sample(model, batch, std, k, n_steps, gen)\n",
          "            elif method == \"proj\":\n"
          "                cons = NuScenesConstraint(batch, meta, ccfg)\n"
          "                x = sample(model, batch, std, k, n_steps, gen)\n"
          "                x = cons.project_feasible(x, sweeps=int(ccfg.get(\"proj_sweeps_final\", 50)))\n"
          "            else:\n                x = sample(model, batch, std, k, n_steps, gen)\n")
    patch(p, '{"fm": "flow_matching", "yflow": "yflow"}[method]',
          '{"fm": "flow_matching", "yflow": "yflow", "proj": "fm_posthoc_projection"}[method]')
    patch(p, 'choices=["fm", "yflow", "cv", "gt"]', 'choices=["fm", "yflow", "proj", "cv", "gt"]')


if __name__ == "__main__":
    main()
