import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _stubs

from flow_drive.utils.train_utils import load_params
from flow_drive.utils.normalizer import StateNormalizer
from flow_drive.yflow.sampler import YFlowConfig, YFlowContext, YFlowStats, project_target

ROOT = os.path.join(os.path.dirname(__file__), "..")
params = load_params(os.path.join(ROOT, "flow_drive/config/config.yaml"))
norm = StateNormalizer.from_json(params.data_processing)
T = params.diffuser.pred_horizon
DT = 0.1
tt = torch.arange(1, T + 1, dtype=torch.float64) * DT

cases = {}
v = 5.0;  cases["accel +4"] = (torch.stack([v * tt + 2.0 * tt ** 2, 0 * tt], -1), (v, 0.0))
v = 10.0; tb = torch.clamp(tt, max=v / 5.0)
cases["brake -5"] = (torch.stack([v * tb - 2.5 * tb ** 2, 0 * tt], -1), (v, 0.0))
v, R = 8.0, 15.0; th = v * tt / R
cases["turn 4.3 lat"] = (torch.stack([R * torch.sin(th), R * (1 - torch.cos(th))], -1), (v, 0.0))
v, R = 8.0, 12.0; th = v * tt / R
cases["turn 5.3 lat"] = (torch.stack([R * torch.sin(th), R * (1 - torch.cos(th))], -1), (v, 0.0))


def lonlat(pos, v0):
    B = pos.shape[0]
    full = torch.cat([(-v0 * DT)[:, None], torch.zeros(B, 1, 2, dtype=pos.dtype), pos], 1)
    a = (full[:, 2:] - 2 * full[:, 1:-1] + full[:, :-2]) / DT ** 2
    d = full[:, 2:] - full[:, :-2]
    h = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    n = torch.stack([-h[..., 1], h[..., 0]], -1)
    return (a * h).sum(-1), (a * n).sum(-1)


def run(mode):
    pos = torch.stack([c[0] for c in cases.values()])
    v0 = torch.tensor([c[1] for c in cases.values()], dtype=torch.float64)
    d = torch.diff(torch.cat([torch.zeros(len(pos), 1, 2, dtype=torch.float64), pos], 1), dim=1)
    cs = torch.nn.functional.normalize(d, dim=-1)
    x1 = norm(torch.cat([pos, cs], -1).float())
    cfg = YFlowConfig(use_corridor=False, accel_mode=mode)
    ctx = YFlowContext(v0=v0.float(), v_limit=torch.full((len(pos),), 30.0))
    st = YFlowStats()
    out = norm.inverse(project_target(x1, norm, cfg, ctx, 1.0, True, st)).double()[..., :2]
    lon, lat = lonlat(out, v0)
    shift = (out - pos).norm(dim=-1).mean(1)
    return lon, lat, shift, st


res = {}
for mode in ("ball", "lonlat"):
    lon, lat, shift, st = run(mode)
    res[mode] = (lon, lat, shift)
    print(f"[{mode:6}] levels={st.levels} viol={ {k: round(v, 3) for k, v in st.viol.items()} }")
    for i, name in enumerate(cases):
        print(f"    {name:13} lon [{lon[i].min():+.2f}, {lon[i].max():+.2f}]  |lat| max {lat[i].abs().max():.2f}"
              f"  |a| max {torch.hypot(lon[i], lat[i]).max():.2f}  mean shift {shift[i]:.3f} m")

tol = 0.3
lon, lat, shift = res["lonlat"]
assert (lon.max() <= 2.2 + tol) and (lon.min() >= -3.8 - tol), "kin2 lon bound"
assert lat.abs().max() <= 4.5 + tol, "kin2 lat bound"
i = list(cases).index("turn 4.3 lat")
assert res["lonlat"][2][i] < 0.5 * res["ball"][2][i], "kin2 should leave a 4.3 m/s² turn nearly untouched"
lon_b, lat_b, _ = res["ball"]
assert torch.hypot(lon_b, lat_b).max() <= 4.0 + tol, "ball bound (default mode unchanged)"
print("[lonlat] PASS")
