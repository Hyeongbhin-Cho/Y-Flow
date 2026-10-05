import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _stubs

import shapely
from flow_drive.utils.train_utils import load_params, get_noise_scheduler, load_checkpoint_directly
from flow_drive.utils.normalizer import StateNormalizer
from flow_drive.yflow import constraints as C
from flow_drive.yflow.sampler import YFlowConfig, YFlowContext, YFlowStats, project_target, sample_action_yflow
from flow_drive.yflow.smoothing import smooth_positions

ROOT = os.path.join(os.path.dirname(__file__), "..")
params = load_params(os.path.join(ROOT, "flow_drive/config/config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, net = load_checkpoint_directly(params, os.path.join(ROOT, "flow_drive/checkpoint/flow_drive_model.pth"), "cpu")
net = net.to(dev).eval()
norm = StateNormalizer.from_json(params.data_processing)
T = params.diffuser.pred_horizon

x, y = torch.randn(64, T, 2, dtype=torch.float64), torch.randn(64, T, 2, dtype=torch.float64)
v0 = torch.randn(64, 2, dtype=torch.float64)
ratio = ((smooth_positions(x, v0, 0.1, 2.0) - smooth_positions(y, v0, 0.1, 2.0)).flatten(1).norm(dim=1)
         / (x - y).flatten(1).norm(dim=1)).max().item()
print(f"[P] max ||P(x)-P(y)|| / ||x-y|| = {ratio:.4f} (must be <= 1)")
assert ratio <= 1 + 1e-9

xs = torch.arange(-5.0, 80.0, 1.0, dtype=torch.float64)
corr = C.CorridorData(center=torch.stack([xs, torch.zeros_like(xs)], -1),
                      normal=torch.tensor([[0.0, 1.0]], dtype=torch.float64).repeat(len(xs), 1),
                      tangent=torch.tensor([[1.0, 0.0]], dtype=torch.float64).repeat(len(xs), 1),
                      lo=torch.full((len(xs),), -1.85, dtype=torch.float64),
                      hi=torch.full((len(xs),), 1.85, dtype=torch.float64),
                      valid=torch.ones(len(xs), dtype=torch.bool))
box = shapely.box(20.0, -1.0, 24.5, 1.0)
obstacles = [np.array([box], dtype=object) for _ in range(T)]

tt = torch.arange(1, T + 1, dtype=torch.float64) * 0.1
raws = [
    torch.stack([9.5 * tt, 0 * tt], -1),
    torch.stack([6.0 * tt, 1.6 * torch.sin(2.0 * tt)], -1),
    torch.stack([6.0 * tt + 0.3 * tt ** 2, 0 * tt], -1),
    torch.stack([6.0 * tt, 0.25 * tt ** 2], -1),
]
pos_raw = torch.stack(raws)
B = pos_raw.shape[0]
d = torch.diff(torch.cat([torch.zeros(B, 1, 2, dtype=torch.float64), pos_raw], 1), dim=1)
cs_raw = torch.nn.functional.normalize(d, dim=-1)
x1_norm = norm(torch.cat([pos_raw, cs_raw], -1).float())
v0 = torch.tensor([[6.0, 0.0]]).repeat(B, 1)

for use_obs in (False, True):
    ctx = YFlowContext(v0=v0, v_limit=torch.full((B,), 8.0), corridor=corr,
                       obstacles=obstacles if use_obs else None)
    cfg = YFlowConfig(use_obstacles=use_obs)
    st = YFlowStats()
    out = project_target(x1_norm, norm, cfg, ctx, gamma=1.0, final=True, stats=st)
    phys = norm.inverse(out).double()
    pos = phys[..., :2]
    full = torch.cat([torch.zeros(B, 1, 2, dtype=pos.dtype), pos], 1)
    speed = (full[:, 1:] - full[:, :-1]).norm(dim=-1) / 0.1
    print(f"[project obs={use_obs}] level={st.level} viol={ {k: round(v, 4) for k, v in st.viol.items()} } "
          f"shift={st.shift_m:.2f}m {st.solve_ms:.0f}ms | max speed {speed.max():.2f} m/s, max |y| {pos[..., 1].abs().max():.2f} m, "
          f"min x-gap to car {(20.0 - pos[2, :, 0].max()).item():.2f} m (rear axle)")
    want = "kin+corr+obs" if use_obs else "kin+corr"
    assert all(l == want for l in st.levels), st.levels
    assert st.viol["speed"] < 0.1 and st.viol["accel"] < 0.3 and st.viol["corridor"] < 0.05, st.viol
    if use_obs:
        assert st.viol["obstacle"] < 0.05, st.viol
        hs = torch.nn.functional.normalize(phys[..., 2:4], dim=-1)
        worst = min(shapely.distance(box, shapely.points((pos[b, k] + o * hs[b, k]).numpy()))
                    for b in range(B) for k in range(T) for o in C.CIRCLE_OFFSETS)
        print(f"   min body-circle clearance to the car: {worst - C.CIRCLE_RADIUS:.2f} m (target >= 0.2)")
        assert worst >= C.CIRCLE_RADIUS + 0.1

Bs = 2
g = torch.Generator().manual_seed(1)
obs_cond = {"encoding": torch.randn(Bs, 60, params.diffuser.hidden_dim, generator=g).to(dev),
            "mask": torch.zeros(Bs, 60, dtype=torch.bool).to(dev)}
ego = norm(torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(Bs, 1)).to(dev)
ctx = YFlowContext(v0=torch.tensor([[6.0, 0.0]]).repeat(Bs, 1), v_limit=torch.full((Bs,), 8.0), corridor=corr)
torch.manual_seed(0)
out, st = sample_action_yflow(params, net, obs_cond, get_noise_scheduler(params), ego, norm, YFlowConfig(), ctx)
assert torch.isfinite(out).all() and st.active_steps >= 1
print(f"[sampler] runs: active_steps={st.active_steps} level={st.level} solve={st.solve_ms:.0f}ms")
print("[constraints] PASS")
