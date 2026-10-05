import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _stubs

from flow_drive.utils.train_utils import load_params, get_noise_scheduler, load_checkpoint_directly
from flow_drive.utils.normalizer import StateNormalizer
from flow_drive.yflow import constraints as C
from flow_drive.yflow.sampler import YFlowConfig, YFlowContext, sample_action_yflow, _sample_once

ROOT = os.path.join(os.path.dirname(__file__), "..")
params = load_params(os.path.join(ROOT, "flow_drive/config/config.yaml"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, net = load_checkpoint_directly(params, os.path.join(ROOT, "flow_drive/checkpoint/flow_drive_model.pth"), "cpu")
net = net.to(dev).eval()
norm = StateNormalizer.from_json(params.data_processing)


def lane(y0):
    xs = torch.arange(-5.0, 80.0, 1.0, dtype=torch.float64)
    n = len(xs)
    return C.CorridorData(center=torch.stack([xs, torch.full_like(xs, y0)], -1),
                          normal=torch.tensor([[0.0, 1.0]], dtype=torch.float64).repeat(n, 1),
                          tangent=torch.tensor([[1.0, 0.0]], dtype=torch.float64).repeat(n, 1),
                          lo=torch.full((n,), -1.85, dtype=torch.float64),
                          hi=torch.full((n,), 1.85, dtype=torch.float64),
                          valid=torch.ones(n, dtype=torch.bool))


ok_lane, bad_lane = lane(0.0), lane(3.0)
m = YFlowConfig().corridor_margin

assert C.corridor_feasible_now(ok_lane, m) is True
assert C.corridor_feasible_now(bad_lane, m) is False
print("[precheck] inside -> True, 3 m outside -> False")

B = 3
g = torch.Generator().manual_seed(2)
obs = {"encoding": torch.randn(B, 60, params.diffuser.hidden_dim, generator=g).to(dev),
       "mask": torch.zeros(B, 60, dtype=torch.bool).to(dev)}
ego = norm(torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(B, 1)).to(dev)
ctx_bad = YFlowContext(v0=torch.tensor([[6.0, 0.0]]).repeat(B, 1), v_limit=torch.full((B,), 8.0),
                       corridor=bad_lane)


def sample(cfg, ctx, seed=0):
    torch.manual_seed(seed)
    return sample_action_yflow(params, net, obs, get_noise_scheduler(params), ego, norm, cfg, ctx)


ref_kin, st_kin = sample(YFlowConfig(use_corridor=False), ctx_bad)
assert all(l == "kin" for l in st_kin.levels), st_kin.levels

v1, st_v1 = sample(YFlowConfig(), ctx_bad)
print(f"[v1] levels={st_v1.levels}  max|v1 - kin-only| = {(v1 - ref_kin).abs().max():.3f} (normalised units)")

out, st = sample(YFlowConfig(fallback_resample=True), ctx_bad)
print(f"[resample] levels={st.levels} resampled={st.resampled} solve={st.solve_ms:.0f}ms")
assert st.resampled >= 1
for b, lv in enumerate(st.levels):
    if lv.endswith("(rs)"):
        d = (out[b] - ref_kin[b]).abs().max().item()
        assert d < 1e-5, (b, d)
    else:
        assert lv == "kin+corr", lv
assert any(lv.endswith("(rs)") for lv in st.levels)
print("[resample] failed samples == kin-only flow from the same noise")

out3, st3 = sample(YFlowConfig(corridor_precheck=True, fallback_resample=True), ctx_bad)
assert st3.corr_skipped and st3.resampled == 0 and all(l == "kin" for l in st3.levels), st3.as_dict()
assert (out3 - ref_kin).abs().max().item() < 1e-5
ctx_ok = YFlowContext(v0=ctx_bad.v0, v_limit=ctx_bad.v_limit, corridor=ok_lane)
_, st4 = sample(YFlowConfig(corridor_precheck=True), ctx_ok)
assert not st4.corr_skipped
print("[precheck] infeasible corridor skipped up front (== kin-only), feasible one kept")

cfg = YFlowConfig()
torch.manual_seed(5)
a, _ = sample_action_yflow(params, net, obs, get_noise_scheduler(params), ego, norm, cfg, ctx_ok)
torch.manual_seed(5)
x0 = torch.randn((B, params.diffuser.pred_horizon, 4), device=dev)
b_, _ = _sample_once(params, net, obs, get_noise_scheduler(params), ego, norm, cfg, ctx_ok, x0, None, None)
assert (a - b_).abs().max().item() == 0.0
print("[fallback] PASS")
