import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _stubs

from flow_drive.utils.train_utils import load_params, get_noise_scheduler, load_checkpoint_directly
from flow_drive.utils.normalizer import StateNormalizer
from flow_drive.utils.infer_utils import sample_action, sample_action_with_speed_and_lateral_offsets
from flow_drive.utils.infer_utils import _apply_speed_and_lateral_adjustments
from flow_drive.yflow.sampler import YFlowConfig, sample_action_yflow

ROOT = os.path.join(os.path.dirname(__file__), "..")
params = load_params(os.path.join(ROOT, "flow_drive/config/config.yaml"))
ckpt = os.path.join(ROOT, "flow_drive/checkpoint/flow_drive_model.pth")
dev = "cuda" if torch.cuda.is_available() else "cpu"
_, net = load_checkpoint_directly(params, ckpt, device="cpu")
net = net.to(dev).eval()
norm = StateNormalizer.from_json(params.data_processing)

B, N, Dh = 3, 60, params.diffuser.hidden_dim
g = torch.Generator().manual_seed(0)
obs = {"encoding": torch.randn(B, N, Dh, generator=g).to(dev),
       "mask": torch.zeros(B, N, dtype=torch.bool).to(dev)}
ego = norm(torch.tensor([[0.0, 0.0, 1.0, 0.0]]).repeat(B, 1)).to(dev)

torch.manual_seed(123)
ref, _ = sample_action(params, net, obs, get_noise_scheduler(params), ego)
torch.manual_seed(123)
out, st = sample_action_yflow(params, net, obs, get_noise_scheduler(params), ego, norm,
                              YFlowConfig(enabled=False), ctx=None)
d1 = (ref - out).abs().max().item()
print(f"[equiv] plain sampler      max|Δ| = {d1:.3e}")

speed_offsets, lateral_offsets = [1.0, 1.0], [0.0, 6.5 / 20, -6.5 / 20]
S = len(speed_offsets) * len(lateral_offsets)
ego6 = torch.cat([ego, torch.zeros(B, 2, device=dev)], -1)
torch.manual_seed(7)
ref2, _ = sample_action_with_speed_and_lateral_offsets(params, net, obs, get_noise_scheduler(params), ego6,
                                                       speed_offsets, lateral_offsets)
obs_e = {"encoding": obs["encoding"][None].expand(S, -1, -1, -1).reshape(S * B, N, Dh),
         "mask": obs["mask"][None].expand(S, -1, -1).reshape(S * B, N)}
ego_e = ego[None].expand(S, -1, -1).reshape(S * B, 4)
spd = torch.zeros(B, device=dev)
hook = lambda x: _apply_speed_and_lateral_adjustments(x, ego_e, speed_offsets, lateral_offsets, B, S,
                                                      params.diffuser.pred_horizon, spd)
torch.manual_seed(7)
out2, _ = sample_action_yflow(params, net, obs_e, get_noise_scheduler(params), ego_e, norm,
                              YFlowConfig(enabled=False), ctx=None, step_hook=hook,
                              hook_index=params.inference.flow_inference_iter // 2 - 1)
d2 = (ref2 - out2).abs().max().item()
print(f"[equiv] offset sampler     max|Δ| = {d2:.3e}")

assert d1 < 1e-4 and d2 < 1e-4, "Y-Flow loop with γ=0 does not reproduce FlowDrive!"
print("[equiv] PASS")
