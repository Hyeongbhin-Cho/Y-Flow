import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import _stubs

from flow_drive.utils.post_processing import bound_speed_and_acceleration, smooth_trajectories_preset
from flow_drive.yflow.posthoc import posthoc_fix

T, dt = 80, 0.1
tt = torch.arange(1, T + 1, dtype=torch.float32) * dt
raw = torch.stack([torch.stack([12.0 * tt, 0.3 * torch.sin(3 * tt), torch.zeros(T)], -1),
                   torch.stack([10.0 * tt - 2.0 * tt ** 2, torch.zeros(T), torch.zeros(T)], -1)])
ego = torch.tensor([[0.0, 0.0, 1.0, 0.0, 10.0, 0.0], [0.0, 0.0, 1.0, 0.0, 10.0, 0.0]])
lim = 8.0


def speed(a):
    full = torch.cat([torch.zeros(a.shape[0], 1, 2), a[..., :2]], 1)
    return (full[:, 1:] - full[:, :-1]).norm(dim=-1) / dt


for smooth in (False, True):
    out = posthoc_fix(raw, ego, lim, smooth=smooth)
    assert out.shape == raw.shape and torch.isfinite(out).all()
    v = speed(out)
    print(f"[posthoc smooth={smooth}] raw max speed {speed(raw).max():.2f} -> {v.max():.2f} m/s "
          f"(limit {lim}, starts at 10 so first steps decelerate), final speed {v[:, -1].tolist()}")
    assert v[:, -1].max() <= lim + 1e-3

cur = torch.stack([ego[:, 0], ego[:, 1], torch.atan2(ego[:, 3], ego[:, 2])], -1)
ref = smooth_trajectories_preset(torch.cat([cur[None, :, None, :], raw[None]], 2), preset="strong")[:, :, 1:, :]
ref = bound_speed_and_acceleration(ref, ego, torch.full((2,), lim))[0]
assert (posthoc_fix(raw, ego, lim, smooth=True) - ref).abs().max() < 1e-6
print("[posthoc] PASS")
