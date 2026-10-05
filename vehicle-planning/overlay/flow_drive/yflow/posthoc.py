import torch

from flow_drive.utils.post_processing import bound_speed_and_acceleration, smooth_trajectories_preset


def posthoc_fix(actions: torch.Tensor, ego_raw: torch.Tensor, speed_limit_mps: float, smooth: bool) -> torch.Tensor:
    a = actions.unsqueeze(0)
    if smooth:
        cur = torch.stack([ego_raw[:, 0], ego_raw[:, 1], torch.atan2(ego_raw[:, 3], ego_raw[:, 2])], -1)
        with_cur = torch.cat([cur[None, :, None, :].to(a.dtype), a], 2)
        a = smooth_trajectories_preset(with_cur, preset="strong")[:, :, 1:, :]
    lim = torch.full((a.shape[1],), float(speed_limit_mps), device=a.device, dtype=a.dtype)
    a = bound_speed_and_acceleration(a.clone(), ego_raw, lim)
    return a[0]
