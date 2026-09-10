"""Small policy-level checks for the factorial design."""

import torch

from experiments.pov_projection_phase2.sampling import corrected_pov_euler_step


def test_terminal_l1_residual_equals_full_replacement():
    x = torch.tensor([[[1.0, 2.0]]])
    v = torch.tensor([[[0.3, -0.2]]])
    z = torch.tensor([[[4.0, 5.0]]])
    t = 6.0 / 7.0
    dt = 1.0 / 7.0
    out, _, _, _ = corrected_pov_euler_step(x, v, z, t, dt, 1.0)
    assert torch.allclose(out, z, atol=1e-6, rtol=0.0)


def test_terminal_l05_is_interpolation_not_replacement():
    x = torch.tensor([[[1.0, 2.0]]])
    v = torch.tensor([[[0.3, -0.2]]])
    z = torch.tensor([[[4.0, 5.0]]])
    out, _, _, _ = corrected_pov_euler_step(
        x, v, z, 6.0 / 7.0, 1.0 / 7.0, 0.5
    )
    assert not torch.allclose(out, z)
