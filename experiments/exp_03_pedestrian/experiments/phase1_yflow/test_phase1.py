# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/test_phase1.py
"""CPU-only integrity tests for the Phase-1 projection and update rules (numpy)."""

import numpy as np
from scipy.optimize import minimize

from experiments.phase1_yflow.constraints import CollisionKinematicConstraint, KinematicConstraint, KinematicLimits
from experiments.phase1_yflow.sampling import (
    damped_target,
    is_terminal_step,
    yflow_next_state,
    yflow_projection_active,
)


F = 12
LIM = KinematicLimits(v_max=1.5, a_max=1.0, dt=0.4)


def _random_case(rng, n=6, scale=2.0):
    x = rng.normal(size=(n, F, 2)) * scale
    p0 = np.zeros((n, 2))
    v0 = rng.normal(size=(n, 2)) * 0.5
    p_prev = p0 - v0 * LIM.dt
    return x, p0, p_prev


def test_feasible_points_are_fixed():
    c = KinematicConstraint(LIM, F)
    t = np.arange(1, F + 1)[:, None] * LIM.dt * 0.8 * np.array([1.0, 0.0])
    x = np.repeat(t[None], 4, axis=0)
    p0 = np.zeros((4, 2))
    p_prev = p0 - np.array([0.8, 0.0]) * LIM.dt
    assert float(c.max_violation(x, p0, p_prev).max()) <= 0.0
    proj, _, _, _ = c.project(x, p0, p_prev)
    assert np.allclose(proj, x, atol=1e-6)


def test_projection_is_feasible_and_idempotent():
    rng = np.random.default_rng(1)
    c = KinematicConstraint(LIM, F)
    x, p0, p_prev = _random_case(rng)
    assert float(c.max_violation(x, p0, p_prev).max()) > 0.0
    p_prev, _ = c.clip_anchor(p0, p_prev)
    proj, _, _, _ = c.project(x, p0, p_prev, n_iters=600, tol=1e-9)
    assert float(c.max_violation(proj, p0, p_prev).max()) <= 1e-5
    again, _, _, _ = c.project(proj, p0, p_prev, n_iters=600, tol=1e-9)
    assert np.allclose(again, proj, atol=1e-4)
    sealed = c.project_feasible(proj, p0, p_prev)
    assert float(c.max_violation(sealed, p0, p_prev).max()) <= 1e-9
    assert np.abs(sealed - proj).max() < 1e-3


def test_projection_matches_slsqp():
    rng = np.random.default_rng(2)
    c = KinematicConstraint(LIM, F)
    x, p0, p_prev = _random_case(rng, n=3, scale=1.0)
    p_prev, _ = c.clip_anchor(p0, p_prev)
    proj, _, _, _ = c.project(x, p0, p_prev, n_iters=800, tol=1e-10)
    for i in range(x.shape[0]):
        target = x[i]

        def obj(z):
            return 0.5 * float(((z.reshape(F, 2) - target) ** 2).sum())

        def cons(z):
            hv = c.h(z.reshape(1, F, 2), p0[i : i + 1], p_prev[i : i + 1])
            return -np.concatenate([hv["speed"][0], hv["acc"][0]])

        res = minimize(obj, proj[i].ravel() + 1e-3, method="SLSQP",
                       constraints={"type": "ineq", "fun": cons},
                       options={"maxiter": 500, "ftol": 1e-12})
        assert res.success, res.message
        assert obj(res.x) >= obj(proj[i].ravel()) - 1e-5
        assert np.allclose(res.x.reshape(F, 2), proj[i], atol=5e-3)


def test_speed_only_straight_line_scales_exactly():
    c = KinematicConstraint(KinematicLimits(v_max=1.0, a_max=1e9, dt=1.0), F)
    step = np.array([3.0, 0.0])
    x = np.cumsum(np.repeat(step[None], F, axis=0), axis=0)[None]
    p0 = np.zeros((1, 2))
    p_prev = p0 - step
    proj, _, _, _ = c.project(x, p0, p_prev, n_iters=800, tol=1e-10)
    expected = np.cumsum(np.repeat(np.array([[1.0, 0.0]]), F, axis=0), axis=0)[None]
    assert np.allclose(proj, expected, atol=1e-6)


def test_anchor_clip_and_greedy_feasibility():
    rng = np.random.default_rng(3)
    c = KinematicConstraint(LIM, F)
    x, p0, _ = _random_case(rng, n=50, scale=3.0)
    p_prev = p0 - rng.normal(size=(50, 2)) * 2.0
    p_prev_c, clipped = c.clip_anchor(p0, p_prev)
    assert clipped.any()
    assert (np.linalg.norm(p0 - p_prev_c, axis=-1) <= LIM.speed_radius + 1e-9).all()
    sealed = c.project_feasible(x, p0, p_prev_c)
    assert float(c.max_violation(sealed, p0, p_prev_c).max()) <= 1e-9


def test_warm_start_reaches_same_point():
    rng = np.random.default_rng(4)
    c = KinematicConstraint(LIM, F)
    x, p0, p_prev = _random_case(rng, n=4, scale=1.0)
    p_prev, _ = c.clip_anchor(p0, p_prev)
    cold, _, _, state = c.project(x, p0, p_prev, n_iters=800, tol=1e-10)
    x2 = x + 1e-3
    warm, it_warm, _, _ = c.project(x2, p0, p_prev, n_iters=800, tol=1e-10, warm=state)
    cold2, it_cold, _, _ = c.project(x2, p0, p_prev, n_iters=800, tol=1e-10)
    assert np.allclose(warm, cold2, atol=1e-4)
    assert it_warm < it_cold


def test_seal_buffer_survives_float32():
    rng = np.random.default_rng(5)
    scale = 0.1054
    c = KinematicConstraint(KinematicLimits(LIM.v_max * scale, LIM.a_max * scale, LIM.dt), F)
    cm = KinematicConstraint(LIM, F)
    x, p0, p_prev = _random_case(rng, n=400, scale=1.0)
    x, p0, p_prev = x * scale, p0 * scale, p_prev * scale
    p_prev, _ = c.clip_anchor(p0, p_prev)
    sealed = c.project_feasible(x, p0, p_prev, buffer=1e-4).astype(np.float32) / np.float32(scale)
    pm0 = (p0 / scale).astype(np.float32)
    pmp = (p_prev / scale).astype(np.float32)
    assert float(cm.max_violation(sealed, pm0, pmp).max()) <= 0.0


def _crossing_case(rng, B=60, M=3):
    t = np.arange(1, F + 1) * LIM.dt
    x = np.stack([1.2 * t, 0 * t], -1)[None].repeat(B, 0) + rng.normal(size=(B, F, 2)) * 0.03
    p0 = np.zeros((B, 2))
    p_prev = p0 - np.array([1.2 * LIM.dt, 0.0])
    q = np.zeros((B, M, F, 2))
    for j in range(M):
        start = rng.uniform(1, 5, size=(B, 1))
        vy = rng.uniform(-1.2, 1.2, size=(B, 1))
        y0 = -vy * rng.uniform(2, 4, size=(B, 1))
        q[:, j, :, 0] = start + 0 * t
        q[:, j, :, 1] = y0 + vy * t
    mask = np.ones((B, M), bool)
    mask[:, -1] = False
    return x, p0, p_prev, q, mask


def test_collision_projection_clears_and_matches_slsqp():
    rng = np.random.default_rng(6)
    lim = KinematicLimits(v_max=1.85, a_max=2.38, dt=0.4)
    c = CollisionKinematicConstraint(lim, r_safe=0.2, n_future=F)
    x, p0, p_prev, q, mask = _crossing_case(rng)
    p_prev, _ = c.clip_anchor(p0, p_prev)
    assert (c.min_clearance(x, q, mask) < 0.2).any()
    n = c.normals(x, q)
    proj, _, res, _ = c.project_mixed(x, p0, p_prev, q, mask, n, n_iters=800, tol=1e-9)
    ok = res < 1e-6
    assert ok.mean() > 0.8
    assert (c.min_clearance(proj, q, mask)[ok] >= 0.2 - 1e-5).all()
    assert float(c.max_violation(proj[ok], p0[ok], p_prev[ok]).max()) <= 1e-5
    masked_d = np.linalg.norm(proj[:, None] - q, axis=-1)[:, -1]
    assert np.isfinite(masked_d).all()
    for i in np.flatnonzero(ok)[:2]:
        def obj(z):
            return 0.5 * float(((z.reshape(F, 2) - x[i]) ** 2).sum())

        def cons(z):
            zz = z.reshape(1, F, 2)
            hv = c.h(zz, p0[i : i + 1], p_prev[i : i + 1])
            hs = ((zz[0][None] - q[i]) * n[i]).sum(-1) - 0.2
            return np.concatenate([-hv["speed"][0], -hv["acc"][0], hs[mask[i]].ravel()])

        r = minimize(obj, proj[i].ravel(), method="SLSQP", constraints={"type": "ineq", "fun": cons},
                     options={"maxiter": 500, "ftol": 1e-12})
        assert r.success, r.message
        assert np.allclose(r.x.reshape(F, 2), proj[i], atol=1e-3)


def test_collision_rows_inactive_when_masked():
    rng = np.random.default_rng(7)
    lim = KinematicLimits(v_max=1.85, a_max=2.38, dt=0.4)
    c = CollisionKinematicConstraint(lim, r_safe=0.2, n_future=F)
    kc = KinematicConstraint(lim, F)
    x, p0, p_prev, q, _ = _crossing_case(rng, B=10)
    p_prev, _ = c.clip_anchor(p0, p_prev)
    mask = np.zeros(q.shape[:2], bool)
    a, _, _, _ = c.project_mixed(x, p0, p_prev, q, mask, c.normals(x, q), n_iters=800, tol=1e-10)
    b, _, _, _ = kc.project(x, p0, p_prev, n_iters=800, tol=1e-10)
    assert np.allclose(a, b, atol=1e-4)


def test_neighbour_frame_matches_loader_rotation():
    from experiments._layout import DATASET_ROOT
    from experiments.phase1_yflow.neighbors import NeighbourTable
    table = NeighbourTable(DATASET_ROOT / "zara1" / "zara1_test.pkl", rotate=True, rotate_time_frame=6)
    back = table.traj[:, 6] - table.traj[:, 7]
    rot = np.einsum("nij,nj->ni", table.R, back)
    assert np.abs(rot[:, 1]).max() < 1e-4
    assert (rot[:, 0] >= 0).all()
    idx = np.arange(20)
    own = table.own_future(idx)
    manual = np.einsum("nij,nfj->nfi", table.R[idx], table.traj[idx, 8:] - table.traj[idx, 7:8])
    assert np.allclose(own, manual)
    q, mask = table.gather(idx)
    assert q.shape[0] == 20 and mask.shape == q.shape[:2]


def test_report_block_bootstrap_brackets_mean():
    from experiments.phase2_cfm.report_study import block_bootstrap
    rng = np.random.default_rng(0)
    d = rng.normal(-0.01, 0.005, size=600)
    lo, hi = block_bootstrap(d, rng)
    assert lo < d.mean() < hi
    assert hi < 0


def test_neighbour_cv_source():
    from experiments._layout import DATASET_ROOT
    from experiments.phase1_yflow.neighbors import NeighbourTable
    table = NeighbourTable(DATASET_ROOT / "zara2" / "zara2_test.pkl", rotate=True, rotate_time_frame=6)
    j = 5
    cv = table.future_of(j, "cv")
    step = table.traj[j, 7] - table.traj[j, 6]
    assert np.allclose(cv[0], table.traj[j, 7] + step)
    assert np.allclose(np.diff(cv, axis=0), step)
    q_gt, m_gt = table.gather(np.arange(10), source="gt")
    q_cv, m_cv = table.gather(np.arange(10), source="cv")
    assert q_gt.shape == q_cv.shape and np.array_equal(m_gt, m_cv)


def test_update_rules():
    assert is_terminal_step(0.9, 0.1)
    assert not is_terminal_step(0.5, 0.1)
    assert yflow_projection_active(0.2, 0.1, t_on=0.5) is False
    assert yflow_projection_active(0.5, 0.1, t_on=0.5) is True
    assert yflow_projection_active(0.9, 0.1, t_on=0.99) is True
    y = np.array([0.0, 0.0])
    x1 = np.array([1.0, 2.0])
    assert np.allclose(yflow_next_state(y, x1, 0.5, 0.1), y + 0.1 * (x1 - y) / 0.5)
    assert np.allclose(yflow_next_state(y, x1, 0.9, 0.1), x1)
    assert np.allclose(damped_target(y, x1, 0.0), y)
    assert np.allclose(damped_target(y, x1, 1.0), x1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
