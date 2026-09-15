# -*- coding: utf-8 -*-
# experiments/exp_03_pedestrian/experiments/phase1_yflow/neighbors.py
"""Neighbour future trajectories in each agent's MoFlow frame (planning setting).

MoFlow's ETH/UCY loader is single-agent: every sample is translated to its last
observed position and rotated by ``rotate_traj``. This module rebuilds the
co-present agents of each 20-frame window from ``seq_start_end`` and maps their
ground-truth futures into the same frame, so they can act as known moving
obstacles (the Exp-04 analogue of known obstacles).
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np


PAST = 8


def rotation_matrices(traj: np.ndarray, rotate_time_frame: int, rotate: bool) -> np.ndarray:
    """Replicates data/dataloader_eth_ucy.rotate_traj: [N, 2, 2]."""
    n = traj.shape[0]
    if not rotate:
        return np.repeat(np.eye(2)[None], n, axis=0)
    past_abs = traj[:, :PAST]
    past_rel = past_abs - past_abs[:, -1:]
    diff = past_rel[:, rotate_time_frame]
    theta = np.arctan(diff[:, 1] / (diff[:, 0] + 1e-5))
    theta = np.where(diff[:, 0] < 0, theta + math.pi, theta)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, s], -1), np.stack([-s, c], -1)], -2)


class NeighbourTable:
    def __init__(self, pkl_path: Path, rotate: bool, rotate_time_frame: int):
        with Path(pkl_path).open("rb") as f:
            payload = pickle.load(f)
        traj = np.asarray(payload["traj"], dtype=np.float64)
        seq = np.asarray(payload["seq_start_end"], dtype=np.int64)
        self.traj = traj
        self.R = rotation_matrices(traj, rotate_time_frame, rotate)
        self.origin = traj[:, PAST - 1]
        self.window = np.empty(traj.shape[0], dtype=np.int64)
        for w, (a, b) in enumerate(seq):
            self.window[a:b] = w
        self.seq = seq

    def to_frame(self, i: int, points_abs: np.ndarray) -> np.ndarray:
        return (points_abs - self.origin[i]) @ self.R[i].T

    def own_future(self, idx: np.ndarray) -> np.ndarray:
        return np.stack([self.to_frame(i, self.traj[i, PAST:]) for i in idx])

    def future_of(self, j: int, source: str) -> np.ndarray:
        """Absolute future of agent j: 'gt' (oracle) or 'cv' (constant velocity from its last two observed frames)."""
        if source == "gt":
            return self.traj[j, PAST:]
        if source == "cv":
            last = self.traj[j, PAST - 1]
            vel = last - self.traj[j, PAST - 2]
            steps = np.arange(1, self.traj.shape[1] - PAST + 1)[:, None]
            return last[None] + steps * vel[None]
        raise ValueError(f"unknown neighbour source {source!r}")

    def gather(self, idx: np.ndarray, reach: np.ndarray | None = None, source: str = "gt") -> tuple[np.ndarray, np.ndarray]:
        """q [B, M, F, 2] (metres, agent frame), mask [B, M]; M = max neighbours in batch.

        ``reach`` [F] drops neighbours that never enter the agent's kinematic
        reachable disc (||q_{j,k}|| > reach_k for all k). With reach_k =
        v_max * dt * k + r_safe their clearance rows cannot be active for any
        speed-feasible trajectory, so dropping them is exact.
        """
        lists = []
        for i in idx:
            a, b = self.seq[self.window[i]]
            others = [j for j in range(a, b) if j != i]
            if reach is not None:
                others = [j for j in others
                          if (np.linalg.norm(self.to_frame(i, self.future_of(j, source)), axis=-1) <= reach).any()]
            lists.append(others)
        M = max(1, max(len(o) for o in lists))
        F = self.traj.shape[1] - PAST
        q = np.zeros((len(idx), M, F, 2))
        mask = np.zeros((len(idx), M), dtype=bool)
        for bi, (i, others) in enumerate(zip(idx, lists)):
            for m, j in enumerate(others):
                q[bi, m] = self.to_frame(i, self.future_of(j, source))
                mask[bi, m] = True
        return q, mask
