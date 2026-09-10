"""Robot-domain terminal optimizer for the documented no-P Y-Flow path."""

from __future__ import annotations

import numpy as np

from experiments.pov_projection_phase2.projection import GoalAwareTrajectoryProjector


class YFlowRobotTerminalOptimizer(GoalAwareTrajectoryProjector):
    """Acados terminal optimizer with Y-Flow's time-varying raw penalty.

    The remote-main 2-D physical manifold operator has no robot equivalent in
    this repository.  This therefore implements Y-Flow's documented ``mu=0``
    path: raw-target tracking plus domain cost under the robot constraints.
    """

    def set_raw_tracking_weight(self, raw_weight: float) -> None:
        if self.goal_mode != "soft":
            raise ValueError("Phase 1.5 requires the soft robot task cost")
        width = self.stage_yref_size
        weight = np.eye(width)
        weight[:7, :7] *= raw_weight
        weight[7:14, 7:14] *= 1e-3
        weight[14:21, 14:21] *= 1e-3
        weight[21:35, 21:35] *= 1e-4
        weight[-3:, -3:] *= self.goal_weight
        for stage in range(self.horizon):
            self.solver.cost_set(stage, "W", weight)

    def optimize_terminal(
        self,
        candidate_trajectory: np.ndarray,
        start_state: np.ndarray,
        goal: np.ndarray,
        raw_weight: float,
    ):
        self.set_raw_tracking_weight(raw_weight)
        return self.project_trajectory(candidate_trajectory, start_state, goal)
