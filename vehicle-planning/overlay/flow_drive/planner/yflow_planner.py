import json
import os
import time
from typing import Optional

import numpy as np
import torch
from shapely.geometry import Point

from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

from flow_drive.planner.planner import FlowDrivePlannerWrapper, outputs_to_trajectory
from flow_drive.utils.train_utils import set_seed
from flow_drive.utils.infer_utils import _apply_speed_and_lateral_adjustments
from flow_drive.utils.post_processing import smooth_trajectories_preset, bound_speed_and_acceleration
from flow_drive.yflow.constraints import CorridorData
from flow_drive.yflow.posthoc import posthoc_fix
from flow_drive.yflow.sampler import YFlowConfig, YFlowContext, YFlowStats, sample_action_yflow


class FlowDriveYFlowPlannerWrapper(FlowDrivePlannerWrapper):
    def __init__(
        self,
        device: str = "cpu",
        mlflow_exp_name: str = "None",
        ckpt_path: str = "None",
        load_run_name: str = None,
        load_epoch: int = 0,
        post_mode: int = 2,
        render: bool = False,
        video_dir: str = None,
        emergency_brake_enabled: bool = True,
        yflow: Optional[dict] = None,
        corridor_source: str = "route",
        default_speed_limit: float = 15.0,
        stats_dir: Optional[str] = None,
        torch_threads: int = 1,
    ):
        super().__init__(device, mlflow_exp_name, ckpt_path, load_run_name, load_epoch,
                         post_mode, render, video_dir, emergency_brake_enabled)
        self._yflow_cfg = YFlowConfig(**(yflow or {}))
        self._corridor_source = corridor_source
        self._default_speed_limit = default_speed_limit
        self._stats_dir = None if stats_dir in (None, "None", "") else stats_dir
        if torch_threads and torch_threads > 0:
            torch.set_num_threads(int(torch_threads))

    def name(self) -> str:
        return "flow_drive_yflow"

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        if self._post_process in (4, 5):
            return self._plan_posthoc(current_input)
        if self._post_process not in (2, 3, 6):
            return super().compute_planner_trajectory(current_input)

        t_start = time.perf_counter()
        set_seed(self._master_seed + self._iteration)
        inputs = self.planner_input_to_model_inputs(current_input)
        ego_raw = inputs["ego_current_state"].clone()
        scorer = self._trajectory_scorer
        current_lane = scorer.prepare_scoring(current_input)
        speed_limit = current_lane.speed_limit_mps if current_lane is not None else None
        if speed_limit is None:
            speed_limit = self._default_speed_limit

        ctx = self._build_context(current_input, speed_limit)

        if self._post_process in (2, 6):
            scorer._iteration += 1
            outputs, stats = self._plan_yflow(inputs, ctx)
            if self._post_process == 6:
                with torch.inference_mode():
                    outputs = posthoc_fix(outputs, ego_raw.to(outputs.device), speed_limit, smooth=True)
            trajectory = InterpolatedTrajectory(
                trajectory=outputs_to_trajectory(outputs, current_input.history.ego_states,
                                                 self._future_horizon, self._step_interval))
        else:
            outputs, stats = self._plan_multiple_yflow(inputs, ctx, speed_limit)
            scores, ego_states_list = scorer.score_plans(outputs)
            trajectory = InterpolatedTrajectory(trajectory=ego_states_list[int(np.argmax(scores))])

        self._log(stats, (time.perf_counter() - t_start) * 1e3)
        self._iteration += 1
        return trajectory

    def _plan_posthoc(self, current_input: PlannerInput) -> AbstractTrajectory:
        t_start = time.perf_counter()
        set_seed(self._master_seed + self._iteration)
        inputs = self.planner_input_to_model_inputs(current_input)
        ego_raw = inputs["ego_current_state"].clone()
        scorer = self._trajectory_scorer
        current_lane = scorer.prepare_scoring(current_input)
        speed_limit = current_lane.speed_limit_mps if current_lane is not None else None
        if speed_limit is None:
            speed_limit = self._default_speed_limit
        scorer._iteration += 1
        raw = self._planner(inputs)
        with torch.inference_mode():
            outputs = posthoc_fix(raw, ego_raw.to(raw.device), speed_limit, smooth=self._post_process == 5)
        trajectory = InterpolatedTrajectory(
            trajectory=outputs_to_trajectory(outputs, current_input.history.ego_states,
                                             self._future_horizon, self._step_interval))
        stats = YFlowStats(level="posthoc", levels=["posthoc"] * outputs.shape[0])
        stats.shift_m = (outputs[..., :2] - raw[..., :2]).norm(dim=-1).mean().item()
        self._log(stats, (time.perf_counter() - t_start) * 1e3)
        self._iteration += 1
        return trajectory

    def _encode(self, inputs):
        fp = self._planner
        inputs_n = fp.observation_normalizer(inputs)
        return fp, inputs_n, fp.encoder(inputs_n)

    def _plan_yflow(self, inputs, ctx):
        with torch.inference_mode():
            fp, inputs_n, enc = self._encode(inputs)
            action_norm, stats = sample_action_yflow(
                fp.params, fp.decoder, enc, fp.noise_scheduler,
                inputs_n["ego_current_state"][:, :4], fp.action_normalizer, self._yflow_cfg, ctx)
            action = fp.action_normalizer.inverse(action_norm)
            heading = torch.atan2(action[:, :, 3], action[:, :, 2])
            return torch.cat([action[:, :, :2], heading.unsqueeze(-1)], -1), stats

    def _plan_multiple_yflow(self, inputs, ctx, speed_limit):
        speed_offsets, lateral_offsets = self._speed_offsets, self._lateral_offset
        with torch.inference_mode():
            fp = self._planner
            ego_raw = inputs["ego_current_state"].clone()
            fp_, inputs_n, enc = self._encode(inputs)
            B, N, Dh = enc["encoding"].shape
            T = fp.params.diffuser.pred_horizon
            S = len(speed_offsets) * len(lateral_offsets)
            enc_e = {"encoding": enc["encoding"][None].expand(S, -1, -1, -1).reshape(S * B, N, Dh),
                     "mask": enc["mask"][None].expand(S, -1, -1).reshape(S * B, N)}
            ego6 = inputs_n["ego_current_state"][:, :6]
            ego_e = ego6[..., :4][None].expand(S, -1, -1).reshape(S * B, 4)
            ego_speed = torch.norm(ego6[..., 4:6], dim=-1)
            hook = lambda x: _apply_speed_and_lateral_adjustments(
                x, ego_e, speed_offsets, lateral_offsets, B, S, T, ego_speed)
            ctx_e = YFlowContext(v0=ctx.v0.repeat(S * B, 1), v_limit=ctx.v_limit.repeat(S * B),
                                 corridor=ctx.corridor, obstacles=ctx.obstacles, cache=ctx.cache)
            action_norm, stats = sample_action_yflow(
                fp.params, fp.decoder, enc_e, fp.noise_scheduler, ego_e, fp.action_normalizer,
                self._yflow_cfg, ctx_e, step_hook=hook,
                hook_index=fp.params.inference.flow_inference_iter // 2 - 1)
            actions = fp.action_normalizer.inverse(action_norm).view(-1, B, T, 4)
            heading = torch.atan2(actions[..., 3], actions[..., 2])
            actions = torch.cat([actions[..., :2], heading.unsqueeze(-1)], -1)
            cur = torch.stack([ego_raw[:, 0], ego_raw[:, 1], torch.atan2(ego_raw[:, 3], ego_raw[:, 2])], -1)
            with_cur = torch.cat([cur[None, :, None, :].expand(actions.shape[0], -1, 1, -1), actions], 2)
            actions = smooth_trajectories_preset(with_cur, preset="strong")[:, :, 1:, :]
            actions = bound_speed_and_acceleration(
                actions, ego_raw, torch.tensor([speed_limit], device=actions.device))
        return actions, stats

    def _build_context(self, current_input, speed_limit) -> YFlowContext:
        cfg = self._yflow_cfg
        ego_state = current_input.history.current_state[0]
        ra = ego_state.rear_axle
        ex, ey, eh = ra.x, ra.y, ra.heading
        c, s = np.cos(eh), np.sin(eh)
        R_T = np.array([[c, s], [-s, c]])

        def to_ego(xy):
            return (np.asarray(xy, dtype=np.float64) - np.array([ex, ey])) @ R_T.T

        vel = ego_state.dynamic_car_state.rear_axle_velocity_2d
        v0 = torch.tensor([[vel.x, vel.y]], dtype=torch.float32)
        ctx = YFlowContext(v0=v0, v_limit=torch.tensor([float(speed_limit)], dtype=torch.float32))
        if cfg.use_corridor:
            ctx.corridor = self._corridor(ex, ey, float(np.hypot(vel.x, vel.y)), to_ego, R_T)
        if cfg.use_obstacles:
            ctx.obstacles = self._obstacles(ego_state, ex, ey, to_ego)
        return ctx

    def _corridor(self, ex, ey, speed, to_ego, R_T) -> Optional[CorridorData]:
        import shapely
        from shapely.ops import unary_union

        sc = self._trajectory_scorer
        cl = sc._centerline
        if cl is None:
            return None
        s0 = float(cl.project(Point(ex, ey)))
        ahead = max(30.0, 4.0 * speed + 25.0)
        s_arr = np.arange(max(0.0, s0 - 5.0), min(cl.length, s0 + ahead), 1.0)
        if len(s_arr) < 3:
            return None
        se2 = cl.interpolate(s_arr, as_array=True)
        cxy, th = se2[:, :2], se2[:, 2]
        tan = np.stack([np.cos(th), np.sin(th)], -1)
        nor = np.stack([-np.sin(th), np.cos(th)], -1)

        if self._corridor_source == "route":
            polys = [lane.polygon for lane in sc._route_lane_dict.values()]
        else:
            polys = list(sc._drivable_area_map._geometries)
        ego_pt = Point(ex, ey)
        polys = [p for p in polys if p.distance(ego_pt) < ahead + 10.0]
        if not polys:
            return None
        area = unary_union(polys)
        W = 12.0
        rays = shapely.linestrings(np.stack([cxy - W * nor, cxy + W * nor], 1))
        inter = shapely.intersection(rays, area)
        lo = np.zeros(len(s_arr)); hi = np.zeros(len(s_arr)); valid = np.zeros(len(s_arr), dtype=bool)
        for i, g in enumerate(inter):
            if g is None or g.is_empty:
                continue
            pieces = [g] if g.geom_type == "LineString" else [p for p in getattr(g, "geoms", [])
                                                               if p.geom_type == "LineString"]
            pt = shapely.points(cxy[i])
            for piece in pieces:
                if piece.distance(pt) < 1e-3:
                    proj = (np.asarray(piece.coords) - cxy[i]) @ nor[i]
                    lo[i], hi[i], valid[i] = proj.min(), proj.max(), True
                    break
        if valid.sum() < 3:
            return None
        f = lambda a: torch.as_tensor(a, dtype=torch.float64)
        return CorridorData(center=f(to_ego(cxy)), normal=f(nor @ R_T.T), tangent=f(tan @ R_T.T),
                            lo=f(lo), hi=f(hi), valid=torch.as_tensor(valid))

    def _obstacles(self, ego_state, ex, ey, to_ego):
        import shapely

        obs = self._trajectory_scorer._observation
        T = self._planner.params.diffuser.pred_horizon
        ego_poly = ego_state.car_footprint.geometry
        ignore = set(obs[0].intersects(ego_poly))
        ego_pt = Point(ex, ey)
        out = []
        for k in range(1, T + 1):
            try:
                occ = obs[k]
            except AssertionError:
                out.append(np.array([], dtype=object))
                continue
            geoms = np.asarray(occ._geometries, dtype=object)
            tokens = list(occ.tokens)
            if len(geoms) == 0:
                out.append(geoms)
                continue
            keep = (shapely.distance(geoms, ego_pt) < 60.0) & np.array([t not in ignore for t in tokens])
            geoms = geoms[keep]
            out.append(shapely.transform(geoms, to_ego) if len(geoms) else geoms)
        return out

    def _log(self, stats, total_ms):
        if self._stats_dir is None:
            return
        os.makedirs(self._stats_dir, exist_ok=True)
        rec = {"planner_id": self._planner_id, "iter": self._iteration, "mode": self._post_process,
               "total_ms": total_ms, **stats.as_dict()}
        with open(os.path.join(self._stats_dir, f"{self._planner_id}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
