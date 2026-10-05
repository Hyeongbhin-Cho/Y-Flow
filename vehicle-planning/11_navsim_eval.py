#!/usr/bin/env python
import argparse
import os
import time
import traceback

import numpy as np
import pandas as pd

MODES = {"fd": 0, "fdstar": 1, "yflow": 2, "yflow_star": 3, "fd_clip": 4, "fd_post": 5, "yflow_post": 6}
YFLOW = {"yflow", "yflow_star", "yflow_post"}
SPLIT_CFG = {
    "mini": ["scenario_builder=nuplan_mini", "scenario_filter=one_of_each_scenario_type"],
    "val14": ["scenario_builder=nuplan", "scenario_filter=val14"],
    "test14-hard": ["scenario_builder=nuplan_challenge", "scenario_filter=test14-hard"],
    "test14-random": ["scenario_builder=nuplan_challenge", "scenario_filter=test14-random"],
}
COLS = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
        "traffic_light_compliance": "TLC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
        "lane_keeping": "LK", "history_comfort": "HC"}


REFS = {"human", "pdmc"}


def parse_yset(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        vl = v.strip().lower()
        if vl in ("true", "false"):
            out[k] = vl == "true"
        else:
            try:
                out[k] = int(v) if vl.lstrip("-").isdigit() else float(v)
            except ValueError:
                out[k] = v
    return out


def parse_method(spec):
    base, _, stage = spec.partition("-")
    if base in REFS:
        return base, None, None, base
    if base not in MODES:
        raise SystemExit(f"[err] unknown method {spec}")
    if base in YFLOW:
        stage = stage or "kin"
        if stage not in ("kin", "corr", "full", "kin2", "corr2"):
            raise SystemExit(f"[err] stage must be kin|corr|full|kin2|corr2: {spec}")
        return base, MODES[base], stage, f"{base}-{stage}-v2"
    return base, MODES[base], None, base


def build_scenarios(a):
    from hydra import compose, initialize_config_dir
    from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
    from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential

    cfgdir = os.path.join(os.environ["NUPLAN_DEVKIT_ROOT"], "nuplan/planning/script/config/simulation")
    ovs = ["+simulation=closed_loop_nonreactive_agents"] + SPLIT_CFG[a.split]
    if a.split == "mini":
        ovs += [f"scenario_filter.limit_total_scenarios={a.mini_n}",
                f"scenario_filter.num_scenarios_per_type={a.per_type}"]
    if a.data_root:
        ovs.append(f"scenario_builder.data_root={a.data_root}")
    ovs.append("hydra.searchpath=[pkg://flow_drive.config.scenario_filter, pkg://flow_drive.config, "
               "pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]")
    with initialize_config_dir(config_dir=cfgdir):
        cfg = compose(config_name="default_simulation", overrides=ovs)
    builder = build_scenario_builder(cfg)
    scenarios = builder.get_scenarios(build_scenario_filter(cfg.scenario_filter), Sequential())
    scenarios = sorted(scenarios, key=lambda s: s.token)
    if a.nshard > 1:
        scenarios = scenarios[a.shard::a.nshard]
    if a.limit:
        scenarios = scenarios[: a.limit]
    return scenarios


class Shifted:

    def __init__(self, sc, k):
        self._sc, self._k = sc, k

    def __getattr__(self, name):
        return getattr(self._sc, name)

    @property
    def initial_ego_state(self):
        return self._sc.get_ego_state_at_iteration(self._k)

    @property
    def initial_tracked_objects(self):
        return self._sc.get_tracked_objects_at_iteration(self._k)

    @property
    def start_time(self):
        return self._sc.get_time_point(self._k)

    def get_number_of_iterations(self):
        return self._sc.get_number_of_iterations() - self._k

    def get_time_point(self, iteration):
        return self._sc.get_time_point(iteration + self._k)

    def get_ego_state_at_iteration(self, iteration):
        return self._sc.get_ego_state_at_iteration(iteration + self._k)

    def get_tracked_objects_at_iteration(self, iteration, *args, **kw):
        return self._sc.get_tracked_objects_at_iteration(iteration + self._k, *args, **kw)

    def get_traffic_light_status_at_iteration(self, iteration):
        return self._sc.get_traffic_light_status_at_iteration(iteration + self._k)

    def get_ego_past_trajectory(self, iteration, *args, **kw):
        return self._sc.get_ego_past_trajectory(iteration + self._k, *args, **kw)

    def get_ego_future_trajectory(self, iteration, *args, **kw):
        return self._sc.get_ego_future_trajectory(iteration + self._k, *args, **kw)

    def get_past_tracked_objects(self, iteration, *args, **kw):
        return self._sc.get_past_tracked_objects(iteration + self._k, *args, **kw)

    def get_future_tracked_objects(self, iteration, *args, **kw):
        return self._sc.get_future_tracked_objects(iteration + self._k, *args, **kw)


def navsim_parts():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents

    prop = TrajectorySampling(num_poses=40, interval_length=0.1)

    class MCP(MetricCacheProcessor):
        def _interpolate_traffic_light_status(self, scenario):
            H, dt, st = self._proposal_sampling.time_horizon, self._proposal_sampling.interval_length, scenario.database_interval
            ts = np.arange(0, int(round(H / dt)) + 1) * dt
            return [list(scenario.get_traffic_light_status_at_iteration(int(round(t / st)))) for t in ts]

    return dict(prop=prop, mcp=MCP(None, False, prop), simulator=PDMSimulator(prop),
                scorer=PDMScorer(prop, PDMScorerConfig(human_penalty_filter=True)),
                agents=LogReplayTrafficAgents(prop))


def to_navsim_trajectory(traj, ego_state):
    from nuplan.common.actor_state.state_representation import TimePoint
    from nuplan.common.geometry.convert import absolute_to_relative_poses
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory

    t0 = ego_state.time_point.time_us
    end = traj.end_time.time_us
    pts = [TimePoint(min(t0 + int(0.5e6 * i), end)) for i in range(1, 9)]
    poses = [s.rear_axle for s in traj.get_state_at_times(pts)]
    rel = absolute_to_relative_poses([ego_state.rear_axle] + poses)[1:]
    return Trajectory(poses=np.array([[p.x, p.y, p.heading] for p in rel], dtype=np.float32),
                      trajectory_sampling=TrajectorySampling(num_poses=8, interval_length=0.5))


def two_frame_ec(now_states, prev_states, dt_frames, interval):
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_two_frame_extended_comfort)
    o = int(round(dt_frames / interval))
    cur, prev = now_states[:-o], prev_states[o:]
    tp = np.arange(cur.shape[0]) * interval
    return float(ego_is_two_frame_extended_comfort(cur[None], prev[None], tp)[0])


def final_score(row, ec):
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex as W
    wm = np.array(row["weighted_metrics"], dtype=np.float64).copy()
    wa = np.array(row["weighted_metrics_array"], dtype=np.float64).copy()
    if ec is None or np.isnan(ec):
        wm[W.TWO_FRAME_EXTENDED_COMFORT] = 0.0
        wa[W.TWO_FRAME_EXTENDED_COMFORT] = 0.0
    else:
        wm[W.TWO_FRAME_EXTENDED_COMFORT] = ec
    return float(row["multiplicative_metrics_prod"]) * float((wm * wa).sum() / wa.sum())


class Planner:

    def __init__(self, ckpt, device, yset=None):
        self.yset = yset or {}
        from flow_drive.planner.yflow_planner import FlowDriveYFlowPlannerWrapper
        self.p = FlowDriveYFlowPlannerWrapper(device=device, ckpt_path=ckpt, mlflow_exp_name="None",
                                              load_run_name=None, load_epoch=0, post_mode=0, stats_dir=None)
        self.loaded = False

    def plan(self, spec, init, planner_input):
        from flow_drive.planner.planner import TrajectoryScorer
        from flow_drive.yflow.sampler import YFlowConfig
        _, mode, stage, _ = spec
        p = self.p
        if not self.loaded:
            p.initialize(init)
            self.loaded = True
        p._iteration = 0
        p._map_api = init.map_api
        p._route_roadblock_ids = init.route_roadblock_ids
        p._initialization = init
        eb = p._trajectory_scorer._emergency_brake_enabled
        p._trajectory_scorer = TrajectoryScorer(emergency_brake_enabled=eb)
        p._trajectory_scorer.initialize(init.map_api, init.route_roadblock_ids)
        p._post_process = mode
        if stage is not None:
            kw = dict(accel_mode="lonlat") if stage.endswith("2") else {}
            p._yflow_cfg = YFlowConfig(use_corridor=stage in ("corr", "full", "corr2"), use_obstacles=stage == "full",
                                       fallback_resample=True, corridor_precheck=True, **{**kw, **self.yset})
        t = time.perf_counter()
        traj = p.compute_planner_trajectory(planner_input)
        return traj, (time.perf_counter() - t) * 1e3


def planner_io(sc, buffer_s=2.0):
    from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
    from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
    from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
    from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
    init = PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                 mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
    dur = buffer_s + sc.database_interval
    size = int(dur / sc.database_interval) + 1
    hist = SimulationHistoryBuffer.initialize_from_scenario(size, sc, DetectionsTracks)
    hist.append(sc.initial_ego_state, sc.initial_tracked_objects)
    pin = PlannerInput(iteration=SimulationIteration(index=0, time_point=sc.start_time), history=hist,
                       traffic_light_data=list(sc.get_traffic_light_status_at_iteration(0)))
    return init, pin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=list(SPLIT_CFG))
    ap.add_argument("--chunk", type=int, default=0, help="chunk id from 05_fetch_chunk.py (data_root = chunk dir)")
    ap.add_argument("--data_root", default="")
    ap.add_argument("--methods", nargs="+", default=["fd", "fd_post", "yflow-kin", "yflow_post-kin", "fdstar", "yflow_star-kin"])
    ap.add_argument("--mini_n", type=int, default=300)
    ap.add_argument("--per_type", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--yset", nargs="*", default=[], help="Y-Flow 설정 덮어쓰기, 예: a_lon_min=-6 retime_final_only=true")
    ap.add_argument("--ytag", default="", help="--yset을 쓸 때 Y-Flow 결과 이름 끝에 붙일 라벨 (예: b6)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("RESULTS_DIR", "/root/fd_yflow_results"), "navsim_eval"))
    a = ap.parse_args()
    if a.chunk and not a.data_root:
        a.data_root = os.path.join(os.environ["NUPLAN_DATA_ROOT"], "chunks", a.split, f"c{a.chunk:02d}")
    specs = [parse_method(m) for m in a.methods]
    yset = parse_yset(a.yset)
    if yset and not a.ytag:
        raise SystemExit("[err] --yset을 쓰면 --ytag로 결과 이름을 구분해야 함 (기존 결과를 덮지 않도록)")
    if a.ytag:
        specs = [(b, m, st, f"{t}-{a.ytag}" if st is not None else t) for b, m, st, t in specs]
    part = (f"c{a.chunk:02d}" if a.chunk else "all") + (f".shard{a.shard}" if a.nshard > 1 else "")
    paths = {s[3]: os.path.join(a.out, a.split, s[3], part + ".parquet") for s in specs}
    todo = [s for s in specs if not os.path.exists(paths[s[3]])]
    for s in specs:
        if s not in todo:
            print(f"[skip] {s[3]} ({paths[s[3]]} 있음)")
    if not todo:
        return

    scenarios = build_scenarios(a)
    print(f"[navsim] {a.split} {part}: 시나리오 {len(scenarios)}개, 방법 {[s[3] for s in todo]}", flush=True)
    nv = navsim_parts()
    from navsim.evaluate.pdm_score import pdm_score
    planner = Planner(os.environ["CKPT_PATH"], a.device, yset) if any(s[0] not in REFS for s in todo) else None
    rows = {s[3]: [] for s in todo}
    t_start = time.time()
    for n, sc in enumerate(scenarios):
        k = int(round(0.5 / sc.database_interval))
        frames = {}
        try:
            for fname, off in (("prev", 0), ("now", k)):
                view = Shifted(sc, off)
                mc = nv["mcp"].compute_metric_cache(view)
                init, pin = planner_io(view)
                frames[fname] = (view, mc, init, pin)
        except Exception:
            print(f"[warn] {sc.token}: metric cache 실패 -> 건너뜀\n{traceback.format_exc(limit=3)}", flush=True)
            continue
        for s in todo:
            res = {}
            for fname, (view, mc, init, pin) in frames.items():
                try:
                    if s[0] == "human":
                        nt, ms = mc.human_trajectory, 0.0
                    elif s[0] == "pdmc":
                        nt, ms = to_navsim_trajectory(mc.trajectory, mc.ego_state), 0.0
                    else:
                        traj, ms = planner.plan(s, init, pin)
                        nt = to_navsim_trajectory(traj, mc.ego_state)
                    df, states = pdm_score(mc, nt, nv["prop"], nv["simulator"], nv["scorer"], nv["agents"])
                    res[fname] = (df.iloc[0].to_dict(), states, ms, mc.timepoint.time_s)
                except Exception:
                    print(f"[warn] {sc.token} {fname} {s[3]}: {traceback.format_exc(limit=3)}", flush=True)
            ec = np.nan
            if "prev" in res and "now" in res:
                ec = two_frame_ec(res["now"][1], res["prev"][1], res["now"][3] - res["prev"][3],
                                  nv["prop"].interval_length)
            for fname, (r, _, ms, _) in res.items():
                e = ec if fname == "now" else np.nan
                out = {"token": sc.token, "scenario_type": sc.scenario_type, "log_name": sc.log_name, "frame": fname}
                out.update({v: float(r[c]) for c, v in COLS.items()})
                out.update({"EC": e, "EPDMS": final_score(r, e), "PDMS_nav": float(r["pdm_score"]), "plan_ms": ms})
                rows[s[3]].append(out)
        if (n + 1) % 10 == 0 or n + 1 == len(scenarios):
            el = time.time() - t_start
            print(f"[navsim] {n + 1}/{len(scenarios)}  {el / 60:.1f}분 경과, 남은 예상 {el / (n + 1) * (len(scenarios) - n - 1) / 60:.1f}분", flush=True)

    for s in todo:
        df = pd.DataFrame(rows[s[3]])
        os.makedirs(os.path.dirname(paths[s[3]]), exist_ok=True)
        df.to_parquet(paths[s[3]])
        now = df[df["frame"] == "now"]
        print(f"[done] {s[3]:>20}: EPDMS {100 * now['EPDMS'].mean():.2f} (n={len(now)}) -> {paths[s[3]]}")


if __name__ == "__main__":
    main()
