#!/usr/bin/env python
import argparse
import glob
import os

import numpy as np

EXP = os.environ.get("NUPLAN_EXP_ROOT", "/root/nuplan/exp")
C_BG, C_ROAD, C_LANE = "#dedbd5", "#fcfcfb", "#b9b7b2"
C_FD, C_YF, C_EGO, C_AGENT, C_RED = "#4a7fc1", "#e0703f", "#2b2b2b", "#a3a29c", "#c62f3e"


def run_dirs(split, ch, tag):
    return sorted(d for d in glob.glob(os.path.join(EXP, "**", "flow_drive", split, ch, tag, "*"), recursive=True)
                  if os.path.isdir(d))


def tag_names(split, ch):
    ds = glob.glob(os.path.join(EXP, "**", "flow_drive", split, ch, "*"), recursive=True)
    return sorted({os.path.basename(d) for d in ds if os.path.isdir(d)})


def chunk_tags(split, ch, tag):
    import re
    pat = re.compile(rf"^{re.escape(tag)}(-{re.escape(split)}-c\d+)?$")
    return [t for t in tag_names(split, ch) if pat.match(t)]


def merged_scores(split, ch, tag):
    import pandas as pd
    parts = []
    for t in chunk_tags(split, ch, tag):
        rd = run_dirs(split, ch, t)
        s = scen_scores(rd[-1]) if rd else None
        if s is not None:
            parts.append(s)
    if not parts:
        return None
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")]


def find_log(split, ch, tag, token):
    for d in reversed(run_dirs(split, ch, tag)):
        hits = glob.glob(os.path.join(d, "simulation_log", "**", f"{token}.msgpack.xz"), recursive=True) + \
            glob.glob(os.path.join(d, "simulation_log", "**", f"{token}.pkl.xz"), recursive=True)
        if hits:
            return hits[0], d
    raise SystemExit(f"[err] simulation log 없음: {split}/{ch}/{tag}/…/{token} (15_viz_run.sh로 기록을 켜고 다시 돌려야 함)")


def scen_scores(run_dir):
    import pandas as pd
    fs = glob.glob(os.path.join(run_dir, "aggregator_metric", "*.parquet"))
    if not fs:
        return None
    df = pd.read_parquet(fs[0])
    df = df[df["scenario"] != "final_score"]
    if "log_name" in df:
        df = df[df["log_name"].notna()]
    return df.set_index("scenario")


def list_mode(a):
    import pandas as pd
    short = {"score": "CLS", "no_ego_at_fault_collisions": "col", "drivable_area_compliance": "drv",
             "time_to_collision_within_bound": "ttc", "ego_progress_along_expert_route": "prog"}
    tabs, typ = {}, None
    for tag in a.tags:
        s = merged_scores(a.split, a.ch, tag)
        if s is None:
            raise SystemExit(f"[err] 결과 없음: {a.split}/{a.ch}/{tag}  (--avail 로 이름 확인)")
        print(f"[scores] {tag}: 시나리오 {len(s)}개 ({len(chunk_tags(a.split, a.ch, tag))}개 결과 폴더 합침)")
        typ = s["scenario_type"] if typ is None else typ
        tabs[tag] = s[[c for c in short if c in s]].rename(columns=short)
    idx = sorted(set.intersection(*[set(t.index) for t in tabs.values()]))
    t = pd.DataFrame({"type": typ.loc[idx]})
    for i, tag in enumerate(a.tags):
        for c in tabs[tag].columns:
            t[f"{i}{c}"] = tabs[tag].loc[idx, c].round(2)
    pairs = [(a.tags[i], a.tags[i + 1]) for i in range(0, len(a.tags) - 1, 2)]
    for j, (x, y) in enumerate(pairs):
        t[f"d{j}"] = (100 * (tabs[y].loc[idx, "CLS"] - tabs[x].loc[idx, "CLS"])).round(1)
    t["spread"] = t[[f"d{j}" for j in range(len(pairs))]].abs().sum(axis=1)
    pd.set_option("display.width", 300, "display.max_columns", 60)
    print("열 번호: " + ", ".join(f"{i}={tag}" for i, tag in enumerate(a.tags)))
    print("d0, d1 ... = 짝 차이 (뒤 - 앞, CLS 점): " + ", ".join(f"d{j}={y} - {x}" for j, (x, y) in enumerate(pairs)))
    cols = ["type"] + [f"{i}CLS" for i in range(len(a.tags))] + [f"d{j}" for j in range(len(pairs))] + \
           [f"{i}{c}" for i in range(len(a.tags)) for c in ("col", "drv", "ttc", "prog")]
    t = t[[c for c in cols if c in t] + ["spread"]]
    for j in range(len(pairs)):
        print(f"\n[d{j} 가장 낮은 {a.top}개]")
        print(t.sort_values(f"d{j}").head(a.top).drop(columns="spread").to_string())
        print(f"\n[d{j} 가장 높은 {a.top}개]")
        print(t.sort_values(f"d{j}").tail(a.top).drop(columns="spread").to_string())
    cand = t.sort_values("spread", ascending=False).head(a.top)
    print(f"\n[추천 후보: 짝 차이 합(spread)이 큰 {a.top}개]")
    print(cand.to_string())
    if len(cand):
        print(f"PICK {cand.index[0]}")


def poly(ax, geom, **kw):
    from matplotlib.patches import Polygon as P
    geoms = getattr(geom, "geoms", [geom])
    for g in geoms:
        if g.is_empty or not hasattr(g, "exterior"):
            continue
        ax.add_patch(P(np.asarray(g.exterior.coords)[:, :2], closed=True, **kw))


def xy_path(states):
    return np.array([[s.x, s.y] for s in states]) if len(states) else np.zeros((0, 2))


class Scene:

    def __init__(self, map_api, center, radius):
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        pt = Point2D(float(center[0]), float(center[1]))
        layers = [L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA, L.LANE, L.LANE_CONNECTOR]
        objs = map_api.get_proximal_map_objects(pt, radius, layers)
        self.road = [o.polygon for k in (L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA)
                     for o in objs.get(k, [])]
        self.lanes = [xy_path(o.baseline_path.discrete_path) for k in (L.LANE, L.LANE_CONNECTOR) for o in objs.get(k, [])]
        self.map_api = map_api
        self._conn = {}

    def connector_path(self, cid):
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        if cid not in self._conn:
            try:
                o = self.map_api.get_map_object(str(cid), L.LANE_CONNECTOR)
                self._conn[cid] = xy_path(o.baseline_path.discrete_path) if o is not None else None
            except Exception:
                self._conn[cid] = None
        return self._conn[cid]

    def draw(self, ax):
        ax.set_facecolor(C_BG)
        for g in self.road:
            poly(ax, g, facecolor=C_ROAD, edgecolor="none", zorder=1)
        for p in self.lanes:
            if len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], color=C_LANE, lw=0.8, zorder=2)


def offroad_steps(map_api, samples, thr=0.3):
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
    out = []
    for s in samples:
        bad = False
        for c in s.ego_state.car_footprint.oriented_box.all_corners():
            _, d = map_api.get_distance_to_nearest_map_object(Point2D(c.x, c.y), layer=L.DRIVABLE_AREA)
            if d is None or d >= thr:
                bad = True
                break
        out.append(bad)
    return np.array(out)


def render(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusType
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.planning.simulation.simulation_log import SimulationLog

    labels = a.labels or a.tags
    if len(labels) != len(a.tags):
        raise SystemExit("[err] --labels 개수가 --tags와 같아야 함")
    sides = []
    for tag, label in zip(a.tags, labels):
        path, rd = find_log(a.split, a.ch, tag, a.token)
        log = SimulationLog.load_data(__import__("pathlib").Path(path))
        samples = list(log.simulation_history.data)
        sc = scen_scores(rd)
        cls = 100 * float(sc.loc[a.token, "score"]) if sc is not None and a.token in sc.index else float("nan")
        color = C_YF if tag.startswith("yflow") else C_FD
        sides.append(dict(log=log, samples=samples, label=label, color=color, cls=cls))
        print(f"[load] {label}: {len(samples)} steps, CLS {cls:.2f}  <- {path}")

    scenario = sides[0]["log"].scenario
    map_api = scenario.map_api
    expert = xy_path([s.center for s in scenario.get_expert_ego_trajectory()])
    all_xy = np.concatenate([expert] + [xy_path([s.ego_state.center for s in d["samples"]]) for d in sides])
    center = all_xy.mean(0)
    radius = float(np.linalg.norm(all_xy - center, axis=1).max()) + a.view / 2 + 20
    scene = Scene(map_api, center, radius)
    for d in sides:
        d["off"] = offroad_steps(map_api, d["samples"])
        print(f"[off-road] {d['label']}: {int(d['off'].sum())} / {len(d['off'])} steps")

    n = min(len(d["samples"]) for d in sides)
    frames = list(range(0, n, a.stride))
    ncol = 1 if len(sides) == 1 else 2
    nrow = int(np.ceil(len(sides) / ncol))
    leg_rows = 1 if ncol == 2 else 3
    head = 0.35 + 0.28 * leg_rows
    fig_h = 6.6 * nrow + head
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.6 * ncol, fig_h), dpi=a.dpi, squeeze=False)
    axes = axes.ravel()
    for ax in axes[len(sides):]:
        ax.axis("off")
    fig.patch.set_facecolor("#fbfbf9")
    legend = [Line2D([], [], color="k", ls="--", lw=1.6, label="expert (log)"),
              Line2D([], [], color="#555", ls=":", lw=2, label="ego history"),
              Patch(facecolor=C_EGO, label="ego"),
              Line2D([], [], color=C_FD, lw=2.2, label="plan (FlowDrive)"),
              Line2D([], [], color=C_YF, lw=2.2, label="plan (Y-Flow)"),
              Patch(facecolor=C_AGENT, edgecolor="#6f6e69", label="other agents"),
              Line2D([], [], color=C_RED, lw=2, label="red light"),
              Line2D([], [], color=C_RED, marker="o", ls="none", label="off-road step"),
              Patch(facecolor=C_BG, label="off drivable area")]
    fig.legend(handles=legend, loc="upper center", ncol=9 if ncol == 2 else 3, frameon=False, fontsize=8.5)
    title_type = scenario.scenario_type

    def draw(k):
        i = frames[k]
        for ax, d in zip(axes[:len(sides)], sides):
            ax.clear()
            s = d["samples"][i]
            ego = s.ego_state
            if a.follow:
                cx, cy = ego.center.x, ego.center.y
            else:
                cx, cy = center
            hw, hh = a.view / 2, a.view / 2 * 0.9
            scene.draw(ax)
            for tl in s.traffic_light_status:
                if tl.status == TrafficLightStatusType.RED:
                    p = scene.connector_path(tl.lane_connector_id)
                    if p is not None and len(p) > 1:
                        ax.plot(p[:, 0], p[:, 1], color=C_RED, lw=1.6, alpha=0.8, zorder=3)
            ax.plot(expert[:, 0], expert[:, 1], "k--", lw=1.4, zorder=4)
            hist = xy_path([t.ego_state.center for t in d["samples"][: i + 1]])
            ax.plot(hist[:, 0], hist[:, 1], ls=":", color="#555", lw=2, zorder=5)
            off = [j for j in range(i + 1) if d["off"][j]]
            if off:
                oxy = hist[off]
                ax.plot(oxy[:, 0], oxy[:, 1], "o", color=C_RED, ms=3.5, zorder=9)
            for obj in s.observation.tracked_objects.tracked_objects:
                if obj.tracked_object_type == TrackedObjectType.EGO:
                    continue
                if abs(obj.center.x - cx) > hw + 10 or abs(obj.center.y - cy) > hh + 10:
                    continue
                veh = obj.tracked_object_type == TrackedObjectType.VEHICLE
                poly(ax, obj.box.geometry, facecolor=C_AGENT if veh else "#c9c7c0", edgecolor="#6f6e69",
                     lw=0.7, zorder=6)
            plan = xy_path([t.center for t in s.trajectory.get_sampled_trajectory()])
            ax.plot(plan[:, 0], plan[:, 1], color=d["color"], lw=2.4, alpha=0.9, zorder=7)
            poly(ax, ego.car_footprint.oriented_box.geometry, facecolor=C_EGO, edgecolor="#111", zorder=8)
            ax.set_xlim(cx - hw, cx + hw)
            ax.set_ylim(cy - hh, cy + hh)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{d['label']}\n{title_type}  {a.token[:8]}   CLS {d['cls']:.1f}", fontsize=11)
            ax.text(0.03, 0.95, f"t = {i * 0.1:.1f} s   v = {ego.dynamic_car_state.speed:.1f} m/s", transform=ax.transAxes,
                    fontsize=11, va="top", bbox=dict(facecolor="white", edgecolor="none", pad=3), zorder=10)
        return []

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - head / fig_h, wspace=0.03, hspace=0.12)
    out = a.out or f"viz_{a.token}.gif"
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out, writer=PillowWriter(fps=a.fps))
    draw(len(frames) - 1)
    fig.savefig(os.path.splitext(out)[0] + "_last.png")
    print(f"[done] {out}  ({len(frames)} frames, {a.fps} fps) + 마지막 장면 PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--ch", default="nr")
    ap.add_argument("--tags", nargs="+", default=[], help="결과 이름들 (2개면 1x2, 4개면 2x2)")
    ap.add_argument("--labels", nargs="*", default=[], help="패널 제목 (tags와 같은 순서)")
    ap.add_argument("--token", default="")
    ap.add_argument("--list", action="store_true", help="점수 차이가 큰 시나리오 목록만 출력 (청크 결과 자동 합침)")
    ap.add_argument("--avail", action="store_true", help="<split>/<ch> 아래 결과 이름 목록만 출력")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--view", type=float, default=70.0, help="화면 가로 폭 [m]")
    ap.add_argument("--follow", type=int, default=1, help="1이면 자차를 따라가는 화면, 0이면 고정 화면")
    ap.add_argument("--stride", type=int, default=2, help="몇 step마다 한 프레임 (1 step = 0.1 s)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.avail:
        print("\n".join(tag_names(a.split, a.ch)))
        return
    if a.list:
        return list_mode(a)
    if not a.token:
        raise SystemExit("[err] --token 필요 (먼저 --list로 고르기)")
    render(a)


if __name__ == "__main__":
    main()
