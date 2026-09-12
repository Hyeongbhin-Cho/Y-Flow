# -*- coding: utf-8 -*-
# data/clevrer_state.py
"""Packed CLEVRER scene state S, recognition dataset, and hard constraints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from data.base import BaseConstraint, register_dataset
from data.clevrer import CLEVRERDataset, VideoDataBundle, collate_clevrer
from utils.paths import ROOT

CLEVRER_COLORS = ("blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow")
CLEVRER_MATERIALS = ("metal", "rubber")
CLEVRER_SHAPES = ("cube", "cylinder", "sphere")
_COLOR = {name: i for i, name in enumerate(CLEVRER_COLORS)}
_MATERIAL = {name: i for i, name in enumerate(CLEVRER_MATERIALS)}
_SHAPE = {name: i for i, name in enumerate(CLEVRER_SHAPES)}


@dataclass(frozen=True)
class CLEVRERStateLayout:
    n_slots: int = 6
    n_frames: int = 33
    n_color: int = 8
    n_material: int = 2
    n_shape: int = 3

    @property
    def n_pairs(self) -> int:
        k = self.n_slots
        return k * (k - 1) // 2

    @property
    def dim(self) -> int:
        k, t = self.n_slots, self.n_frames
        return (
            k * (self.n_color + self.n_material + self.n_shape)
            + k * t
            + 2 * k * t * 3
            + t * self.n_pairs
        )

    def slices(self) -> dict[str, tuple[int, int]]:
        k, t = self.n_slots, self.n_frames
        cursor = 0
        out: dict[str, tuple[int, int]] = {}
        for name, size in (
            ("color", k * self.n_color),
            ("material", k * self.n_material),
            ("shape", k * self.n_shape),
            ("vis", k * t),
            ("pos", k * t * 3),
            ("vel", k * t * 3),
            ("coll", t * self.n_pairs),
        ):
            out[name] = (cursor, cursor + size)
            cursor += size
        return out


def pair_index(i: int, j: int, n_slots: int) -> int:
    if i > j:
        i, j = j, i
    if i == j:
        raise ValueError("pair_index requires i != j")
    return i * (2 * n_slots - i - 1) // 2 + (j - i - 1)


def layout_from_cfg(cfg: DictConfig | None = None) -> CLEVRERStateLayout:
    data = cfg.data if cfg is not None else None
    n_slots = int(data.get("n_slots", 6)) if data is not None else 6
    n_frames = int(data.get("n_frames", 33)) if data is not None else 33
    return CLEVRERStateLayout(n_slots=n_slots, n_frames=n_frames)


def _one_hot(index: int, size: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[int(index)] = 1.0
    return out


def unpack_state(packed: torch.Tensor | np.ndarray, layout: CLEVRERStateLayout) -> dict[str, Any]:
    """Split a packed vector [..., D] into named tensors."""
    is_torch = isinstance(packed, torch.Tensor)
    slices = layout.slices()
    k, t = layout.n_slots, layout.n_frames

    def chunk(name: str):
        lo, hi = slices[name]
        return packed[..., lo:hi]

    color = chunk("color").reshape(*packed.shape[:-1], k, layout.n_color)
    material = chunk("material").reshape(*packed.shape[:-1], k, layout.n_material)
    shape = chunk("shape").reshape(*packed.shape[:-1], k, layout.n_shape)
    vis = chunk("vis").reshape(*packed.shape[:-1], k, t)
    pos = chunk("pos").reshape(*packed.shape[:-1], k, t, 3)
    vel = chunk("vel").reshape(*packed.shape[:-1], k, t, 3)
    coll_flat = chunk("coll").reshape(*packed.shape[:-1], t, layout.n_pairs)
    if is_torch:
        coll = packed.new_zeros(*packed.shape[:-1], t, k, k)
    else:
        coll = np.zeros((*packed.shape[:-1], t, k, k), dtype=np.float32)
        coll_flat = np.asarray(coll_flat)
    pair = 0
    for i in range(k):
        for j in range(i + 1, k):
            if is_torch:
                coll[..., :, i, j] = coll_flat[..., :, pair]
                coll[..., :, j, i] = coll_flat[..., :, pair]
            else:
                coll[..., :, i, j] = coll_flat[..., :, pair]
                coll[..., :, j, i] = coll_flat[..., :, pair]
            pair += 1
    return {
        "color": color,
        "material": material,
        "shape": shape,
        "vis": vis,
        "pos": pos,
        "vel": vel,
        "coll": coll,
    }


def pack_parts(
    color: np.ndarray,
    material: np.ndarray,
    shape: np.ndarray,
    vis: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    coll: np.ndarray,
    layout: CLEVRERStateLayout,
) -> np.ndarray:
    k, t = layout.n_slots, layout.n_frames
    pairs = np.zeros((t, layout.n_pairs), dtype=np.float32)
    pair = 0
    for i in range(k):
        for j in range(i + 1, k):
            pairs[:, pair] = 0.5 * (coll[:, i, j] + coll[:, j, i])
            pair += 1
    return np.concatenate(
        [
            color.reshape(-1),
            material.reshape(-1),
            shape.reshape(-1),
            vis.reshape(-1),
            pos.reshape(-1),
            vel.reshape(-1),
            pairs.reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def annotation_to_state(
    annotation: dict[str, Any],
    frame_indices: np.ndarray | torch.Tensor,
    layout: CLEVRERStateLayout,
    *,
    dt: float = 1.0 / 16.0,
    collision_frame_tol: int = 3,
) -> np.ndarray:
    """Pack simulator JSON + sampled original frame ids into S."""
    if annotation is None:
        raise ValueError("Recognition state requires annotations")
    indices = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    if indices.shape[0] != layout.n_frames:
        raise ValueError(f"expected {layout.n_frames} frame indices, got {indices.shape[0]}")
    objects = sorted(annotation.get("object_property") or [], key=lambda o: int(o["object_id"]))
    if len(objects) > layout.n_slots:
        objects = objects[: layout.n_slots]
    id_to_slot = {int(obj["object_id"]): slot for slot, obj in enumerate(objects)}
    k, t = layout.n_slots, layout.n_frames
    color = np.zeros((k, layout.n_color), dtype=np.float32)
    material = np.zeros((k, layout.n_material), dtype=np.float32)
    shape = np.zeros((k, layout.n_shape), dtype=np.float32)
    vis = np.zeros((k, t), dtype=np.float32)
    pos = np.zeros((k, t, 3), dtype=np.float32)
    vel = np.zeros((k, t, 3), dtype=np.float32)
    coll = np.zeros((t, k, k), dtype=np.float32)
    for slot, obj in enumerate(objects):
        color[slot] = _one_hot(_COLOR[str(obj["color"])], layout.n_color)
        material[slot] = _one_hot(_MATERIAL[str(obj["material"])], layout.n_material)
        shape[slot] = _one_hot(_SHAPE[str(obj["shape"])], layout.n_shape)
    by_frame = {int(entry["frame_id"]): entry for entry in annotation.get("motion_trajectory") or []}
    for time_i, frame_id in enumerate(indices):
        entry = by_frame.get(int(frame_id))
        if entry is None:
            raise ValueError(f"missing motion_trajectory frame_id {int(frame_id)}")
        for obj in entry.get("objects") or []:
            slot = id_to_slot.get(int(obj["object_id"]))
            if slot is None:
                continue
            loc = np.asarray(obj.get("location", [0.0, 0.0, 0.0]), dtype=np.float32)
            pos[slot, time_i, : loc.shape[0]] = loc[:3]
            if "velocity" in obj:
                v = np.asarray(obj["velocity"], dtype=np.float32)
                vel[slot, time_i, : v.shape[0]] = v[:3]
                vis[slot, time_i] = 1.0 if obj.get("inside_camera_view", True) else 0.0
    for slot in range(len(objects)):
        missing = np.allclose(vel[slot], 0.0) and np.any(np.abs(pos[slot]) > 0)
        if missing:
            vel[slot, :-1] = (pos[slot, 1:] - pos[slot, :-1]) / float(dt)
            vel[slot, -1] = vel[slot, -2]
    for event in annotation.get("collision") or []:
        ids = [int(i) for i in (event.get("object_ids") or event.get("object") or [])]
        if len(ids) != 2 or ids[0] not in id_to_slot or ids[1] not in id_to_slot:
            continue
        frame_id = int(event["frame_id"])
        nearest = int(np.argmin(np.abs(indices - frame_id)))
        if abs(int(indices[nearest]) - frame_id) > int(collision_frame_tol):
            continue
        i, j = id_to_slot[ids[0]], id_to_slot[ids[1]]
        coll[nearest, i, j] = 1.0
        coll[nearest, j, i] = 1.0
    return pack_parts(color, material, shape, vis, pos, vel, coll, layout)


@dataclass
class CLEVRERStateMeta:
    n_slots: int
    n_frames: int
    fps: float
    dt: float
    r_xy: float
    z0: float
    tau_z: float
    v_max: float
    tau_kin: float
    tau_acc: float
    d0: float
    d_min: float
    tau_col: float
    delta_v: float
    eps_ident: float
    eps_null: float
    mean: tuple[float, ...]
    std: tuple[float, ...]


class CLEVRERStateConstraint(BaseConstraint):
    """Hard constraints on packed scene state S in simulator world coordinates."""

    def __init__(self, meta: CLEVRERStateMeta | dict[str, Any], layout: CLEVRERStateLayout | None = None):
        if isinstance(meta, dict):
            meta = CLEVRERStateMeta(**{key: meta[key] for key in CLEVRERStateMeta.__dataclass_fields__})
        self.meta = meta
        self.layout = layout or CLEVRERStateLayout(n_slots=meta.n_slots, n_frames=meta.n_frames)

    def _as_torch(self, p: torch.Tensor | np.ndarray) -> tuple[torch.Tensor, bool]:
        if isinstance(p, torch.Tensor):
            return p, False
        return torch.from_numpy(np.asarray(p, dtype=np.float32)), True

    def _parts(self, p: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = unpack_state(p, self.layout)
        vis = parts["vis"].clamp(0.0, 1.0)
        parts["vis"] = vis
        occupied = (parts["color"].clamp_min(0).sum(dim=-1) > 0.5).to(dtype=vis.dtype)
        parts["occupied"] = occupied
        parts["active"] = torch.maximum(vis.amax(dim=-1), occupied)
        return parts

    def h(self, p: torch.Tensor | np.ndarray) -> dict[str, torch.Tensor | np.ndarray]:
        p_t, to_numpy = self._as_torch(p)
        parts = self._parts(p_t)
        meta = self.meta
        layout = self.layout
        vis = parts["vis"]
        pos = parts["pos"]
        vel = parts["vel"]
        coll = parts["coll"]
        active = parts["active"]
        dt = float(meta.dt)

        color_mass = parts["color"].clamp_min(0).sum(dim=-1)
        mat_mass = parts["material"].clamp_min(0).sum(dim=-1)
        shape_mass = parts["shape"].clamp_min(0).sum(dim=-1)
        vocab = torch.stack(
            [(1.0 - color_mass).abs(), (1.0 - mat_mass).abs(), (1.0 - shape_mass).abs()],
            dim=-1,
        ).amax(dim=-1)
        occupied = parts["occupied"]
        h_vocab = (vocab * occupied).amax(dim=-1)

        n_obj = occupied.sum(dim=-1)
        h_count = torch.maximum(n_obj - float(layout.n_slots), 3.0 - n_obj)

        h_ident = p_t.new_zeros(p_t.shape[:-1])
        h_null = ((1.0 - occupied) * pos.norm(dim=-1).amax(dim=-1)).amax(dim=-1) - float(meta.eps_null)

        vis_pos = vis.unsqueeze(-1)
        table = (vis * (pos[..., :2].abs().amax(dim=-1) - float(meta.r_xy))).amax(dim=(-1, -2))
        plane = (vis * ((pos[..., 2] - float(meta.z0)).abs() - float(meta.tau_z))).amax(dim=(-1, -2))

        step = (pos[..., 1:, :] - pos[..., :-1, :]).norm(dim=-1)
        vis_step = vis[..., :-1] * vis[..., 1:]
        h_step = (vis_step * (step - float(meta.v_max) * dt)).amax(dim=(-1, -2))

        kin = (pos[..., 1:, :] - pos[..., :-1, :] - vel[..., :-1, :] * dt).norm(dim=-1)
        h_kin = (vis_step * (kin - float(meta.tau_kin))).amax(dim=(-1, -2))

        du = (vel[..., 1:, :] - vel[..., :-1, :]).norm(dim=-1)
        # coll: [..., T, K, K] -> per-slot collision flag [..., K, T]
        coll_slot = coll.amax(dim=-1).transpose(-1, -2)
        near_hit = torch.maximum(coll_slot[..., :-1], coll_slot[..., 1:])
        vis_jump = vis[..., :-1] * vis[..., 1:]
        h_acc = (vis_jump * (1.0 - near_hit) * (du - float(meta.tau_acc))).amax(dim=(-1, -2))

        h_sym = coll.transpose(-1, -2).sub(coll).abs().amax(dim=(-1, -2, -3))
        h_sym = torch.maximum(h_sym, coll.diagonal(dim1=-2, dim2=-1).abs().amax(dim=(-1, -2)))

        vis_t = vis.transpose(-1, -2)  # [..., T, K]
        vis_i = vis_t.unsqueeze(-1)
        vis_j = vis_t.unsqueeze(-2)
        h_colvis = (coll * (2.0 - vis_i - vis_j)).amax(dim=(-1, -2, -3))

        pi = pos.transpose(-2, -3)  # [..., T, K, 3]
        pj = pi.unsqueeze(-2)
        pi = pi.unsqueeze(-3)
        dist = (pi - pj).norm(dim=-1)
        h_contact = (coll * ((dist - float(meta.d0)).abs() - float(meta.tau_col))).amax(dim=(-1, -2, -3))
        vis_pair = vis_i * vis_j
        eye = torch.eye(layout.n_slots, device=p_t.device, dtype=p_t.dtype)
        off = 1.0 - eye
        h_penetrate = (vis_pair * off * (float(meta.d_min) - dist)).amax(dim=(-1, -2, -3))

        if vis.shape[-1] > 2:
            dvel = (vel[..., 2:, :] - vel[..., :-2, :]).norm(dim=-1)  # [..., K, T-2]
            dvel_t = dvel.transpose(-1, -2)  # [..., T-2, K]
            impulse_i = dvel_t.unsqueeze(-1)
            impulse_j = dvel_t.unsqueeze(-2)
            h_impulse = (
                coll[..., 1:-1, :, :] * (float(meta.delta_v) - impulse_i - impulse_j)
            ).amax(dim=(-1, -2, -3))
        else:
            h_impulse = p_t.new_zeros(p_t.shape[:-1])

        out = {
            "vocab": h_vocab,
            "count": h_count,
            "ident": h_ident,
            "null": h_null,
            "table": table,
            "plane": plane,
            "step": h_step,
            "kin": h_kin,
            "acc": h_acc,
            "sym": h_sym,
            "colvis": h_colvis,
            "contact": h_contact,
            "penetrate": h_penetrate,
            "impulse": h_impulse,
        }
        if to_numpy:
            return {key: value.detach().cpu().numpy() for key, value in out.items()}
        return out

    def cost(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        h_dict = self.h(p)
        terms = [value.clamp_min(0.0).square() for value in h_dict.values()]
        if isinstance(p, torch.Tensor):
            return 0.5 * torch.stack(terms, dim=0).sum(dim=0)
        return 0.5 * np.stack([np.asarray(t) for t in terms], axis=0).sum(axis=0)

    def project_feasible(self, p: torch.Tensor, buffer: float = 1e-4) -> torch.Tensor:
        p_t, to_numpy = self._as_torch(p)
        parts = unpack_state(p_t, self.layout)
        meta = self.meta
        vis = parts["vis"].clamp(0.0, 1.0)
        pos = parts["pos"].clone()
        xy = pos[..., :2].clamp(-float(meta.r_xy) + buffer, float(meta.r_xy) - buffer)
        z = pos[..., 2:3].clamp(float(meta.z0) - float(meta.tau_z) + buffer, float(meta.z0) + float(meta.tau_z) - buffer)
        pos = torch.cat([xy, z], dim=-1)
        color = torch.nn.functional.one_hot(parts["color"].argmax(dim=-1), self.layout.n_color).to(p_t.dtype)
        material = torch.nn.functional.one_hot(parts["material"].argmax(dim=-1), self.layout.n_material).to(p_t.dtype)
        shape = torch.nn.functional.one_hot(parts["shape"].argmax(dim=-1), self.layout.n_shape).to(p_t.dtype)
        coll = 0.5 * (parts["coll"] + parts["coll"].transpose(-1, -2))
        coll = coll - torch.diag_embed(coll.diagonal(dim1=-2, dim2=-1))
        def _np(x: torch.Tensor) -> np.ndarray:
            return x.detach().cpu().numpy()

        if p_t.ndim == 1:
            out = pack_parts(_np(color), _np(material), _np(shape), _np(vis), _np(pos), _np(parts["vel"]), _np(coll), self.layout)
            result = torch.from_numpy(out).to(device=p_t.device, dtype=p_t.dtype)
            return result.numpy() if to_numpy else result
        rows = []
        for i in range(p_t.shape[0]):
            rows.append(
                pack_parts(
                    _np(color[i]),
                    _np(material[i]),
                    _np(shape[i]),
                    _np(vis[i]),
                    _np(pos[i]),
                    _np(parts["vel"][i]),
                    _np(coll[i]),
                    self.layout,
                )
            )
        result = torch.from_numpy(np.stack(rows)).to(device=p_t.device, dtype=p_t.dtype)
        return result.numpy() if to_numpy else result

    def project_physical(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        return self.project_feasible(p, buffer=0.0)

    def energy(self, p: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.cost(p)

    def energy_grad(self, p: np.ndarray, **kwargs) -> np.ndarray:
        p_t = torch.from_numpy(np.asarray(p, dtype=np.float32)).requires_grad_(True)
        cost = self.cost(p_t)
        grad = torch.autograd.grad(cost.sum(), p_t)[0]
        return grad.detach().cpu().numpy()

    def progress(self, p: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        p_t, to_numpy = self._as_torch(p)
        vis = unpack_state(p_t, self.layout)["vis"].clamp(0.0, 1.0)
        frac = vis.mean(dim=(-1, -2))
        return frac.detach().cpu().numpy() if to_numpy else frac


def _constraint_meta(layout: CLEVRERStateLayout, cfg: DictConfig, mean: np.ndarray, std: np.ndarray) -> CLEVRERStateMeta:
    data = cfg.data
    fps = float(data.get("fps", 16.0) or 16.0)
    return CLEVRERStateMeta(
        n_slots=layout.n_slots,
        n_frames=layout.n_frames,
        fps=fps,
        dt=1.0 / fps,
        r_xy=float(data.get("r_xy", 12.0)),
        z0=float(data.get("z0", 0.20)),
        tau_z=float(data.get("tau_z", 0.05)),
        v_max=float(data.get("v_max", 3.2)),
        tau_kin=float(data.get("tau_kin", 0.20)),
        tau_acc=float(data.get("tau_acc", 0.60)),
        d0=float(data.get("d0", 0.45)),
        d_min=float(data.get("d_min", 0.38)),
        tau_col=float(data.get("tau_col", 0.20)),
        delta_v=float(data.get("delta_v", 0.15)),
        eps_ident=float(data.get("eps_ident", 0.05)),
        eps_null=float(data.get("eps_null", 1.0e-3)),
        mean=tuple(float(x) for x in mean.tolist()),
        std=tuple(float(x) for x in std.tolist()),
    )


def state_normalization(layout: CLEVRERStateLayout, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale pos/vel channels; leave one-hot, visibility, and collision as-is."""
    mean = np.zeros((layout.dim,), dtype=np.float32)
    std = np.ones((layout.dim,), dtype=np.float32)
    slices = layout.slices()
    packed = np.asarray(states, dtype=np.float32)
    for name in ("pos", "vel"):
        lo, hi = slices[name]
        chunk = packed[:, lo:hi]
        mean[lo:hi] = chunk.mean(axis=0)
        scale = chunk.std(axis=0)
        std[lo:hi] = np.where(scale < 1e-6, 1.0, scale)
    return mean, std


def normalize_state(state: np.ndarray | torch.Tensor, mean: np.ndarray, std: np.ndarray):
    if isinstance(state, torch.Tensor):
        m = torch.as_tensor(mean, device=state.device, dtype=state.dtype)
        s = torch.as_tensor(std, device=state.device, dtype=state.dtype)
        return (state - m) / s
    return (np.asarray(state) - mean) / std


def denormalize_state(state: np.ndarray | torch.Tensor, mean: np.ndarray, std: np.ndarray):
    if isinstance(state, torch.Tensor):
        m = torch.as_tensor(mean, device=state.device, dtype=state.dtype)
        s = torch.as_tensor(std, device=state.device, dtype=state.dtype)
        return state * s + m
    return np.asarray(state) * std + mean


class CLEVRERRecognitionDataset(Dataset):
    """Video clip plus packed GT scene state for recognition CFM."""

    def __init__(
        self,
        videos: CLEVRERDataset,
        layout: CLEVRERStateLayout,
        mean: np.ndarray,
        std: np.ndarray,
        dt: float = 1.0 / 16.0,
    ) -> None:
        self.videos = videos
        self.layout = layout
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.dt = float(dt)

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.videos[index]
        if item["annotation"] is None:
            raise ValueError(f"scene {item['scene_index']} has no annotation")
        state = annotation_to_state(item["annotation"], item["frame_indices"], self.layout, dt=self.dt)
        item["state"] = torch.from_numpy(state)
        item["state_z"] = torch.from_numpy(normalize_state(state, self.mean, self.std))
        return item


def collate_clevrer_recognition(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_clevrer(samples)
    batch["state"] = torch.stack([item["state"] for item in samples])
    batch["state_z"] = torch.stack([item["state_z"] for item in samples])
    return batch


def _collect_states(dataset: CLEVRERDataset, layout: CLEVRERStateLayout, dt: float) -> np.ndarray:
    rows = []
    for _, _video, annotation_path in dataset.records:
        if annotation_path is None:
            continue
        annotation = json.loads(Path(annotation_path).read_text())
        # Sample the same time grid as CLEVRERDataset without decoding MP4.
        dummy = CLEVRERDataset.__new__(CLEVRERDataset)
        dummy.n_frames = dataset.n_frames
        dummy.fps = dataset.fps
        dummy.start_time = dataset.start_time
        dummy.height = dataset.height
        dummy.width = dataset.width
        # Use a light decode of timestamps only via one video? Too slow.
        # Approximate: original 25 fps, indices = nearest to start + arange(n)/fps
        fps = float(dataset.fps or 25.0)
        count = int(dataset.n_frames)
        requested = dataset.start_time + np.arange(count) / fps
        source_fps = 25.0
        source = np.arange(0, 128) / source_fps
        right = np.searchsorted(source, requested).clip(0, 127)
        left = (right - 1).clip(0)
        indices = np.where(np.abs(source[left] - requested) <= np.abs(source[right] - requested), left, right)
        rows.append(annotation_to_state(annotation, indices, layout, dt=dt))
    if not rows:
        raise FileNotFoundError("No annotated CLEVRER scenes to build recognition states")
    return np.stack(rows, axis=0)


@register_dataset("clevrer_recognition")
def build_clevrer_recognition(cfg: DictConfig) -> VideoDataBundle:
    data = cfg.data
    root = Path(str(data.get("cache_dir", "datasets/clevrer"))).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    layout = layout_from_cfg(cfg)
    options = {key: data.get(key, None) for key in ("n_frames", "fps", "height", "width")}
    options["start_time"] = float(data.get("start_time", 0.0))
    options["load_questions"] = False
    train_split = str(data.get("train_split", "train"))
    eval_split = str(data.get("eval_split", "validation"))
    train_videos = CLEVRERDataset(root, train_split, limit=data.get("n_train", None), **options)
    eval_videos = CLEVRERDataset(root, eval_split, limit=data.get("n_eval", None), **options)
    dt = 1.0 / float(options["fps"] or 16.0)
    train_states = _collect_states(train_videos, layout, dt)
    mean, std = state_normalization(layout, train_states)
    meta = _constraint_meta(layout, cfg, mean, std)
    constraint = CLEVRERStateConstraint(meta, layout)
    train = CLEVRERRecognitionDataset(train_videos, layout, mean, std, dt=dt)
    evaluation = CLEVRERRecognitionDataset(eval_videos, layout, mean, std, dt=dt)
    bundle_meta = {
        "dataset": "clevrer_recognition",
        "root": str(root),
        "n_train": len(train),
        "n_eval": len(evaluation),
        "state_dim": layout.dim,
        "n_slots": layout.n_slots,
        "n_frames": layout.n_frames,
        "annotation_coordinates": "original_simulator_world",
        "mean": mean.tolist(),
        "std": std.tolist(),
        **options,
        **{key: getattr(meta, key) for key in ("r_xy", "z0", "tau_z", "v_max", "tau_kin", "d0", "d_min")},
    }
    return VideoDataBundle(train=train, eval=evaluation, meta=bundle_meta, constraint=constraint)
