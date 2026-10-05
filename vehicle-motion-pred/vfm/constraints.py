from __future__ import annotations

import torch
import torch.nn.functional as F

ALL = ("speed", "continuity", "accel", "drivable", "static", "lane", "drivable_fp", "coll",
       "goal", "waypoint")


class NuScenesConstraint:

    def __init__(self, batch: dict[str, torch.Tensor], meta: dict, cfg: dict):
        self.cfg = cfg
        hist = batch["hist"]
        self.dtype, self.device = hist.dtype, hist.device
        self.dt = 1.0 / float(meta["sample_hz"])
        self.prev = hist[:, -2]
        self.cont_valid = batch["hist_mask"][:, -2]
        self.v_max = float(cfg.get("v_max", 25.0))
        self.a_max = float(cfg.get("a_max", 6.0))
        self.a_cont = float(cfg.get("a_cont", self.a_max))
        self.da_margin = float(cfg.get("drivable_margin", 0.5))
        self.st_margin = float(cfg.get("static_margin", 0.0))
        self.half_w = float(cfg.get("lane_half_width", 2.0))
        self.hard = tuple(cfg.get("hard", ("speed", "continuity", "accel", "drivable", "static")))
        self.w_hard = float(cfg.get("w_hard", 0.0))
        self.w_lane = float(cfg.get("w_lane", 0.1))
        self.sweeps = int(cfg.get("proj_sweeps", 10))
        self.inner = int(cfg.get("proj_inner", 4))
        self.relax = float(cfg.get("proj_relax", 1.8))
        self.tail = int(cfg.get("proj_tail", 3))

        self.sdf = None
        if "sdf" in batch and "sdf" in meta:
            g = meta["sdf"]
            self.grid = (float(g["x_min"]), float(g["y_min"]), float(g["res"]) * int(g["size"]))
            self.sdf = (batch["sdf"].to(self.dtype) * float(g["scale"]))[:, None]

        self.obs = batch.get("obs")
        self.obs_mask = batch.get("obs_mask")
        width = batch["focal_size"][:, 1] if "focal_size" in batch else torch.full_like(hist[:, 0, 0], 1.9)
        length = batch["focal_size"][:, 0] if "focal_size" in batch else torch.full_like(hist[:, 0, 0], 4.5)
        self.focal_w, self.focal_l = width, length
        self.r_focal = 0.5 * width + self.st_margin
        self.fp_margin = float(cfg.get("fp_margin", 0.5))
        self.coll_margin = float(cfg.get("coll_margin", 0.0))
        self.coll_horizon = float(cfg.get("coll_horizon_s", 6.0))

        nbr, nmask = batch["nbr"], batch["nbr_mask"]
        last = nbr[:, :, -1]
        prev = nbr[:, :, -2] if nbr.shape[2] > 1 else last
        both = nmask[:, :, -1] & (nmask[:, :, -2] if nbr.shape[2] > 1 else nmask[:, :, -1])
        vel = torch.where(both[..., None], (last - prev) / self.dt, torch.zeros_like(last))
        self.nbr_valid = nmask[:, :, -1]
        self.nbr_last, self.nbr_vel = last, vel
        if "nbr_size" in batch:
            self.nbr_l, self.nbr_w = batch["nbr_size"][..., 0], batch["nbr_size"][..., 1]
        else:
            self.nbr_l = torch.full_like(last[..., 0], 4.5)
            self.nbr_w = torch.full_like(last[..., 0], 1.9)
        if "nbr_yaw" in batch:
            self.nbr_yaw = batch["nbr_yaw"]
        else:
            self.nbr_yaw = torch.atan2(vel[..., 1], vel[..., 0])
        b0 = hist.shape[0]
        origin = torch.zeros(b0, 1, 1, 2, dtype=hist.dtype, device=hist.device)
        fwd0 = torch.zeros_like(origin)
        fwd0[..., 0] = 1.0
        self.fp_margin_b = torch.full((b0,), self.fp_margin, dtype=hist.dtype, device=hist.device)
        if self.sdf is not None:
            v0, _ = self._fp_sdf(origin, fwd0)
            v0 = v0.amax(dim=(1, 2, 3))
            self.fp_margin_b = torch.maximum(self.fp_margin_b, v0 + float(cfg.get("fp_slack", 0.1)))
        self.goal_r = float(cfg.get("goal_radius", 1.0))
        self.targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        fut, fmask = batch.get("fut"), batch.get("fut_mask")
        if fut is not None and fmask is not None and ({"goal", "waypoint"} & set(self.hard)):
            T = fut.shape[1]
            last = T - 1 - fmask.flip(-1).float().argmax(-1)
            ar = torch.arange(fut.shape[0], device=fut.device)
            g = torch.Generator().manual_seed(int(cfg.get("goal_seed", 0)))
            sigma = float(cfg.get("goal_noise", 0.0))
            def _tgt(idx):
                x = fut[ar, idx]
                if sigma > 0:
                    x = x + sigma * torch.randn(x.shape, generator=g).to(x.device, x.dtype)
                return idx, x
            if "goal" in self.hard:
                self.targets["goal"] = _tgt(last)
                if str(cfg.get("goal_source", "gt")) != "gt":
                    self.targets["goal"] = (last, torch.zeros_like(fut[ar, last]))
            if "waypoint" in self.hard:
                j = int(round(float(cfg.get("waypoint_s", 3.0)) / self.dt)) - 1
                self.targets["waypoint"] = _tgt(torch.clamp(last, max=max(j, 0)))

        self.nbr_dropped = torch.zeros_like(self.nbr_valid)
        if self.nbr_valid.shape[1]:
            fc, fr = self._circles(origin[:, 0], fwd0[:, 0], self.focal_l[:, None].to(hist.dtype), self.focal_w[:, None].to(hist.dtype))
            nf = torch.stack([torch.cos(self.nbr_yaw), torch.sin(self.nbr_yaw)], -1)
            nc, nr = self._circles(self.nbr_last, nf, self.nbr_l, self.nbr_w)
            d0 = (fc[:, :, :, None, :] - nc[:, :, None, :, :]).norm(dim=-1)
            over0 = (fr[:, :, None, None] + nr[:, :, None, None] - d0).amax(dim=(2, 3))
            self.nbr_dropped = self.nbr_valid & (over0 > 0)
            self.nbr_valid = self.nbr_valid & ~self.nbr_dropped

        lane, lmask = batch["lane"], batch["lane_mask"]
        b = lane.shape[0]
        self.seg_a = lane[:, :, :-1].reshape(b, -1, 2)
        self.seg_b = lane[:, :, 1:].reshape(b, -1, 2)
        self.seg_valid = (lmask[:, :, :-1] & lmask[:, :, 1:]).reshape(b, -1)
        self.has_lane = self.seg_valid.any(dim=1)

    def _seq(self, p: torch.Tensor) -> torch.Tensor:
        b, k = p.shape[:2]
        prev = self.prev[:, None, None].expand(b, k, 1, 2).to(p.dtype)
        zero = torch.zeros_like(prev)
        return torch.cat([prev, zero, p], dim=2)

    def sdf_at(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0, y0, ext = self.grid
        gx = 2.0 * (p[..., 0] - x0) / ext - 1.0
        gy = 2.0 * (p[..., 1] - y0) / ext - 1.0
        b = p.shape[0]
        grid = torch.stack([gx, gy], dim=-1).reshape(b, -1, 1, 2)
        v = F.grid_sample(self.sdf.to(p.dtype), grid, mode="bilinear", padding_mode="border", align_corners=False)
        inside = (gx.abs() <= 1.0) & (gy.abs() <= 1.0)
        return v.reshape(p.shape[:-1]), inside

    def _box(self, p: torch.Tensor):
        o = self.obs[:, None, None]
        c, s = torch.cos(o[..., 2]), torch.sin(o[..., 2])
        d = p[..., None, :] - o[..., :2]
        lx = d[..., 0] * c + d[..., 1] * s
        ly = -d[..., 0] * s + d[..., 1] * c
        qx = lx.abs() - 0.5 * o[..., 3]
        qy = ly.abs() - 0.5 * o[..., 4]
        ox, oy = qx.clamp_min(0), qy.clamp_min(0)
        out = torch.sqrt(ox * ox + oy * oy + 1e-12)
        is_out = (qx > 0) | (qy > 0)
        sd = torch.where(is_out, out, torch.maximum(qx, qy))
        sx = torch.where(lx >= 0, 1.0, -1.0).to(p.dtype)
        sy = torch.where(ly >= 0, 1.0, -1.0).to(p.dtype)
        nx = torch.where(is_out, sx * ox / out, torch.where(qx > qy, sx, torch.zeros_like(sx)))
        ny = torch.where(is_out, sy * oy / out, torch.where(qx > qy, torch.zeros_like(sy), sy))
        n = torch.stack([nx * c - ny * s, nx * s + ny * c], dim=-1)
        return sd, n

    def _nearest_lane(self, p: torch.Tensor):
        a = self.seg_a[:, None, None]
        bb = self.seg_b[:, None, None]
        ab = bb - a
        ap = p[..., None, :] - a
        u = ((ap * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
        c = a + u[..., None] * ab
        d = (p[..., None, :] - c).norm(dim=-1)
        d = d.masked_fill(~self.seg_valid[:, None, None], float("inf"))
        dmin, idx = d.min(dim=-1)
        cmin = torch.gather(c, -2, idx[..., None, None].expand(*idx.shape, 1, 2)).squeeze(-2)
        return dmin, cmin

    def heading(self, p: torch.Tensor, min_move: float = 0.2) -> torch.Tensor:
        b, k, t = p.shape[:3]
        q = torch.cat([torch.zeros_like(p[:, :, :1]), p], dim=2)
        d = q[:, :, 1:] - q[:, :, :-1]
        moving = d.norm(dim=-1) > min_move
        head = torch.zeros(b, k, 2, dtype=p.dtype, device=p.device)
        head[..., 0] = 1.0
        hs = []
        for i in range(t):
            u = d[:, :, i] / d[:, :, i].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            head = torch.where(moving[:, :, i, None], u, head)
            hs.append(head)
        return torch.stack(hs, dim=2)

    def _corners(self, p: torch.Tensor, fwd: torch.Tensor) -> torch.Tensor:
        left = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
        hl = (0.5 * self.focal_l)[:, None, None, None].to(p.dtype)
        hw = (0.5 * self.focal_w)[:, None, None, None].to(p.dtype)
        return torch.stack([p + fwd * hl + left * hw, p + fwd * hl - left * hw,
                            p - fwd * hl + left * hw, p - fwd * hl - left * hw], dim=3)

    def _fp_sdf(self, p: torch.Tensor, fwd: torch.Tensor):
        b, k, t = p.shape[:3]
        cor = self._corners(p, fwd)
        v, inside = self.sdf_at(cor.reshape(b, k, t * 4, 2))
        v = torch.where(inside, v, torch.full_like(v, -1e3)).reshape(b, k, t, 4)
        return v, cor

    def _circles(self, c: torch.Tensor, fwd: torch.Tensor, length: torch.Tensor, width: torch.Tensor):
        off = (0.5 * (length - width)).clamp_min(0.0)[..., None, None] * torch.tensor(
            [-1.0, 0.0, 1.0], dtype=c.dtype, device=c.device)[:, None]
        return c[..., None, :] + off * fwd[..., None, :], 0.5 * width

    def _coll_pairs(self, p: torch.Tensor, fwd: torch.Tensor):
        b, k, t = p.shape[:3]
        steps = torch.arange(1, t + 1, dtype=p.dtype, device=p.device) * self.dt
        npos = self.nbr_last[:, :, None] + self.nbr_vel[:, :, None] * steps[:, None]
        nf = torch.stack([torch.cos(self.nbr_yaw), torch.sin(self.nbr_yaw)], -1)[:, :, None].expand_as(npos)
        nc, nr = self._circles(npos, nf, self.nbr_l[:, :, None].expand(npos.shape[:3]), self.nbr_w[:, :, None].expand(npos.shape[:3]))
        fc, fr = self._circles(p, fwd, self.focal_l[:, None, None].expand(p.shape[:3]).to(p.dtype),
                               self.focal_w[:, None, None].expand(p.shape[:3]).to(p.dtype))
        diff = fc[:, :, None, :, :, None, :] - nc[:, None, :, :, None, :, :]
        dist = diff.norm(dim=-1)
        r = fr[:, :, None, :, None, None] + nr[:, None, :, :, None, None] + self.coll_margin
        h = r - dist
        valid = self.nbr_valid[:, None, :, None, None, None] & (steps <= self.coll_horizon + 1e-6)[None, None, None, :, None, None]
        h = h.masked_fill(~valid, -1e3)
        return h, diff, dist

    def h_steps(self, p: torch.Tensor, extras: bool = True) -> dict[str, torch.Tensor]:
        b, k, t = p.shape[:3]
        q = self._seq(p)
        out = {}
        vel = q[:, :, 2:] - q[:, :, 1:-1]
        out["speed"] = vel.norm(dim=-1) / self.dt - self.v_max
        acc = (q[:, :, 2:] - 2 * q[:, :, 1:-1] + q[:, :, :-2]).norm(dim=-1) / self.dt ** 2
        cont = acc[..., :1] - self.a_cont
        out["continuity"] = torch.where(self.cont_valid[:, None, None], cont, torch.full_like(cont, -1.0))
        out["accel"] = acc[..., 1:] - self.a_max if t > 1 else acc[..., :0]
        if self.sdf is not None:
            v, inside = self.sdf_at(p)
            out["drivable"] = torch.where(inside, v - self.da_margin, torch.full_like(v, -1.0))
        if self.obs is not None:
            sd, _ = self._box(p)
            hs = self.r_focal[:, None, None, None] - sd
            hs = hs.masked_fill(~self.obs_mask[:, None, None], -1.0)
            out["static"] = hs.max(dim=-1).values if hs.shape[-1] else torch.full(p.shape[:3], -1.0, dtype=p.dtype, device=p.device)
        dl, _ = self._nearest_lane(p)
        hl = dl - self.half_w
        out["lane"] = torch.where(self.has_lane[:, None, None] & torch.isfinite(hl), hl, torch.full_like(hl, -1.0))
        for name, (idx, tgt) in self.targets.items():
            ar = torch.arange(p.shape[0], device=p.device)
            t = tgt[:, None] if tgt.dim() == 2 else tgt
            out[name] = ((p[ar, :, idx] - t).norm(dim=-1) - self.goal_r)[..., None]
        if extras or "drivable_fp" in self.hard or "coll" in self.hard:
            fwd = self.heading(p)
            if self.sdf is not None:
                v, _ = self._fp_sdf(p, fwd)
                out["drivable_fp"] = (v.max(dim=-1).values - self.fp_margin_b[:, None, None]).clamp_min(-1.0)
            if self.nbr_valid.shape[1]:
                h, _, _ = self._coll_pairs(p, fwd)
                out["coll"] = h.amax(dim=(2, 4, 5)).clamp_min(-1.0)
        return out

    def h(self, p: torch.Tensor, step_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        hs = self.h_steps(p)
        if step_mask is not None:
            m = step_mask[:, None].expand(p.shape[:3])
            prev = torch.cat([torch.ones_like(m[..., :1]), m[..., :-1]], dim=-1)
            prev2 = torch.cat([torch.ones_like(m[..., :1]), prev[..., :-1]], dim=-1)
            masks = {"speed": m & prev, "continuity": m[..., :1], "accel": (m & prev & prev2)[..., 1:],
                     "drivable": m, "static": m, "lane": m, "drivable_fp": m, "coll": m}
            hs = {n: v.masked_fill(~masks.get(n, torch.ones_like(v, dtype=torch.bool)), -1.0)
                  for n, v in hs.items()}
        return {n: (v.max(dim=-1).values if v.shape[-1] else torch.full(p.shape[:2], -1.0, device=p.device, dtype=p.dtype))
                for n, v in hs.items()}

    def cost(self, p: torch.Tensor) -> torch.Tensor:
        hs = self.h_steps(p, extras=False)
        c = torch.zeros(p.shape[:2], dtype=p.dtype, device=p.device)
        for n in self.hard:
            if n in hs:
                c = c + self.w_hard * 0.5 * hs[n].clamp_min(0).square().sum(-1)
        if self.w_lane > 0 and "lane" not in self.hard:
            c = c + self.w_lane * 0.5 * hs["lane"].clamp_min(0).square().sum(-1)
        return c

    def _proj_linear(self, q, starts, coef, r, free, relax=1.0):
        b, k = q.shape[:2]
        m = len(coef)
        idx = starts[:, None] + torch.arange(m, device=q.device)
        x = q[:, :, idx]
        cf = torch.tensor(coef, dtype=q.dtype, device=q.device)
        e = (cf[:, None] * x).sum(dim=-2)
        n = e.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        delta = relax * e * (1.0 - r / n).clamp_min(0.0)
        w = cf[None] * free[idx].to(q.dtype)
        denom = (w * w).sum(-1).clamp_min(1e-9)
        upd = -w[None, None, :, :, None] * delta[:, :, :, None, :] / denom[None, None, :, None, None]
        return q.index_add(2, idx.reshape(-1), upd.reshape(b, k, -1, 2))

    def _proj_kinematic(self, q: torch.Tensor, relax: float = 1.0) -> torch.Tensor:
        n = q.shape[2]
        t = n - 2
        free = torch.ones(n, dtype=torch.bool, device=q.device)
        free[:2] = False
        dt2 = self.dt ** 2
        if "speed" in self.hard:
            for g in range(2):
                starts = torch.arange(1 + g, n - 1, 2, device=q.device)
                if len(starts):
                    q = self._proj_linear(q, starts, (-1.0, 1.0), self.v_max * self.dt, free, relax)
        use_c, use_a = "continuity" in self.hard, "accel" in self.hard
        for g in range(3):
            starts = torch.arange(g, t, 3, device=q.device)
            if not len(starts):
                continue
            r = torch.full((q.shape[0], 1, len(starts), 1), self.a_max * dt2 if use_a else float("inf"),
                           dtype=q.dtype, device=q.device)
            if g == 0:
                rc = self.a_cont * dt2 if use_c else float("inf")
                r[:, 0, 0, 0] = torch.where(self.cont_valid, torch.full_like(r[:, 0, 0, 0], rc),
                                            torch.full_like(r[:, 0, 0, 0], float("inf")))
            q = self._proj_linear(q, starts, (1.0, -2.0, 1.0), r, free, relax)
        return q

    def _forward_clamp(self, p: torch.Tensor, buffer: float) -> torch.Tensor:
        b, k, t = p.shape[:3]
        use_s, use_a, use_c = "speed" in self.hard, "accel" in self.hard, "continuity" in self.hard
        prev = self.prev[:, None].expand(b, k, 2).to(p.dtype)
        v_prev = torch.where(self.cont_valid[:, None, None], -prev / self.dt, torch.zeros_like(prev))
        pos = torch.zeros_like(prev)
        out = []
        for i in range(t):
            v_des = (p[:, :, i] - pos) / self.dt
            a = (v_des - v_prev) / self.dt
            lim = None
            if i == 0 and use_c:
                lim = torch.where(self.cont_valid[:, None], torch.full((b, 1), self.a_cont, dtype=p.dtype, device=p.device),
                                  torch.full((b, 1), float("inf"), dtype=p.dtype, device=p.device)).expand(b, k)
                if use_a:
                    lim = torch.minimum(lim, torch.full_like(lim, self.a_max))
            elif i > 0 and use_a:
                lim = torch.full((b, k), self.a_max, dtype=p.dtype, device=p.device)
            if lim is not None:
                n = a.norm(dim=-1)
                a = a * ((lim - buffer).clamp_min(0) / n.clamp_min(1e-9)).clamp(max=1.0)[..., None]
            v = v_prev + a * self.dt
            if use_s:
                n = v.norm(dim=-1)
                v = v * ((self.v_max - buffer) / n.clamp_min(1e-9)).clamp(max=1.0)[..., None]
            pos = pos + v * self.dt
            out.append(pos)
            v_prev = v
        return torch.stack(out, dim=2)

    def _proj_drivable(self, p: torch.Tensor, buffer: float) -> torch.Tensor:
        with torch.enable_grad():
            x = p.detach().requires_grad_(True)
            v, inside = self.sdf_at(x)
            g = torch.autograd.grad(v.sum(), x)[0]
        gn = g.norm(dim=-1, keepdim=True)
        viol = (v.detach() - self.da_margin + buffer).clamp_min(0.0) * inside
        ok = (gn[..., 0] > 0.5) & (viol > 0)
        step = viol[..., None] * g / gn.square().clamp_min(1e-6)
        return torch.where(ok[..., None], p - step, p)

    def _proj_static(self, p: torch.Tensor, buffer: float) -> torch.Tensor:
        sd, n = self._box(p)
        viol = (self.r_focal[:, None, None, None] + buffer - sd).masked_fill(~self.obs_mask[:, None, None], -1.0)
        if not viol.shape[-1]:
            return p
        vmax, j = viol.max(dim=-1)
        nj = torch.gather(n, -2, j[..., None, None].expand(*j.shape, 1, 2)).squeeze(-2)
        return p + vmax.clamp_min(0.0)[..., None] * nj

    def _proj_drivable_fp(self, p: torch.Tensor, buffer: float) -> torch.Tensor:
        fwd = self.heading(p.detach())
        with torch.enable_grad():
            x = p.detach().requires_grad_(True)
            v, _ = self._fp_sdf(x, fwd)
            worst = v.max(dim=-1).values
            g = torch.autograd.grad(worst.sum(), x)[0]
        gn = g.norm(dim=-1, keepdim=True)
        viol = (worst.detach() - self.fp_margin_b[:, None, None] + buffer).clamp_min(0.0)
        ok = (gn[..., 0] > 0.5) & (viol > 0) & (worst.detach() > -1e2)
        step = viol[..., None] * g / gn.square().clamp_min(1e-6)
        return torch.where(ok[..., None], p - step, p)

    @staticmethod
    def _along(pts: torch.Tensor, cum: torch.Tensor, target: torch.Tensor):
        seg = pts[..., 1:, :] - pts[..., :-1, :]
        seglen = seg.norm(dim=-1)
        target = torch.minimum(target.clamp_min(0.0), cum[..., -1:])
        idx = torch.searchsorted(cum[..., 1:].contiguous(), target.contiguous()).clamp(max=seg.shape[-2] - 1)
        g = lambda x: torch.gather(x, -2, idx[..., None].expand(*idx.shape, 2))
        base, sg = g(pts[..., :-1, :]), g(seg)
        l0 = torch.gather(cum[..., :-1], -1, idx)
        ln = torch.gather(seglen, -1, idx)
        frac = ((target - l0) / ln.clamp_min(1e-9)).clamp(0.0, 1.0)
        pos = base + frac[..., None] * sg
        fwd = torch.where((ln > 1e-3)[..., None], sg / ln.clamp_min(1e-9)[..., None],
                          torch.tensor([1.0, 0.0], dtype=pts.dtype, device=pts.device).expand_as(sg))
        return pos, fwd

    def _proj_coll(self, p: torch.Tensor, buffer: float, iters: int = 14) -> torch.Tensor:
        if not self.nbr_valid.shape[1]:
            return p
        hk = self._coll_pairs(p, self.heading(p))[0].amax(dim=(2, 4, 5))
        viol = hk > 0
        if not viol.any():
            return p
        pts = torch.cat([torch.zeros_like(p[:, :, :1]), p], dim=2)
        cum = torch.cat([torch.zeros_like(p[..., :1, 0]), (pts[:, :, 1:] - pts[:, :, :-1]).norm(dim=-1).cumsum(-1)], -1)
        s_now = cum[..., 1:]
        lo, hi = torch.zeros_like(s_now), s_now.clone()
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            pos, fwd = self._along(pts, cum, mid)
            ok = self._coll_pairs(pos, fwd)[0].amax(dim=(2, 4, 5)) <= -buffer
            lo = torch.where(ok, mid, lo)
            hi = torch.where(ok, hi, mid)
        cap = torch.where(viol, lo, s_now)
        cap = torch.flip(torch.cummin(torch.flip(torch.minimum(s_now, cap), [-1]), -1).values, [-1])
        d = torch.diff(cap, dim=-1, prepend=torch.zeros_like(cap[..., :1]))
        if "accel" in self.hard:
            lim = self.a_max * self.dt ** 2
            ds = list(d.unbind(-1))
            for i in range(len(ds) - 2, -1, -1):
                ds[i] = torch.minimum(ds[i], ds[i + 1] + lim)
            d = torch.stack(ds, -1)
        s_new = d.clamp_min(0.0).cumsum(-1)
        pos, _ = self._along(pts, cum, s_new)
        return pos

    def set_target(self, name: str, tgt: torch.Tensor, step: int | None = None) -> None:
        if name in self.targets and step is None:
            idx = self.targets[name][0]
        else:
            idx = torch.full((tgt.shape[0],), int(step if step is not None else -1), dtype=torch.long,
                             device=tgt.device)
        self.targets[name] = (idx, tgt)

    def _proj_target(self, p: torch.Tensor, buffer: float) -> torch.Tensor:
        ar = torch.arange(p.shape[0], device=p.device)
        r = max(self.goal_r - buffer, 0.0)
        for name, (idx, tgt) in self.targets.items():
            if name not in self.hard:
                continue
            t = tgt[:, None] if tgt.dim() == 2 else tgt
            x = p[ar, :, idx]
            d = x - t
            n = d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            p = p.clone()
            p[ar, :, idx] = t + d * (r / n).clamp(max=1.0)
        return p

    def project_feasible(self, p: torch.Tensor, buffer: float = 1e-3, sweeps: int | None = None) -> torch.Tensor:
        n = int(self.sweeps if sweeps is None else sweeps)
        kin = "speed" in self.hard or "accel" in self.hard or "continuity" in self.hard
        for i in range(n):
            if "drivable" in self.hard and self.sdf is not None:
                p = self._proj_drivable(p, buffer)
            if "drivable_fp" in self.hard and self.sdf is not None:
                p = self._proj_drivable_fp(p, buffer)
            if "static" in self.hard and self.obs is not None:
                p = self._proj_static(p, buffer)
            if kin:
                q = self._seq(p)
                for _ in range(self.inner):
                    q = self._proj_kinematic(q, self.relax if i < n - self.tail else 1.0)
                p = q[:, :, 2:]
            if "coll" in self.hard:
                p = self._proj_coll(p, buffer)
            if kin:
                p = self._forward_clamp(p, buffer)
            if self.targets:
                p = self._proj_target(p, buffer)
        return p

    def footprint_offroad(self, p: torch.Tensor, length: torch.Tensor, width: torch.Tensor,
                          min_move: float = 0.2) -> torch.Tensor:
        b, k, t = p.shape[:3]
        q = torch.cat([torch.zeros_like(p[:, :, :1]), p], dim=2)
        d = q[:, :, 1:] - q[:, :, :-1]
        moving = d.norm(dim=-1) > min_move
        head = torch.zeros(b, k, 2, dtype=p.dtype, device=p.device)
        head[..., 0] = 1.0
        hs = []
        for i in range(t):
            u = d[:, :, i] / d[:, :, i].norm(dim=-1, keepdim=True).clamp_min(1e-9)
            head = torch.where(moving[:, :, i, None], u, head)
            hs.append(head)
        fwd = torch.stack(hs, dim=2)
        left = torch.stack([-fwd[..., 1], fwd[..., 0]], dim=-1)
        hl = (0.5 * length)[:, None, None, None]
        hw = (0.5 * width)[:, None, None, None]
        corners = torch.stack([p + fwd * hl + left * hw, p + fwd * hl - left * hw,
                               p - fwd * hl + left * hw, p - fwd * hl - left * hw], dim=3)
        v, inside = self.sdf_at(corners.reshape(b, k, t * 4, 2))
        v = torch.where(inside, v, torch.full_like(v, -1e3))
        return v.max(dim=-1).values

    def project_physical(self, p: torch.Tensor) -> torch.Tensor:
        d, c = self._nearest_lane(p)
        far = torch.isfinite(d) & (d > self.half_w) & self.has_lane[:, None, None]
        scale = (self.half_w / d.clamp_min(1e-9))[..., None]
        return torch.where(far[..., None], c + (p - c) * scale, p)

    def estimate_lipschitz(self, p: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        base = self.project_physical(p)
        best = torch.zeros(p.shape[:2], dtype=p.dtype, device=p.device)
        for d in ((eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps)):
            shift = torch.tensor(d, dtype=p.dtype, device=p.device)
            r = (self.project_physical(p + shift) - base).norm(dim=-1) / eps
            best = torch.maximum(best, r.max(dim=-1).values)
        return best


def violation_summary(h: dict[str, torch.Tensor], hard: tuple[str, ...], tol: float = 1e-3) -> dict:
    out = {}
    worst = None
    for n, v in h.items():
        bad = v > tol
        out[f"viol_rate_{n}"] = float(bad.float().mean())
        out[f"scene_viol_rate_{n}"] = float(bad.any(dim=-1).float().mean()) if bad.dim() > 1 else out[f"viol_rate_{n}"]
        out[f"viol_excess_mean_{n}"] = float(v.clamp_min(0).mean())
        if n in hard:
            worst = v if worst is None else torch.maximum(worst, v)
    if worst is not None:
        out["all_hard_safe"] = float((worst <= tol).float().mean())
        out["scene_all_hard_safe"] = float((worst <= tol).all(dim=-1).float().mean()) if worst.dim() > 1 else out["all_hard_safe"]
    return out
