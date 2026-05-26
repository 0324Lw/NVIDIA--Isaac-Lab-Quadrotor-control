from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from quadrotor_rl.tasks.task4.task4_config import Task4Config


class QuadrotorTask4World:
    """Torch-native analytic visual racing world for Task4.

    This class replaces the old PyBullet gate-pool and OpenGL depth renderer
    with a batched analytic world layer.

    Responsibilities:
        - fixed start pose
        - procedural five-gate racing track
        - gate orientation aligned with track tangent
        - random aggressive gate roll / pitch perturbations
        - analytic 64x64 depth camera
        - gate pass / gate collision / arena bounds checks

    It deliberately does not launch Isaac Sim and does not create Isaac prims.
    Rendering gates as real USD objects can be added later as an optional
    visualization layer, while RL can train directly on these tensors.
    """

    def __init__(
        self,
        cfg: Task4Config,
        num_envs: Optional[int] = None,
        device: Optional[str] = None,
    ):
        cfg.validate()
        self.cfg = cfg

        self.num_envs = int(num_envs if num_envs is not None else cfg.num_envs)
        self.device = str(device if device is not None else cfg.device)

        torch.manual_seed(int(cfg.seed))

        n = self.num_envs
        g = int(cfg.num_gates)

        self.start_pos = torch.tensor(
            cfg.start_pos,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3).repeat(n, 1)

        self.gate_pos = torch.zeros((n, g, 3), dtype=torch.float32, device=self.device)
        self.gate_quat = torch.zeros((n, g, 4), dtype=torch.float32, device=self.device)
        self.gate_quat[..., 0] = 1.0
        self.gate_rot = torch.eye(3, dtype=torch.float32, device=self.device).view(1, 1, 3, 3).repeat(n, g, 1, 1)
        self.gate_tangent = torch.zeros((n, g, 3), dtype=torch.float32, device=self.device)
        self.gate_tangent[..., 0] = 1.0
        self.gate_valid = torch.ones((n, g), dtype=torch.bool, device=self.device)

        self.target_gate_idx = torch.zeros(n, dtype=torch.long, device=self.device)

        self.centerline = torch.zeros(
            (n, int(cfg.centerline_samples), 3),
            dtype=torch.float32,
            device=self.device,
        )

        self.pixel_dirs_body = self._build_camera_rays_body()

        self.last_depth = torch.ones(
            (n, 1, int(cfg.cam_res_h), int(cfg.cam_res_w)),
            dtype=torch.float32,
            device=self.device,
        )

        self.reset()

    # ------------------------------------------------------------------
    # Reset / procedural track generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return self.start_pos, self.gate_pos, self.gate_quat

        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            self._generate_track_one(env_id)

        self.target_gate_idx[env_ids] = 0
        self.last_depth[env_ids] = 1.0

        return (
            self.start_pos[env_ids].clone(),
            self.get_gate_pose_list(env_ids),
        )

    def _generate_track_one(self, env_id: int) -> None:
        cfg = self.cfg
        g = int(cfg.num_gates)

        x_values = torch.linspace(
            float(cfg.gate_start_x),
            float(cfg.gate_end_x),
            g,
            dtype=torch.float32,
            device=self.device,
        )

        y_values = self._rand_range((g,), cfg.gate_y_range)
        z_values = self._rand_range((g,), cfg.gate_z_range)

        points = torch.stack([x_values, y_values, z_values], dim=-1)
        self.gate_pos[env_id] = points
        self.gate_valid[env_id] = True

        for i in range(g):
            if i < g - 1:
                tangent = points[i + 1] - points[i]
            else:
                tangent = points[i] - points[i - 1] if g > 1 else torch.tensor([1.0, 0.0, 0.0], device=self.device)

            tangent = self._normalize(tangent)

            base_rot = self._rotation_from_local_z_to_vector(tangent)
            base_quat = self._matrix_to_quat(base_rot)

            local_x_world = base_rot[:, 0]
            local_z_world = base_rot[:, 2]

            roll = math.radians(self._rand_scalar(-cfg.max_roll_pitch_deg, cfg.max_roll_pitch_deg))
            pitch = math.radians(self._rand_scalar(-cfg.max_pitch_offset_deg, cfg.max_pitch_offset_deg))

            q_roll = self._axis_angle_quat(local_z_world, roll)
            q_pitch = self._axis_angle_quat(local_x_world, pitch)

            quat = self._quat_multiply(q_roll, self._quat_multiply(q_pitch, base_quat))
            quat = self._normalize_quat(quat)
            rot = self._quat_to_matrix_single(quat)

            self.gate_tangent[env_id, i] = self._normalize(rot[:, 2])
            self.gate_quat[env_id, i] = quat
            self.gate_rot[env_id, i] = rot

        self.centerline[env_id] = self._build_centerline_from_gates(env_id)

    def _build_centerline_from_gates(self, env_id: int) -> torch.Tensor:
        cfg = self.cfg

        pts = torch.cat(
            [
                self.start_pos[env_id].view(1, 3),
                self.gate_pos[env_id],
            ],
            dim=0,
        )

        samples = int(cfg.centerline_samples)
        if pts.shape[0] == 1:
            return pts.repeat(samples, 1)

        seg = pts[1:] - pts[:-1]
        seg_len = torch.norm(seg, dim=-1)
        cumulative = torch.cat(
            [
                torch.zeros(1, dtype=torch.float32, device=self.device),
                torch.cumsum(seg_len, dim=0),
            ],
            dim=0,
        )
        total = torch.clamp(cumulative[-1], min=1.0e-6)
        s = torch.linspace(0.0, float(total.item()), samples, dtype=torch.float32, device=self.device)

        out = torch.zeros((samples, 3), dtype=torch.float32, device=self.device)

        for k in range(samples):
            sk = s[k]
            idx = torch.searchsorted(cumulative, sk, right=True).item() - 1
            idx = int(max(0, min(idx, pts.shape[0] - 2)))

            denom = torch.clamp(cumulative[idx + 1] - cumulative[idx], min=1.0e-6)
            alpha = (sk - cumulative[idx]) / denom
            out[k] = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]

        return out

    def _rand_scalar(self, lo: float, hi: float) -> float:
        return float(torch.empty((), dtype=torch.float32, device=self.device).uniform_(float(lo), float(hi)).item())

    def _rand_range(self, shape, range_tuple) -> torch.Tensor:
        lo, hi = range_tuple
        return float(lo) + (float(hi) - float(lo)) * torch.rand(shape, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # Analytic depth vision
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_depth_vision(self, drone_pos: torch.Tensor, drone_quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Return analytic normalized depth image.

        Args:
            drone_pos: [num_envs, 3] or [3], local position in arena frame.
            drone_quat_wxyz: [num_envs, 4] or [4], body orientation in wxyz.

        Returns:
            depth: [num_envs, 1, 64, 64], values in [0, 1].
                   0 means near, 1 means >= camera far distance.
        """

        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        quat = torch.as_tensor(drone_quat_wxyz, dtype=torch.float32, device=self.device)

        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)
        if quat.ndim == 1:
            quat = quat.view(1, 4).repeat(self.num_envs, 1)

        assert pos.shape == (self.num_envs, 3), f"drone_pos must be {(self.num_envs, 3)}, got {tuple(pos.shape)}"
        assert quat.shape == (self.num_envs, 4), f"drone_quat_wxyz must be {(self.num_envs, 4)}, got {tuple(quat.shape)}"

        quat = self._normalize_quat_batch(quat)

        h = int(self.cfg.cam_res_h)
        w = int(self.cfg.cam_res_w)
        r = h * w

        dirs_body = self.pixel_dirs_body.view(1, r, 3).repeat(self.num_envs, 1, 1)
        dirs_world = self._quat_rotate_batch(quat[:, None, :].expand(-1, r, -1), dirs_body)
        dirs_world = self._normalize_batch(dirs_world)

        origins = pos[:, None, :].expand(-1, r, -1)

        wall_dist = self._ray_arena_box_distance(origins, dirs_world)
        gate_dist = self._ray_gate_frame_distance(origins, dirs_world)

        dist = torch.minimum(wall_dist, gate_dist)
        dist = torch.clamp(dist, float(self.cfg.cam_near), float(self.cfg.cam_far))

        depth = torch.clamp(dist / float(self.cfg.cam_far), 0.0, 1.0)
        depth = torch.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
        depth = depth.view(self.num_envs, 1, h, w)

        self.last_depth = depth.clone()
        return depth

    def _build_camera_rays_body(self) -> torch.Tensor:
        cfg = self.cfg

        h = int(cfg.cam_res_h)
        w = int(cfg.cam_res_w)

        fov = math.radians(float(cfg.cam_fov_deg))
        tan_half = math.tan(0.5 * fov)

        ys = torch.linspace(-1.0, 1.0, w, dtype=torch.float32, device=self.device)
        zs = torch.linspace(1.0, -1.0, h, dtype=torch.float32, device=self.device)

        grid_y, grid_z = torch.meshgrid(ys, zs, indexing="xy")

        x = torch.ones_like(grid_y)
        y = grid_y * tan_half
        z = grid_z * tan_half

        dirs = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        return self._normalize_batch(dirs)

    def _ray_arena_box_distance(self, origins: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        x_min = -float(cfg.arena_half_length)
        x_max = float(cfg.arena_half_length)
        y_min = -float(cfg.arena_half_width)
        y_max = float(cfg.arena_half_width)
        z_min = 0.0
        z_max = float(cfg.arena_height)

        max_range = float(cfg.cam_far)
        eps = 1.0e-6

        candidates = []

        for axis, lo, hi in [(0, x_min, x_max), (1, y_min, y_max), (2, z_min, z_max)]:
            o = origins[..., axis]
            d = dirs[..., axis]

            t_hi = torch.where(torch.abs(d) > eps, (float(hi) - o) / d, torch.full_like(o, max_range))
            t_lo = torch.where(torch.abs(d) > eps, (float(lo) - o) / d, torch.full_like(o, max_range))

            for t in [t_hi, t_lo]:
                p = origins + t.unsqueeze(-1) * dirs
                inside = (
                    (p[..., 0] >= x_min - 1.0e-5)
                    & (p[..., 0] <= x_max + 1.0e-5)
                    & (p[..., 1] >= y_min - 1.0e-5)
                    & (p[..., 1] <= y_max + 1.0e-5)
                    & (p[..., 2] >= z_min - 1.0e-5)
                    & (p[..., 2] <= z_max + 1.0e-5)
                    & (t >= float(cfg.cam_near))
                    & (t <= max_range)
                )
                candidates.append(torch.where(inside, t, torch.full_like(t, max_range)))

        return torch.stack(candidates, dim=-1).min(dim=-1).values

    def _ray_gate_frame_distance(self, origins: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg

        n, r, _ = origins.shape
        g = int(cfg.num_gates)

        max_range = float(cfg.cam_far)
        near = float(cfg.cam_near)

        gate_pos = self.gate_pos
        gate_rot = self.gate_rot
        gate_normal = gate_rot[..., :, 2]
        gate_valid = self.gate_valid

        o = origins[:, :, None, :]
        d = dirs[:, :, None, :]
        p0 = gate_pos[:, None, :, :]
        nrm = gate_normal[:, None, :, :]

        denom = torch.sum(d * nrm, dim=-1)
        numer = torch.sum((p0 - o) * nrm, dim=-1)

        t = numer / torch.clamp(denom, min=1.0e-6)
        t = torch.where(denom.abs() > 1.0e-6, t, torch.full_like(t, max_range))

        hit_point = o + t.unsqueeze(-1) * d
        diff = hit_point - p0

        rot_t = gate_rot.transpose(-1, -2)
        local = torch.einsum("ngij,nrgj->nrgi", rot_t, diff)

        lx = local[..., 0]
        ly = local[..., 1]

        inner = float(cfg.gate_inner_half)
        outer = float(cfg.gate_outer_half)

        within_outer = (lx.abs() <= outer) & (ly.abs() <= outer)
        inside_opening = (lx.abs() <= inner) & (ly.abs() <= inner)

        frame_hit = within_outer & (~inside_opening)

        hit = (
            frame_hit
            & gate_valid[:, None, :]
            & (t >= near)
            & (t <= max_range)
        )

        dist = torch.where(hit, t, torch.full_like(t, max_range))
        return dist.min(dim=-1).values

    # ------------------------------------------------------------------
    # Racing checks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def check_gate_pass(
        self,
        prev_pos: torch.Tensor,
        curr_pos: torch.Tensor,
        target_gate_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check whether segment prev->curr passes through target gate opening.

        Returns:
            passed: [num_envs] bool
            alpha:  [num_envs] crossing interpolation value in [0, 1]
        """

        prev = torch.as_tensor(prev_pos, dtype=torch.float32, device=self.device)
        curr = torch.as_tensor(curr_pos, dtype=torch.float32, device=self.device)

        if prev.ndim == 1:
            prev = prev.view(1, 3).repeat(self.num_envs, 1)
        if curr.ndim == 1:
            curr = curr.view(1, 3).repeat(self.num_envs, 1)

        if target_gate_idx is None:
            idx = self.target_gate_idx
        else:
            idx = torch.as_tensor(target_gate_idx, dtype=torch.long, device=self.device).reshape(self.num_envs)

        idx = torch.clamp(idx, 0, int(self.cfg.num_gates) - 1)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)

        gp = self.gate_pos[env_ids, idx]
        gr = self.gate_rot[env_ids, idx]
        gn = gr[:, :, 2]

        delta = curr - prev
        denom = torch.sum(delta * gn, dim=-1)
        numer = torch.sum((gp - prev) * gn, dim=-1)

        alpha = numer / torch.clamp(denom, min=1.0e-6)
        alpha = torch.where(denom.abs() > 1.0e-6, alpha, torch.full_like(alpha, -1.0))

        cross = prev + alpha.unsqueeze(-1) * delta
        local = torch.bmm(gr.transpose(1, 2), (cross - gp).unsqueeze(-1)).squeeze(-1)

        inner = float(self.cfg.gate_inner_half - self.cfg.pass_gate_margin)

        inside_opening = (local[:, 0].abs() <= inner) & (local[:, 1].abs() <= inner)
        forward_cross = denom > 0.0
        in_segment = (alpha >= 0.0) & (alpha <= 1.0)

        passed = in_segment & forward_cross & inside_opening
        return passed, alpha

    @torch.no_grad()
    def update_gate_progress(self, prev_pos: torch.Tensor, curr_pos: torch.Tensor) -> torch.Tensor:
        passed, _ = self.check_gate_pass(prev_pos, curr_pos, self.target_gate_idx)

        not_finished = self.target_gate_idx < int(self.cfg.num_gates)
        advance = passed & not_finished

        self.target_gate_idx = torch.where(
            advance,
            torch.clamp(self.target_gate_idx + 1, max=int(self.cfg.num_gates)),
            self.target_gate_idx,
        )

        return advance

    @torch.no_grad()
    def check_gate_collision(self, drone_pos: torch.Tensor, robot_radius: Optional[float] = None) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        radius = float(self.cfg.robot_radius if robot_radius is None else robot_radius)

        diff = pos[:, None, :] - self.gate_pos
        local = torch.einsum("ngij,ngj->ngi", self.gate_rot.transpose(-1, -2), diff)

        lx = local[..., 0]
        ly = local[..., 1]
        lz = local[..., 2]

        inner = float(self.cfg.gate_inner_half)
        outer = float(self.cfg.gate_outer_half)

        near_plane = lz.abs() <= max(float(self.cfg.gate_plane_tolerance), radius)

        within_outer = (lx.abs() <= outer + radius) & (ly.abs() <= outer + radius)
        inside_opening_safe = (lx.abs() <= inner - radius) & (ly.abs() <= inner - radius)

        hit_frame = near_plane & within_outer & (~inside_opening_safe) & self.gate_valid
        return hit_frame.any(dim=-1)

    @torch.no_grad()
    def check_out_of_bounds(self, drone_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        return (
            (pos[:, 0] < -float(self.cfg.arena_half_length))
            | (pos[:, 0] > float(self.cfg.arena_half_length))
            | (pos[:, 1] < -float(self.cfg.arena_half_width))
            | (pos[:, 1] > float(self.cfg.arena_half_width))
            | (pos[:, 2] < float(self.cfg.min_flight_z))
            | (pos[:, 2] > float(self.cfg.max_flight_z))
        )

    @torch.no_grad()
    def check_success(self) -> torch.Tensor:
        return self.target_gate_idx >= int(self.cfg.num_gates)

    @torch.no_grad()
    def current_target_gate_features(self, drone_pos: torch.Tensor, drone_quat_wxyz: torch.Tensor) -> torch.Tensor:
        """Return compact features for the currently targeted gate.

        Features:
            rel gate position in body frame 3
            distance to gate               1
            gate normal in body frame      3
            normalized gate index          1
            next gate relative position    3
            depth min / mean / center      3
        Total: 14
        """

        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        quat = torch.as_tensor(drone_quat_wxyz, dtype=torch.float32, device=self.device)

        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)
        if quat.ndim == 1:
            quat = quat.view(1, 4).repeat(self.num_envs, 1)

        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        idx = torch.clamp(self.target_gate_idx, 0, int(self.cfg.num_gates) - 1)
        next_idx = torch.clamp(idx + 1, 0, int(self.cfg.num_gates) - 1)

        gp = self.gate_pos[env_ids, idx]
        gp_next = self.gate_pos[env_ids, next_idx]
        gn = self.gate_tangent[env_ids, idx]

        rel = gp - pos
        rel_next = gp_next - pos

        quat_inv = quat.clone()
        quat_inv[:, 1:4] *= -1.0

        rel_b = self._quat_rotate_batch(quat_inv, rel)
        rel_next_b = self._quat_rotate_batch(quat_inv, rel_next)
        gn_b = self._quat_rotate_batch(quat_inv, gn)

        dist = torch.norm(rel, dim=-1, keepdim=True)
        idx_norm = self.target_gate_idx.float().view(-1, 1) / max(float(self.cfg.num_gates), 1.0)

        depth = self.last_depth.view(self.num_envs, -1)
        depth_min = depth.min(dim=-1, keepdim=True).values
        depth_mean = depth.mean(dim=-1, keepdim=True)
        center = self.last_depth[:, 0, int(self.cfg.cam_res_h // 2), int(self.cfg.cam_res_w // 2)].view(-1, 1)

        features = torch.cat(
            [
                rel_b / float(self.cfg.arena_length),
                dist / float(self.cfg.arena_length),
                gn_b,
                idx_norm,
                rel_next_b / float(self.cfg.arena_length),
                depth_min,
                depth_mean,
                center,
            ],
            dim=-1,
        )

        return torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

    # ------------------------------------------------------------------
    # Export / summary
    # ------------------------------------------------------------------
    def get_gate_pose_list(self, env_ids: Optional[torch.Tensor] = None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        return {
            "pos": self.gate_pos[env_ids].clone(),
            "quat_wxyz": self.gate_quat[env_ids].clone(),
            "rot": self.gate_rot[env_ids].clone(),
            "tangent": self.gate_tangent[env_ids].clone(),
            "valid": self.gate_valid[env_ids].clone(),
        }

    def get_world_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "start_pos": self.start_pos,
            "gate_pos": self.gate_pos,
            "gate_quat": self.gate_quat,
            "gate_rot": self.gate_rot,
            "gate_tangent": self.gate_tangent,
            "gate_valid": self.gate_valid,
            "target_gate_idx": self.target_gate_idx,
            "centerline": self.centerline,
            "last_depth": self.last_depth,
        }

    def summary(self) -> Dict[str, float]:
        gate_y = self.gate_pos[..., 1]
        gate_z = self.gate_pos[..., 2]
        tang = self.gate_tangent

        seg = self.gate_pos[:, 1:, :] - self.gate_pos[:, :-1, :]
        seg_len = torch.norm(seg, dim=-1) if seg.numel() else torch.zeros((self.num_envs, 1), device=self.device)

        return {
            "num_envs": float(self.num_envs),
            "arena_length": float(self.cfg.arena_length),
            "arena_width": float(self.cfg.arena_width),
            "arena_height": float(self.cfg.arena_height),
            "num_gates": float(self.cfg.num_gates),
            "gate_size": float(self.cfg.gate_size),
            "gate_thickness": float(self.cfg.gate_thickness),
            "cam_res_w": float(self.cfg.cam_res_w),
            "cam_res_h": float(self.cfg.cam_res_h),
            "cam_fov_deg": float(self.cfg.cam_fov_deg),
            "cam_far": float(self.cfg.cam_far),
            "gate_y_mean": float(gate_y.mean().item()),
            "gate_y_min": float(gate_y.min().item()),
            "gate_y_max": float(gate_y.max().item()),
            "gate_z_mean": float(gate_z.mean().item()),
            "gate_z_min": float(gate_z.min().item()),
            "gate_z_max": float(gate_z.max().item()),
            "segment_len_mean": float(seg_len.mean().item()),
            "tangent_norm_mean": float(torch.norm(tang, dim=-1).mean().item()),
        }

    # ------------------------------------------------------------------
    # Quaternion / geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(v: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return v / torch.clamp(torch.norm(v), min=eps)

    @staticmethod
    def _normalize_batch(v: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return v / torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=eps)

    @staticmethod
    def _normalize_quat(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return q / torch.clamp(torch.norm(q), min=eps)

    @staticmethod
    def _normalize_quat_batch(q: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
        return q / torch.clamp(torch.norm(q, dim=-1, keepdim=True), min=eps)

    def _rotation_from_local_z_to_vector(self, vec: torch.Tensor) -> torch.Tensor:
        z = self._normalize(vec)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device)

        x = torch.cross(up, z, dim=0)
        if torch.norm(x).item() < 1.0e-4:
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=self.device)
            x = torch.cross(up, z, dim=0)

        x = self._normalize(x)
        y = self._normalize(torch.cross(z, x, dim=0))

        return torch.stack([x, y, z], dim=1)

    def _axis_angle_quat(self, axis: torch.Tensor, angle: float) -> torch.Tensor:
        axis = self._normalize(axis)
        half = 0.5 * float(angle)
        s = math.sin(half)

        return torch.tensor(
            [math.cos(half), axis[0].item() * s, axis[1].item() * s, axis[2].item() * s],
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def _matrix_to_quat(self, m: torch.Tensor) -> torch.Tensor:
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace.item() > 0.0:
            s = torch.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        return self._normalize_quat(torch.stack([qw, qx, qy, qz], dim=0))

    @staticmethod
    def _quat_to_matrix_single(q: torch.Tensor) -> torch.Tensor:
        q = q / torch.clamp(torch.norm(q), min=1.0e-6)

        w, x, y, z = q

        return torch.stack(
            [
                torch.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)]),
                torch.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)]),
                torch.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)]),
            ],
            dim=0,
        )

    @staticmethod
    def _quat_rotate_batch(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_w = q[..., 0:1]
        q_vec = q[..., 1:4]
        t = 2.0 * torch.cross(q_vec, v, dim=-1)
        return v + q_w * t + torch.cross(q_vec, t, dim=-1)


Task4World = QuadrotorTask4World
CrazyflieTask4World = QuadrotorTask4World
