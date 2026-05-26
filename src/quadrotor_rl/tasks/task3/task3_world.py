from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

from quadrotor_rl.tasks.task3.task3_config import Task3Config


class QuadrotorTask3World:
    """Torch-native analytic world for Task3 dynamic obstacle navigation.

    This class replaces the old simulator-specific obstacle world with a
    batched analytic world layer.

    It is responsible for:
        - start / goal sampling
        - static obstacle sampling
        - dynamic obstacle sampling and motion integration
        - 2D LiDAR ray-circle / ray-wall intersection
        - collision / success / boundary checks

    It deliberately does not create Isaac prims. This keeps Task3 lightweight
    and suitable for high-parallel RL training.
    """

    def __init__(
        self,
        cfg: Task3Config,
        num_envs: Optional[int] = None,
        device: Optional[str] = None,
    ):
        cfg.validate()
        self.cfg = cfg

        self.num_envs = int(num_envs if num_envs is not None else cfg.num_envs)
        self.device = str(device if device is not None else cfg.device)

        torch.manual_seed(int(cfg.seed))

        n = self.num_envs
        ns = int(cfg.num_static_obs)
        nd = int(cfg.num_dynamic_obs)

        self.start_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)
        self.goal_pos = torch.zeros((n, 3), dtype=torch.float32, device=self.device)

        self.static_pos = torch.zeros((n, ns, 2), dtype=torch.float32, device=self.device)
        self.static_radius = torch.zeros((n, ns), dtype=torch.float32, device=self.device)
        self.static_valid = torch.zeros((n, ns), dtype=torch.bool, device=self.device)

        self.dynamic_pos = torch.zeros((n, nd, 2), dtype=torch.float32, device=self.device)
        self.dynamic_vel = torch.zeros((n, nd, 2), dtype=torch.float32, device=self.device)
        self.dynamic_radius = torch.full((n, nd), float(cfg.dynamic_radius), dtype=torch.float32, device=self.device)
        self.dynamic_valid = torch.zeros((n, nd), dtype=torch.bool, device=self.device)

        self.last_lidar = torch.ones(
            (n, int(cfg.lidar_num_rays)),
            dtype=torch.float32,
            device=self.device,
        )

        self.reset()

    # ------------------------------------------------------------------
    # Reset / generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0:
            return self.start_pos, self.goal_pos

        for env_id_t in env_ids:
            env_id = int(env_id_t.item())
            start, goal = self._sample_start_goal_one()
            self.start_pos[env_id] = start
            self.goal_pos[env_id] = goal
            self._sample_obstacles_one(env_id)

        self.last_lidar[env_ids] = 1.0

        return self.start_pos[env_ids].clone(), self.goal_pos[env_ids].clone()

    def _rand_scalar(self, lo: float, hi: float) -> float:
        return float(torch.empty((), dtype=torch.float32, device=self.device).uniform_(float(lo), float(hi)).item())

    def _sample_start_goal_one(self) -> Tuple[torch.Tensor, torch.Tensor]:
        bound = float(self.cfg.start_goal_bound)

        for _ in range(512):
            sx = self._rand_scalar(-bound, bound)
            sy = self._rand_scalar(-bound, bound)

            dist = self._rand_scalar(self.cfg.min_start_goal_dist, self.cfg.max_start_goal_dist)
            ang = self._rand_scalar(0.0, 2.0 * math.pi)

            gx = sx + dist * math.cos(ang)
            gy = sy + dist * math.sin(ang)

            if -bound <= gx <= bound and -bound <= gy <= bound:
                start = torch.tensor([sx, sy, float(self.cfg.start_goal_z)], dtype=torch.float32, device=self.device)
                goal = torch.tensor([gx, gy, float(self.cfg.start_goal_z)], dtype=torch.float32, device=self.device)
                return start, goal

        # Extremely unlikely fallback.
        start = torch.tensor(
            [-0.5 * float(self.cfg.min_start_goal_dist), 0.0, float(self.cfg.start_goal_z)],
            dtype=torch.float32,
            device=self.device,
        )
        goal = torch.tensor(
            [0.5 * float(self.cfg.min_start_goal_dist), 0.0, float(self.cfg.start_goal_z)],
            dtype=torch.float32,
            device=self.device,
        )
        return start, goal

    def _sample_obstacles_one(self, env_id: int) -> None:
        cfg = self.cfg
        half = float(cfg.arena_half)

        self.static_pos[env_id].zero_()
        self.static_radius[env_id].zero_()
        self.static_valid[env_id].fill_(False)

        self.dynamic_pos[env_id].zero_()
        self.dynamic_vel[env_id].zero_()
        self.dynamic_radius[env_id].fill_(float(cfg.dynamic_radius))
        self.dynamic_valid[env_id].fill_(False)

        start_xy = self.start_pos[env_id, :2].detach().cpu().numpy()
        goal_xy = self.goal_pos[env_id, :2].detach().cpu().numpy()

        circles: list[tuple[float, float, float]] = []

        # Static obstacle forest.
        for obs_i in range(int(cfg.num_static_obs)):
            placed = False
            for _ in range(200):
                x = self._rand_scalar(-half, half)
                y = self._rand_scalar(-half, half)
                r = self._rand_scalar(cfg.static_radius_min, cfg.static_radius_max)

                if self._is_xy_valid(x, y, r, circles, start_xy, goal_xy):
                    self.static_pos[env_id, obs_i, 0] = x
                    self.static_pos[env_id, obs_i, 1] = y
                    self.static_radius[env_id, obs_i] = r
                    self.static_valid[env_id, obs_i] = True
                    circles.append((x, y, r))
                    placed = True
                    break

            if not placed:
                # Keep invalid and far away.
                self.static_pos[env_id, obs_i] = torch.tensor([half + 100.0, half + 100.0], device=self.device)

        # Dynamic interceptors near the start-goal corridor.
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        for obs_i in range(int(cfg.num_dynamic_obs)):
            placed = False
            for _ in range(200):
                t = self._rand_scalar(cfg.dynamic_spawn_line_t_min, cfg.dynamic_spawn_line_t_max)
                x = sx + t * (gx - sx) + self._rand_scalar(-cfg.dynamic_lateral_spread, cfg.dynamic_lateral_spread)
                y = sy + t * (gy - sy) + self._rand_scalar(-cfg.dynamic_lateral_spread, cfg.dynamic_lateral_spread)
                r = float(cfg.dynamic_radius)

                if self._is_xy_valid(x, y, r, circles, start_xy, goal_xy):
                    ang = self._rand_scalar(0.0, 2.0 * math.pi)
                    vx = float(cfg.dynamic_speed) * math.cos(ang)
                    vy = float(cfg.dynamic_speed) * math.sin(ang)

                    self.dynamic_pos[env_id, obs_i, 0] = x
                    self.dynamic_pos[env_id, obs_i, 1] = y
                    self.dynamic_vel[env_id, obs_i, 0] = vx
                    self.dynamic_vel[env_id, obs_i, 1] = vy
                    self.dynamic_radius[env_id, obs_i] = r
                    self.dynamic_valid[env_id, obs_i] = True
                    circles.append((x, y, r))
                    placed = True
                    break

            if not placed:
                self.dynamic_pos[env_id, obs_i] = torch.tensor([half + 100.0, half + 100.0], device=self.device)

    def _is_xy_valid(
        self,
        x: float,
        y: float,
        radius: float,
        circles: list[tuple[float, float, float]],
        start_xy,
        goal_xy,
    ) -> bool:
        cfg = self.cfg

        if math.hypot(x - float(start_xy[0]), y - float(start_xy[1])) < float(cfg.safe_zone_radius + radius):
            return False

        if math.hypot(x - float(goal_xy[0]), y - float(goal_xy[1])) < float(cfg.safe_zone_radius + radius):
            return False

        half = float(cfg.arena_half)
        if abs(x) + radius > half or abs(y) + radius > half:
            return False

        for cx, cy, cr in circles:
            if math.hypot(x - cx, y - cy) < float(radius + cr + cfg.min_obs_gap):
                return False

        return True

    # ------------------------------------------------------------------
    # Dynamic obstacle motion
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step_dynamics(self, dt: Optional[float] = None, env_ids: Optional[torch.Tensor] = None) -> None:
        if dt is None:
            dt = float(self.cfg.policy_dt)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()

        if env_ids.numel() == 0 or int(self.cfg.num_dynamic_obs) == 0:
            return

        pos = self.dynamic_pos[env_ids]
        vel = self.dynamic_vel[env_ids]
        radius = self.dynamic_radius[env_ids]
        valid = self.dynamic_valid[env_ids]

        pos = pos + vel * float(dt)

        half = float(self.cfg.arena_half)
        bound = half - radius

        x = pos[..., 0]
        y = pos[..., 1]
        vx = vel[..., 0]
        vy = vel[..., 1]

        hit_x_hi = x > bound
        hit_x_lo = x < -bound
        hit_y_hi = y > bound
        hit_y_lo = y < -bound

        vx = torch.where(hit_x_hi | hit_x_lo, -vx, vx)
        vy = torch.where(hit_y_hi | hit_y_lo, -vy, vy)

        x = torch.minimum(torch.maximum(x, -bound), bound)
        y = torch.minimum(torch.maximum(y, -bound), bound)

        vel = torch.stack([vx, vy], dim=-1)
        pos = torch.stack([x, y], dim=-1)

        speed = torch.norm(vel, dim=-1, keepdim=True)
        target_speed = float(self.cfg.dynamic_speed)

        random_ang = torch.rand(speed.shape, dtype=torch.float32, device=self.device) * (2.0 * math.pi)
        random_dir = torch.cat([torch.cos(random_ang), torch.sin(random_ang)], dim=-1)

        safe_dir = vel / torch.clamp(speed, min=1.0e-6)
        refill_dir = torch.where(speed > 0.05, safe_dir, random_dir)

        low_speed = speed < (0.90 * target_speed)
        vel_refill = refill_dir * target_speed
        vel = torch.where(low_speed.expand_as(vel), vel_refill, vel)

        pos = torch.where(valid.unsqueeze(-1), pos, self.dynamic_pos[env_ids])
        vel = torch.where(valid.unsqueeze(-1), vel, self.dynamic_vel[env_ids])

        self.dynamic_pos[env_ids] = pos
        self.dynamic_vel[env_ids] = vel

    # ------------------------------------------------------------------
    # LiDAR
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_lidar_scan(self, drone_pos: torch.Tensor, drone_yaw: torch.Tensor | float) -> torch.Tensor:
        """Return normalized LiDAR distances in [0, 1].

        Args:
            drone_pos: [num_envs, 3] or [3], world-local position.
            drone_yaw: [num_envs] or scalar yaw.

        Returns:
            lidar: [num_envs, lidar_num_rays], where 1.0 means no hit within range.
        """

        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)

        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        assert pos.shape == (self.num_envs, 3), f"drone_pos must be {(self.num_envs, 3)}, got {tuple(pos.shape)}"

        yaw = torch.as_tensor(drone_yaw, dtype=torch.float32, device=self.device)
        if yaw.numel() == 1:
            yaw = yaw.reshape(1).repeat(self.num_envs)
        yaw = yaw.reshape(self.num_envs)

        r = int(self.cfg.lidar_num_rays)
        max_range = float(self.cfg.lidar_max_range)
        offset = float(self.cfg.lidar_start_offset)
        max_t = max_range - offset

        base_angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            r + 1,
            dtype=torch.float32,
            device=self.device,
        )[:-1]

        angles = base_angles.view(1, r) + yaw.view(self.num_envs, 1)
        dirs = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)

        origins = pos[:, None, :2] + offset * dirs

        min_dist = torch.full((self.num_envs, r), max_range, dtype=torch.float32, device=self.device)

        centers, radii, valid = self._combined_obstacles()

        if centers.shape[1] > 0:
            hit_dist = self._ray_circle_distance(
                origins=origins,
                dirs=dirs,
                centers=centers,
                radii=radii,
                valid=valid,
                max_t=max_t,
                offset=offset,
                max_range=max_range,
            )
            min_dist = torch.minimum(min_dist, hit_dist)

        wall_dist = self._ray_wall_distance(
            origins=origins,
            dirs=dirs,
            max_t=max_t,
            offset=offset,
            max_range=max_range,
        )
        min_dist = torch.minimum(min_dist, wall_dist)

        lidar = torch.clamp(min_dist / max_range, 0.0, 1.0)
        lidar = torch.nan_to_num(lidar, nan=1.0, posinf=1.0, neginf=0.0)

        self.last_lidar = lidar.clone()
        return lidar

    def _combined_obstacles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = []
        radii = []
        valid = []

        if int(self.cfg.num_static_obs) > 0:
            centers.append(self.static_pos)
            radii.append(self.static_radius)
            valid.append(self.static_valid)

        if int(self.cfg.num_dynamic_obs) > 0:
            centers.append(self.dynamic_pos)
            radii.append(self.dynamic_radius)
            valid.append(self.dynamic_valid)

        if not centers:
            return (
                torch.zeros((self.num_envs, 0, 2), dtype=torch.float32, device=self.device),
                torch.zeros((self.num_envs, 0), dtype=torch.float32, device=self.device),
                torch.zeros((self.num_envs, 0), dtype=torch.bool, device=self.device),
            )

        return torch.cat(centers, dim=1), torch.cat(radii, dim=1), torch.cat(valid, dim=1)

    def _ray_circle_distance(
        self,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        centers: torch.Tensor,
        radii: torch.Tensor,
        valid: torch.Tensor,
        max_t: float,
        offset: float,
        max_range: float,
    ) -> torch.Tensor:
        # origins: [N, R, 2]
        # dirs:    [N, R, 2]
        # centers: [N, M, 2]
        o = origins[:, :, None, :]
        d = dirs[:, :, None, :]
        c = centers[:, None, :, :]
        rr = radii[:, None, :]

        m = o - c
        b = torch.sum(m * d, dim=-1)
        cc = torch.sum(m * m, dim=-1) - rr * rr

        disc = b * b - cc
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=0.0))

        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        t = torch.where(t1 >= 0.0, t1, t2)

        hit = (
            (disc >= 0.0)
            & (t >= 0.0)
            & (t <= float(max_t))
            & valid[:, None, :]
        )

        dist = torch.where(
            hit,
            t + float(offset),
            torch.full_like(t, float(max_range)),
        )

        return dist.min(dim=-1).values

    def _ray_wall_distance(
        self,
        origins: torch.Tensor,
        dirs: torch.Tensor,
        max_t: float,
        offset: float,
        max_range: float,
    ) -> torch.Tensor:
        half = float(self.cfg.arena_half)
        ox = origins[..., 0]
        oy = origins[..., 1]
        dx = dirs[..., 0]
        dy = dirs[..., 1]

        eps = 1.0e-6
        big = torch.full_like(ox, float(max_range))

        candidates = []

        for x_wall in [half, -half]:
            t = torch.where(torch.abs(dx) > eps, (float(x_wall) - ox) / dx, big)
            y_at = oy + t * dy
            hit = (t >= 0.0) & (t <= float(max_t)) & (y_at >= -half) & (y_at <= half)
            candidates.append(torch.where(hit, t + float(offset), big))

        for y_wall in [half, -half]:
            t = torch.where(torch.abs(dy) > eps, (float(y_wall) - oy) / dy, big)
            x_at = ox + t * dx
            hit = (t >= 0.0) & (t <= float(max_t)) & (x_at >= -half) & (x_at <= half)
            candidates.append(torch.where(hit, t + float(offset), big))

        return torch.stack(candidates, dim=-1).min(dim=-1).values

    # ------------------------------------------------------------------
    # Navigation checks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def check_obstacle_collision(self, drone_pos: torch.Tensor, robot_radius: Optional[float] = None) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        robot_r = float(self.cfg.robot_radius if robot_radius is None else robot_radius)

        centers, radii, valid = self._combined_obstacles()
        if centers.shape[1] == 0:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        d = torch.norm(pos[:, None, :2] - centers, dim=-1)
        hit = (d <= (radii + robot_r)) & valid
        return hit.any(dim=-1)

    @torch.no_grad()
    def check_success(self, drone_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        goal_dist = torch.norm(pos[:, :2] - self.goal_pos[:, :2], dim=-1)
        z_ok = torch.abs(pos[:, 2] - self.goal_pos[:, 2]) <= 0.80
        return (goal_dist <= float(self.cfg.success_radius)) & z_ok

    @torch.no_grad()
    def check_out_of_bounds(self, drone_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        half = float(self.cfg.arena_half)
        xy_oob = (torch.abs(pos[:, 0]) > half) | (torch.abs(pos[:, 1]) > half)
        z_oob = (pos[:, 2] < float(self.cfg.min_flight_z)) | (pos[:, 2] > float(self.cfg.max_flight_z))
        return xy_oob | z_oob

    @torch.no_grad()
    def goal_vector(self, drone_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)
        return self.goal_pos - pos

    @torch.no_grad()
    def distance_to_goal(self, drone_pos: torch.Tensor) -> torch.Tensor:
        return torch.norm(self.goal_vector(drone_pos), dim=-1)

    @torch.no_grad()
    def nearest_obstacle_distance(self, drone_pos: torch.Tensor) -> torch.Tensor:
        pos = torch.as_tensor(drone_pos, dtype=torch.float32, device=self.device)
        if pos.ndim == 1:
            pos = pos.view(1, 3).repeat(self.num_envs, 1)

        centers, radii, valid = self._combined_obstacles()
        if centers.shape[1] == 0:
            return torch.full((self.num_envs,), float(self.cfg.lidar_max_range), dtype=torch.float32, device=self.device)

        d = torch.norm(pos[:, None, :2] - centers, dim=-1) - radii
        d = torch.where(valid, d, torch.full_like(d, float(self.cfg.lidar_max_range)))
        return torch.clamp(d.min(dim=-1).values, min=0.0)

    @torch.no_grad()
    def risk_features(self, drone_pos: torch.Tensor, drone_yaw: torch.Tensor | float) -> torch.Tensor:
        lidar = self.get_lidar_scan(drone_pos, drone_yaw)
        goal_vec = self.goal_vector(drone_pos)
        goal_dist = torch.norm(goal_vec, dim=-1)

        nearest = self.nearest_obstacle_distance(drone_pos)
        front = lidar[:, 0]
        left = lidar[:, int(self.cfg.lidar_num_rays // 4)]
        back = lidar[:, int(self.cfg.lidar_num_rays // 2)]
        right = lidar[:, int(3 * self.cfg.lidar_num_rays // 4)]

        lidar_min = lidar.min(dim=-1).values
        lidar_mean = lidar.mean(dim=-1)

        features = torch.stack(
            [
                torch.clamp(lidar_min, 0.0, 1.0),
                torch.clamp(lidar_mean, 0.0, 1.0),
                torch.clamp(front, 0.0, 1.0),
                torch.clamp(left, 0.0, 1.0),
                torch.clamp(back, 0.0, 1.0),
                torch.clamp(right, 0.0, 1.0),
                torch.clamp(nearest / float(self.cfg.lidar_max_range), 0.0, 1.0),
                torch.clamp(goal_dist / float(self.cfg.max_start_goal_dist), 0.0, 5.0),
            ],
            dim=-1,
        )

        return torch.nan_to_num(features, nan=1.0, posinf=1.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Debug / export
    # ------------------------------------------------------------------
    def get_world_tensors(self) -> Dict[str, torch.Tensor]:
        return {
            "start_pos": self.start_pos,
            "goal_pos": self.goal_pos,
            "static_pos": self.static_pos,
            "static_radius": self.static_radius,
            "static_valid": self.static_valid,
            "dynamic_pos": self.dynamic_pos,
            "dynamic_vel": self.dynamic_vel,
            "dynamic_radius": self.dynamic_radius,
            "dynamic_valid": self.dynamic_valid,
            "last_lidar": self.last_lidar,
        }

    def summary(self) -> Dict[str, float]:
        start_goal_dist = torch.norm(self.goal_pos[:, :2] - self.start_pos[:, :2], dim=-1)
        dynamic_speed = torch.norm(self.dynamic_vel, dim=-1)

        return {
            "num_envs": float(self.num_envs),
            "arena_size": float(self.cfg.arena_size),
            "num_static_obs": float(self.cfg.num_static_obs),
            "num_dynamic_obs": float(self.cfg.num_dynamic_obs),
            "lidar_num_rays": float(self.cfg.lidar_num_rays),
            "lidar_max_range": float(self.cfg.lidar_max_range),
            "start_goal_dist_mean": float(start_goal_dist.mean().item()),
            "start_goal_dist_min": float(start_goal_dist.min().item()),
            "start_goal_dist_max": float(start_goal_dist.max().item()),
            "static_valid_mean": float(self.static_valid.float().mean().item()),
            "dynamic_valid_mean": float(self.dynamic_valid.float().mean().item()),
            "dynamic_speed_mean": float(dynamic_speed[self.dynamic_valid].mean().item())
            if self.dynamic_valid.any().item()
            else 0.0,
        }


Task3World = QuadrotorTask3World
CrazyflieTask3World = QuadrotorTask3World
