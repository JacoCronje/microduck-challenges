"""The roller sprint's own reward terms: race progress, and a tilt guard instead of upright.

Copied from the 2 m Sprint's rewards.py (stride_length left out: a skate has no
footfalls to measure). The Arena times 2 m at whatever speed the policy reaches,
and runs the policy through the 0.5 s settle with the clock stopped: its leaders
skate through the start line at 1.3 m/s and cross the finish at 2.4-2.5 m/s, the
trunk pitched 15-20° and up to 50° at the line. wheel_speed's tanh paid nothing
past 0.3 m/s and upright's Gaussian little past 20°: these terms price the race.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _race_direction(env: ManagerBasedRlEnv, asset) -> torch.Tensor:
    """Unit xy vector of the lane: the heading the robot spawned with, (N, 2).

    Captured on an episode's first steps, as heading_hold does (its own copy, so
    the terms do not depend on their order); calling it twice in a step is harmless.
    """
    yaw = _yaw(asset.data.root_link_quat_w)
    if not hasattr(env, "_race_heading") or env._race_heading.shape[0] != env.num_envs:
        env._race_heading = yaw.clone()
    env._race_heading = torch.where(env.episode_length_buf <= 1, yaw, env._race_heading)
    return torch.stack((torch.cos(env._race_heading), torch.sin(env._race_heading)), dim=-1)


def forward_speed(
    env: ManagerBasedRlEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Speed along the spawn heading (m/s), capped at max_speed: the race's progress.

    The lane runs along the heading the robot spawns with, so this is the rate the
    Arena's clock rewards, whatever the command and however the trunk pitches.
    Going backwards pays negative; the cap stops a dive's burst of speed from
    paying more than the leaders' top speed.
    """
    asset = env.scene[asset_cfg.name]
    along = (asset.data.root_link_lin_vel_w[:, :2] * _race_direction(env, asset)).sum(-1)
    return torch.clamp(along, max=max_speed)


def lateral_drift(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Sideways speed across the lane (m/s, >= 0): a cost, so give it a negative weight.

    heading_hold prices where the trunk faces, not where it goes: the sprint's race6 walker
    faced the lane yet crabbed left 26-30 cm over 2 m, and every fall was it
    meeting the left wall. Training has no walls, so nothing else prices this.
    Unlike the refuted lane_hold (a Gaussian on the position), this costs nothing
    while running straight, wherever in the lane.
    """
    asset = env.scene[asset_cfg.name]
    lane = _race_direction(env, asset)
    across = torch.stack((-lane[:, 1], lane[:, 0]), dim=-1)
    return (asset.data.root_link_lin_vel_w[:, :2] * across).sum(-1).abs()


def tilt_guard(
    env: ManagerBasedRlEnv,
    free_deg: float,
    zero_deg: float,
    back_free_deg: float = 180.0,
    back_zero_deg: float = 180.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """1 while the trunk tilts less than free_deg, falling linearly to 0 at zero_deg.

    Any lean up to free_deg is free (no gradient toward level); past it the reward
    drains before the Arena's fall line (60° held for 0.3 s). Leaning back (nose
    up) has its own, tighter band: free to back_free_deg, 0 at back_zero_deg.
    """
    gravity = env.scene[asset_cfg.name].data.projected_gravity_b
    tilt = torch.rad2deg(torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0)))
    # gravity's body-x component is sin(pitch), positive nose down.
    back = torch.rad2deg(torch.asin((-gravity[:, 0]).clamp(-1.0, 1.0)))
    guard = ((zero_deg - tilt) / (zero_deg - free_deg)).clamp(0.0, 1.0)
    back_guard = ((back_zero_deg - back) / (back_zero_deg - back_free_deg)).clamp(0.0, 1.0)
    return torch.minimum(guard, back_guard)

