"""The sprint's own reward terms: race progress, and a tilt guard instead of upright.

The Arena times 2 m from a standstill, at whatever speed the policy reaches: its
leaders read a 0.25–1.2 m/s command and run at about 2.5 m/s, the head thrown
forward as a counterweight, the trunk pitching 10–30° and up to 60° at the line.
A Gaussian on the command pays nothing past it, and a Gaussian on a level trunk
pays nothing past 30°: these two terms price the race instead.
"""

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def forward_speed(
    env: ManagerBasedRlEnv,
    max_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Speed along the spawn heading (m/s), capped at max_speed: the race's progress.

    The lane runs along the heading the robot spawns with, so this is the rate the
    Arena's clock rewards, whatever the command and however the trunk pitches. The
    spawn heading is captured on an episode's first steps, as heading_hold does
    (its own copy, so the two terms do not depend on their order). Going backwards
    pays negative; the cap stops a dive's burst of speed from paying more than the
    leaders' top speed.
    """
    asset = env.scene[asset_cfg.name]
    yaw = _yaw(asset.data.root_link_quat_w)
    if not hasattr(env, "_race_heading") or env._race_heading.shape[0] != env.num_envs:
        env._race_heading = yaw.clone()
    env._race_heading = torch.where(env.episode_length_buf <= 1, yaw, env._race_heading)

    vel = asset.data.root_link_lin_vel_w
    along = vel[:, 0] * torch.cos(env._race_heading) + vel[:, 1] * torch.sin(env._race_heading)
    return torch.clamp(along, max=max_speed)


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

