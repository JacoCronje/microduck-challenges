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

    heading_hold prices where the trunk faces, not where it goes: race6's walker
    faced the lane yet crabbed left 26-30 cm over 2 m, and every fall was it
    meeting the left wall. Training has no walls, so nothing else prices this.
    Unlike the refuted lane_hold (a Gaussian on the position), this costs nothing
    while running straight, wherever in the lane.
    """
    asset = env.scene[asset_cfg.name]
    lane = _race_direction(env, asset)
    across = torch.stack((-lane[:, 1], lane[:, 0]), dim=-1)
    return (asset.data.root_link_lin_vel_w[:, :2] * across).sum(-1).abs()


def stride_length(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
    max_length: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """At each touchdown, how far that foot travelled along the lane since liftoff, past threshold (m).

    Summed distance alone would only re-pay speed (both feet cover the whole
    course); paying the length beyond threshold per step favours fewer, longer
    strides at the same speed. Contact flicker inside a stance makes steps far
    shorter than threshold, so it pays nothing. Capped at max_length.
    asset_cfg's site_names are the feet, in the contact sensor's order.

    Divided by step_dt: the reward manager multiplies every term by step_dt,
    right for a rate paid every step but 50x too weak for a once-per-touchdown
    payment (race4 and race5 logged 0.007-0.013 for it). Per second, a gait then
    earns weight x (sum of steps' length past threshold).
    """
    asset = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_name]
    feet = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # (N, 2 feet, xy)
    if not hasattr(env, "_foot_liftoff") or env._foot_liftoff.shape != feet.shape:
        env._foot_liftoff = feet.clone()
    new_episode = (env.episode_length_buf <= 1)[:, None, None]
    lifted = sensor.compute_first_air(env.step_dt)[:, :, None]
    env._foot_liftoff = torch.where(new_episode | lifted, feet, env._foot_liftoff)

    travel = ((feet - env._foot_liftoff) * _race_direction(env, asset)[:, None, :]).sum(-1)
    landed = sensor.compute_first_contact(env.step_dt)
    pay = (travel - threshold).clamp(0.0, max_length - threshold)
    return (pay * landed).sum(-1) / env.step_dt


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

