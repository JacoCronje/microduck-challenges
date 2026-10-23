"""The cartwheel's own command, spawns and rewards: microduck_rl's roulade recipe turned sideways.

The roulade (mjlab_microduck/tasks/mdp.py, "Roulade (forward roll) task") rolls forward over the
flat top of the head; a cartwheel rolls the same way about the body's FORWARD axis instead of its
lateral one: tip to one side, over the head (a headstand at 180°), down the other side, back onto
the feet. Everything that made the roulade learnable is kept and only re-aimed:

  • ONE dense progress signal: paid increments of the max-so-far rotation about body x, counted
    only while some part of the robot touches the floor (a ballistic flip earns nothing) and only
    while the forward axis stays near horizontal (a roll that twists out of the plane earns nothing).
  • A head latch: the landing rewards need the flat top of the head to have touched the floor
    around the inverted part of the turn ("used the head to stand upside down").
  • Landing rewards gated on completion (rotation frontier past ~260°), never on a clock.
  • Reverse curriculum: a slice of episodes spawns part-way round, with the turn's momentum.

What the Arena needs on top (a stage plays one policy for the whole show): the duck stands still
until the timeline pulses the trigger, then cartwheels and stands again.

    twist = [0, trigger, 0]   trigger = +1 for 0.5 s: a cartwheel to the duck's left (+y),
                              -1: to its right, 0: stand.

A left cartwheel turns the head toward +y first, which is a NEGATIVE rate about body x (measured on
the stage model): progress = -direction · ω_x. The mirror symmetry negates both twist[1] and ω_x,
so training one direction trains the other.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand, UniformVelocityCommandCfg

from mjlab_microduck.tasks import mdp as microduck_mdp

_ASSET = SceneEntityCfg("robot")
COMMAND = "twist"
SUPPORT_SENSOR = "robot_ground_contact"
HEAD_SENSOR = "head_ground_contact"

# Head latch: flat-top contact while the turn is between these (the inverted part, 180° = headstand).
HEAD_LATCH_LO = math.radians(90.0)
HEAD_LATCH_HI = math.radians(270.0)

# Plane gate: the forward axis's world-z stays ≈0 through a clean cartwheel, whatever the turn;
# full credit within ~30° of horizontal, none past ~60°.
PLANE_FULL = 0.5
PLANE_ZERO = 0.866

# Trunk height (m) at which the robot just touches the floor when turned about body x by 0°, 30°,
# ... 360° at the HOME pose, measured on the stage model (min over every collision vertex).
SPAWN_Z_TABLE = (0.117, 0.130, 0.110, 0.062, 0.102, 0.143, 0.154, 0.143, 0.102, 0.062, 0.110, 0.130, 0.117)


# ── The trigger ────────────────────────────────────────────────────────────────────────────────
class CartwheelCommand(UniformVelocityCommand):
    """twist = [0, trigger, 0]: 0 until a random moment, then ±1 for pulse_s, then 0 again.

    One pulse per episode (resampled only at reset). ``direction`` (+1 left, -1 right) and
    ``fired`` (the pulse has started) are what the rewards read: before the pulse the duck is paid
    to stand still, after it only turning, then landing, pays.

    mjlab resets the spawn events BEFORE the commands, so reset_cartwheel_state draws the direction
    (it needs it to spawn part-way round) and leaves it here as a preset with whether the pulse is
    already long gone; _resample_command honours the preset."""

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self._idle = cfg.idle_range_s
        self._pulse = cfg.pulse_s
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.fire_t = torch.zeros(self.num_envs, device=self.device)
        self.direction = torch.ones(self.num_envs, device=self.device)
        self.preset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.preset_direction = torch.ones(self.num_envs, device=self.device)
        self.preset_gone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def fired(self) -> torch.Tensor:
        return self.t >= self.fire_t

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        lo, hi = self._idle
        preset = self.preset[env_ids]
        drawn = torch.where(torch.rand(n, device=self.device) < 0.5, 1.0, -1.0)
        idle = torch.rand(n, device=self.device) * (hi - lo) + lo
        self.t[env_ids] = 0.0
        self.direction[env_ids] = torch.where(preset, self.preset_direction[env_ids], drawn)
        self.fire_t[env_ids] = torch.where(preset & self.preset_gone[env_ids], torch.full_like(idle, -10.0), idle)
        self.preset[env_ids] = False
        self.vel_command_b[env_ids] = 0.0

    def compute(self, dt: float) -> None:
        super().compute(dt)
        self.t += dt
        on = (self.t >= self.fire_t) & (self.t < self.fire_t + self._pulse)
        self.vel_command_b[:, 0] = 0.0
        self.vel_command_b[:, 1] = self.direction * on.float()
        self.vel_command_b[:, 2] = 0.0

    def _update_command(self) -> None:
        pass

    def _update_metrics(self) -> None:
        pass


@dataclass(kw_only=True)
class CartwheelCommandCfg(UniformVelocityCommandCfg):
    class_type: type = CartwheelCommand
    # Seconds of standing before the pulse, and how long the pulse lasts.
    idle_range_s: tuple = (0.5, 2.5)
    pulse_s: float = 0.5

    def build(self, env: ManagerBasedRlEnv) -> CartwheelCommand:
        return CartwheelCommand(self, env)


def _trigger(env: ManagerBasedRlEnv) -> CartwheelCommand:
    return env.command_manager.get_term(COMMAND)


# ── The turn's bookkeeping ─────────────────────────────────────────────────────────────────────
def _state(env: ManagerBasedRlEnv):
    if not hasattr(env, "_cw_accum"):
        z = torch.zeros(env.num_envs, device=env.device)
        env._cw_accum, env._cw_max, env._cw_paid = z.clone(), z.clone(), z.clone()
        env._cw_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env._cw_step = -1
    return env._cw_accum, env._cw_max, env._cw_paid


def _forward_axis_z(quat: torch.Tensor) -> torch.Tensor:
    """World-z of the body's forward (x) axis: 0 through any amount of pure cartwheel."""
    return 2.0 * (quat[:, 1] * quat[:, 3] - quat[:, 0] * quat[:, 2])


def _turn_rate(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """Rate of turn in the commanded direction (rad/s); a left cartwheel is negative ω_x."""
    w_x = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 0], nan=0.0)
    return -_trigger(env).direction * w_x


def _update(env: ManagerBasedRlEnv, asset: Entity) -> None:
    """Integrate the turn once per control step (several terms read it), only after the pulse,
    only while supported, only while in the plane; latch the head-top contact around 180°."""
    _state(env)
    step = int(env.common_step_counter)
    if step == env._cw_step:
        return
    delta = _turn_rate(env, asset) * env.step_dt * _trigger(env).fired.float()
    supported = microduck_mdp._sensor_any_contact(env, SUPPORT_SENSOR)
    if supported is not None:
        delta = delta * supported.float()
    fz = torch.nan_to_num(_forward_axis_z(asset.data.root_link_quat_w), nan=1.0).abs()
    t = torch.clamp((PLANE_ZERO - fz) / (PLANE_ZERO - PLANE_FULL), 0.0, 1.0)
    delta = delta * t * t * (3.0 - 2.0 * t)
    env._cw_accum = env._cw_accum + delta
    env._cw_max = torch.maximum(env._cw_max, env._cw_accum)
    head = microduck_mdp._sensor_any_contact(env, HEAD_SENSOR)
    if head is not None:
        window = (env._cw_accum > HEAD_LATCH_LO) & (env._cw_accum < HEAD_LATCH_HI)
        env._cw_latch = env._cw_latch | (head & window & microduck_mdp._head_top_down(env, asset))
    env._cw_step = step


def _completion_gate(env: ManagerBasedRlEnv, gate_lo: float, gate_hi: float) -> torch.Tensor:
    """Smoothstep on the turn frontier (0 below gate_lo, 1 above gate_hi), times the head latch."""
    _, frontier, _ = _state(env)
    t = torch.clamp((frontier - gate_lo) / max(gate_hi - gate_lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t) * env._cw_latch.float()


def _height(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    return torch.nan_to_num(asset.data.root_link_pos_w[:, 2] - env.scene.terrain.env_origins[:, 2], nan=0.0)


# ── Spawns ─────────────────────────────────────────────────────────────────────────────────────
def reset_cartwheel_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    standing_prob: float = 0.5,
    midturn_prob: float = 0.5,
    standing_z_range: tuple = (0.11, 0.12),
    standing_tilt_max: float = math.radians(5.0),
    midturn_range: tuple = (math.radians(50.0), math.radians(330.0)),
    midturn_omega_range: tuple = (0.0, 3.0),
    spread_overrides: dict | None = None,
    spread_range: tuple = (0.0, 1.0),
    joint_noise_std: float = 0.08,
    asset_cfg: SceneEntityCfg = _ASSET,
):
    """A standing start (the trigger to come), or a start part-way round the turn.

    Part-way: turned θ in midturn_range about body x in the episode's direction, random yaw,
    legs lerped HOME→spread by a random factor plus joint noise, trunk just clear of the floor
    (SPAWN_Z_TABLE + 8 mm), turning at ω in midturn_omega_range; its pulse is marked long gone,
    the turn frontier starts at θ and the head latch is set (it teaches finishing, not the headstand).
    Draws the episode's direction and hands it to the command (which resets after this event);
    runs after reset_robot_joints (the spread lerps from the HOME pose it wrote)."""
    if env_ids is None or len(env_ids) == 0:
        return
    env_ids = env_ids.to(env.device, dtype=torch.long)
    n = len(env_ids)
    dev = env.device
    asset: Entity = env.scene[asset_cfg.name]
    accum, frontier, paid = _state(env)
    trig = _trigger(env)
    direction = torch.where(torch.rand(n, device=dev) < 0.5, 1.0, -1.0)

    mid = torch.rand(n, device=dev) < midturn_prob / max(standing_prob + midturn_prob, 1e-6)
    trig.preset[env_ids] = True
    trig.preset_direction[env_ids] = direction
    trig.preset_gone[env_ids] = mid
    theta = torch.rand(n, device=dev) * (midturn_range[1] - midturn_range[0]) + midturn_range[0]
    theta = torch.where(mid, theta, torch.zeros_like(theta))

    yaw = torch.rand(n, device=dev) * 2 * math.pi - math.pi
    pitch = (torch.rand(n, device=dev) * 2 - 1) * standing_tilt_max
    roll = (torch.rand(n, device=dev) * 2 - 1) * standing_tilt_max - direction * theta
    cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
    cp, sp = torch.cos(pitch / 2), torch.sin(pitch / 2)
    cr, sr = torch.cos(roll / 2), torch.sin(roll / 2)
    quat = torch.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dim=1)

    table = torch.tensor(SPAWN_Z_TABLE, device=dev)
    pos = (theta % (2 * math.pi)) / math.radians(30.0)
    i = pos.floor().long().clamp(0, len(SPAWN_Z_TABLE) - 2)
    z_mid = table[i] + (pos - i) * (table[i + 1] - table[i]) + 0.008
    z_stand = torch.rand(n, device=dev) * (standing_z_range[1] - standing_z_range[0]) + standing_z_range[0]
    env.sim.data.qpos[env_ids, 2] = torch.where(mid, z_mid, z_stand) + microduck_mdp._env_origin_z(env, env_ids)
    env.sim.data.qpos[env_ids, 3:7] = quat
    env.sim.data.qvel[env_ids, :6] = 0.0

    mid_ids = env_ids[mid]
    if len(mid_ids):
        servo = microduck_mdp._servo_joint_ids(env, asset)
        if spread_overrides:
            u = torch.rand(len(mid_ids), device=dev) * (spread_range[1] - spread_range[0]) + spread_range[0]
            for j, angle in spread_overrides.items():
                col = 7 + servo[j]
                home = env.sim.data.qpos[mid_ids, col]
                env.sim.data.qpos[mid_ids, col] = home + u * (angle - home)
        if joint_noise_std > 0:
            cols = torch.tensor([7 + j for j in servo], device=dev, dtype=torch.long)
            noise = torch.randn(len(mid_ids), len(cols), device=dev) * joint_noise_std
            env.sim.data.qpos[mid_ids.unsqueeze(1), cols.unsqueeze(0)] += noise
        omega = torch.rand(len(mid_ids), device=dev) * (midturn_omega_range[1] - midturn_omega_range[0])
        # Free-joint qvel[3:6] is the body-frame angular velocity: turn about body x.
        env.sim.data.qvel[mid_ids, 3] = -direction[mid] * (omega + midturn_omega_range[0])

    accum[env_ids] = theta
    frontier[env_ids] = theta
    paid[env_ids] = theta
    env._cw_latch[env_ids] = mid


# ── Rewards ────────────────────────────────────────────────────────────────────────────────────
def idle_stand(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """Standing composite (height · upright · HOME pose) until the pulse — the stage's idle.
    Zero once the pulse has started: from then on only turning, then landing, pays."""
    score = microduck_mdp.standing_composite_score(
        env, target_height=target_height, height_std=height_std, upright_std=upright_std,
        pose_std=pose_std, joint_indices=joint_indices, asset_cfg=asset_cfg,
    )
    return score * (~_trigger(env).fired).float()


def turn_progress(
    env: ManagerBasedRlEnv,
    target_angle: float = 2 * math.pi,
    max_paid_rate: float = 5.0,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """Pay increments of the turn frontier up to one full turn, at most max_paid_rate rad/s
    (faster forfeits the excess): the roulade's roulade_progress about body x."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    _, frontier, paid = _state(env)
    new_paid = torch.clamp(frontier, max=target_angle)
    delta = torch.clamp(new_paid - torch.clamp(paid, max=target_angle), min=0.0)
    delta = torch.clamp(delta, max=max_paid_rate * env.step_dt)
    env._cw_paid = torch.maximum(paid, new_paid)
    return delta / (env.step_dt * target_angle)


def head_pivot(
    env: ManagerBasedRlEnv,
    angle_lo: float = math.radians(60.0),
    angle_hi: float = math.radians(300.0),
    rate_norm: float = 2.0,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """Head-floor contact mid-turn × turning rate × (0.3 + 0.7 if it is the flat top): pays for
    pivoting over the head, nothing for resting on it."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    accum, _, _ = _state(env)
    head = microduck_mdp._sensor_any_contact(env, HEAD_SENSOR)
    if head is None:
        return torch.zeros(env.num_envs, device=env.device)
    window = ((accum > angle_lo) & (accum < angle_hi)).float()
    rate = torch.clamp(_turn_rate(env, asset) / rate_norm, 0.0, 1.0)
    top = 0.3 + 0.7 * microduck_mdp._head_top_down(env, asset).float()
    return head.float() * window * rate * top


def landing_composite(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_std: float,
    upright_std: float,
    pose_std: float,
    joint_indices: list,
    gate_lo: float,
    gate_hi: float,
    asset_cfg: SceneEntityCfg = _ASSET,
) -> torch.Tensor:
    """Standing composite once the turn is complete and went over the head: the big annuity."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    score = microduck_mdp.standing_composite_score(
        env, target_height=target_height, height_std=height_std, upright_std=upright_std,
        pose_std=pose_std, joint_indices=joint_indices, asset_cfg=asset_cfg,
    )
    return score * _completion_gate(env, gate_lo, gate_hi)


def upright_after_turn(env: ManagerBasedRlEnv, gate_lo: float, gate_hi: float, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Linear cos(tilt) after completion: a pull toward vertical from any orientation."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    q = asset.data.root_link_quat_w
    upright = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return torch.clamp(upright, min=0.0) * _completion_gate(env, gate_lo, gate_hi)


def height_after_turn(env: ManagerBasedRlEnv, target_height: float, std: float, gate_lo: float, gate_hi: float,
                      asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Broad height Gaussian after completion."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    return torch.exp(-((_height(env, asset) - target_height) / std) ** 2) * _completion_gate(env, gate_lo, gate_hi)


def landing_sharp(env: ManagerBasedRlEnv, target_height: float, height_std: float, upright_std: float,
                  gate_lo: float, gate_hi: float, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Tight upright × height Gaussians after completion: the last mile to standing tall."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    q = asset.data.root_link_quat_w
    upright = torch.exp(-2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2) / upright_std ** 2)
    height = torch.exp(-((_height(env, asset) - target_height) / height_std) ** 2)
    return upright * height * _completion_gate(env, gate_lo, gate_hi)


def stand_tax(env: ManagerBasedRlEnv, target_height: float, gate_lo: float, gate_hi: float,
              asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """SELF-NEGATING (POSITIVE weight): −(height shortfall) after completion — a heap after the
    turn costs, so lying there is never a comfortable end."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    shortfall = torch.clamp(target_height - _height(env, asset), min=0.0)
    return -shortfall * _completion_gate(env, gate_lo, gate_hi)


def rise_velocity(env: ManagerBasedRlEnv, max_height: float, gate_lo: float, gate_hi: float,
                  asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Upward trunk speed below max_height in the turn's second half: end-state rewards have no
    gradient at zero motion, so the rise itself pays."""
    asset: Entity = env.scene[asset_cfg.name]
    _update(env, asset)
    vz = torch.nan_to_num(asset.data.root_link_lin_vel_w[:, 2], nan=0.0)
    rising = torch.clamp(vz, min=0.0) * (_height(env, asset) < max_height).float()
    return rising * _completion_gate(env, gate_lo, gate_hi)


def overspeed(env: ManagerBasedRlEnv, omega_max: float = 7.0, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """max(0, |ω_x| − omega_max)²: a whip costs (positive; NEGATIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    w_x = torch.nan_to_num(asset.data.root_link_ang_vel_b[:, 0], nan=0.0)
    return torch.clamp(w_x.abs() - omega_max, min=0.0) ** 2


def out_of_plane(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """(forward-axis world-z)²: 0 through a clean cartwheel, 1 nose or tail down (NEGATIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(_forward_axis_z(asset.data.root_link_quat_w), nan=0.0) ** 2


def off_axis_rate(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Body-frame ω_y² + ω_z²: turning about anything but the forward axis (NEGATIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    w = asset.data.root_link_ang_vel_b
    return torch.nan_to_num(w[:, 1] ** 2 + w[:, 2] ** 2, nan=0.0)


def forward_drift(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ASSET) -> torch.Tensor:
    """Body-frame forward speed²: a cartwheel travels sideways, never forward (NEGATIVE weight)."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.nan_to_num(asset.data.root_link_lin_vel_b[:, 0] ** 2, nan=0.0)
