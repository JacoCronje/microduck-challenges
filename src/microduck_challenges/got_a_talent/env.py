"""MicroDuck Got a Talent's training environment: Microduck cartwheels on command.

The duck stands still until the timeline pulses the trigger, then turns a cartwheel to that side
(over the flat top of its head: a headstand halfway round) and lands back on its feet:

    twist = [0, trigger, 0]   trigger = +1 for 0.5 s: a cartwheel to its left, -1: to its right.

Built from microduck_rl's roulade task (microduck_roulade_env_cfg.py: the forward roll over the
head, five runs of lessons in its docstring), turned about the body's forward axis; the turn's own
command, spawns and rewards are in rewards.py beside this file. The Arena's parts are the talent
env's: the stage robot (MICRODUCK_STANDUP_ROBOT_CFG), the 61-number observation, the 10/20 contact
solver with a 200-contact buffer, no fall termination, and three reward weights from challenge.toml.

What the roulade taught, kept here:
  • one potential-based progress signal, support-gated (a ballistic flip earns nothing), capped at
    5 rad/s paid (a whip earns less, not the same sooner);
  • landing rewards gated on COMPLETION (and on the head latch), never on a clock, so standing at
    the spawn cannot farm them and no upright pressure opposes the turn;
  • a stand tax after completion, so lying in a heap afterwards is never free;
  • reverse curriculum: episodes also start part-way round, most of them at a moment of a reference
    cartwheel that lands standing (cartwheel_reference.npz, searched in the Arena's own physics);
  • motion-blocking regularisers near zero until the skill exists, introduced late by curriculum.
"""

import math
from copy import deepcopy

# Symmetry: a left cartwheel mirrors into a right one (twist[1] and ω_x both flip).
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to the velocity env for sim2real parity) ────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_KD_RANDOMIZATION              = False  # match velocity (OFF)
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True
# No pushes: a push mid-turn is incoherent (roulade), and the Arena never pushes.

# ── Ranges (matched to the velocity env) ──────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped to 0.015 via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped to 0.01 via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)  # unused (kp DR off)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)    # unused (kd DR off)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode: up to 2.5 s standing before the pulse, a ~2 s turn, a ~1.5 s rise, then standing.
EPISODE_LENGTH_S = 7.0
IDLE_RANGE_S     = (0.5, 2.5)
PULSE_S          = 0.5

# Measured standing trunk height (m).
STAND_Z = 0.115

# Part-way spawns: θ round the turn (90° = on its side, 180° = headstand on the flat top of the
# head, 270° = on the other side), legs spread toward SPREAD_OVERRIDES by a random factor.
MIDTURN_MIN       = math.radians(50.0)
MIDTURN_MAX       = math.radians(330.0)
MIDTURN_OMEGA     = (0.0, 3.0)   # rad/s of turn at the spawn
# Hip roll at its limit (±0.38 rad, the most sideways reach this robot has), outward.
SPREAD_OVERRIDES = {
    1:  -0.35,  # left  hip_roll
    10:  0.35,  # right hip_roll
}

# Turn thresholds (rad) for the completion gates.
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
RISE_GATE_LO    = math.radians(180.0)
RISE_GATE_HI    = math.radians(260.0)

# The standing pose the idle and the landing score: every joint at HOME, the head included. cart1
# scored the legs only and learned to land leaning on its head (trunk at 27°, head 8 cm off the
# floor, never "standing" by the Arena's reading): a free third leg is the more stable rest.
_POSE_JOINTS = list(range(14))

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

from . import rewards as cartwheel


def make_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """MicroDuck Got a Talent's training environment."""

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Head-floor contact (jaw_soft carries the head's collision geoms): the head latch and pivot.
    head_ground_cfg = ContactSensorCfg(
        name=cartwheel.HEAD_SENSOR,
        primary=ContactMatch(mode="body", pattern="jaw_soft", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    # Whole-robot floor contact: the support gate on the turn.
    robot_ground_cfg = ContactSensorCfg(
        name=cartwheel.SUPPORT_SENSOR,
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, head_ground_cfg, robot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: before the pulse, stand ──────────────────────────────────────
    # Same composite as the landing annuity: standing before and after pay alike, and from the
    # pulse until the turn completes nothing but turning pays.
    cfg.rewards["idle_stand"] = RewardTermCfg(
        func=cartwheel.idle_stand,
        weight=knobs["idle_reward"],
        params={
            "target_height": STAND_Z,
            "height_std":    0.04,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "joint_indices": _POSE_JOINTS,
        },
    )

    # ── Rewards: the turn (the roulade's stack about body x) ──────────────────
    cfg.rewards["turn_progress"] = RewardTermCfg(
        func=cartwheel.turn_progress,
        weight=knobs["turn_reward"],
        params={"target_angle": 2 * math.pi, "max_paid_rate": 5.0},
    )
    cfg.rewards["overspeed"] = RewardTermCfg(
        func=cartwheel.overspeed, weight=-0.1, params={"omega_max": 7.0},
    )
    cfg.rewards["head_pivot"] = RewardTermCfg(
        func=cartwheel.head_pivot,
        weight=0.5,
        params={"angle_lo": math.radians(60.0), "angle_hi": math.radians(300.0), "rate_norm": 2.0},
    )

    # Imitation: the reference cartwheel's joint pose at the same point of the turn, 90° to 330°.
    # cart1-3 learned the turn but landed leaning on their heads (27° tilt, knees bent); the
    # reference straightens its legs and comes upright on its way out.
    cfg.rewards["reference_pose"] = RewardTermCfg(
        func=cartwheel.reference_pose,
        weight=3.0,
        params={"reference": "cartwheel_reference.npz", "std": 0.5},
    )

    # ── Rewards: the landing (gated on completion and the head latch) ────────
    gates = {"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI}
    cfg.rewards["landing_composite"] = RewardTermCfg(
        func=cartwheel.landing_composite,
        weight=knobs["landing_reward"],
        params={
            "target_height": STAND_Z,
            "height_std":    0.04,
            "upright_std":   0.40,
            "pose_std":      0.40,
            "joint_indices": _POSE_JOINTS,
            **gates,
        },
    )
    cfg.rewards["upright_after_turn"] = RewardTermCfg(
        func=cartwheel.upright_after_turn, weight=1.5, params=gates,
    )
    cfg.rewards["height_after_turn"] = RewardTermCfg(
        func=cartwheel.height_after_turn, weight=1.0,
        params={"target_height": STAND_Z, "std": 0.04, **gates},
    )
    cfg.rewards["landing_sharp"] = RewardTermCfg(
        func=cartwheel.landing_sharp, weight=2.0,
        params={"target_height": STAND_Z, "height_std": 0.015, "upright_std": 0.3, **gates},
    )
    # POSITIVE weight: the function is self-negating.
    cfg.rewards["stand_tax"] = RewardTermCfg(
        func=cartwheel.stand_tax, weight=5.0, params={"target_height": STAND_Z, **gates},
    )
    cfg.rewards["rise_velocity"] = RewardTermCfg(
        func=cartwheel.rise_velocity, weight=0.75,
        params={"max_height": STAND_Z + 0.01, "gate_lo": RISE_GATE_LO, "gate_hi": RISE_GATE_HI},
    )

    # ── Rewards: stay in the cartwheel's plane ────────────────────────────────
    # The plane gate on the turn makes a twisted roll earn nothing; these give the per-step
    # gradient back toward the plane (the roulade's run-5 values, re-aimed).
    cfg.rewards["out_of_plane"] = RewardTermCfg(func=cartwheel.out_of_plane, weight=-0.5)
    cfg.rewards["off_axis_rate"] = RewardTermCfg(func=cartwheel.off_axis_rate, weight=-0.1)
    cfg.rewards["forward_drift"] = RewardTermCfg(func=cartwheel.forward_drift, weight=-0.5)

    # ── Sim2real regularisers (roulade's: motion-blockers near zero during discovery) ──
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.002   # must stay ≈0: the turn is ω
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards["arrival_damping"] = RewardTermCfg(
        func=microduck_mdp.body_ang_vel_at_height,
        weight=0.0,
        params={
            "height_low":    0.09,
            "height_high":   0.11,
            "tilt_full_deg": 20.0,
            "tilt_zero_deg": 45.0,
            "asset_cfg":     SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    # POSITIVE weight: trunk_vertical_accel_penalty is self-negating (returns -|a_z|).
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )
    # Light: legs may brush the trunk mid-turn.
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )
    # An always-on upright reward would oppose the turn; uprightness lives in the gated terms.
    if "upright" in cfg.rewards:
        del cfg.rewards["upright"]

    # ── Observations (identical layout to walking / sit / standup policies) ───
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # Command slots: head and body zero-padded (the head is part of the move: it is the pivot),
    # keeping the 61-number layout.
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # ── Command: the trigger in twist[1] ──────────────────────────────────────
    command = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs  = 0.0
    command.heading_command   = False
    command.ranges.heading    = None
    # Resampled at reset only: one pulse per episode.
    command.resampling_time_range = (1.0e6, 1.0e6)
    command.debug_vis = False
    cfg.commands["twist"] = cartwheel.CartwheelCommandCfg(
        **{**vars(command), "idle_range_s": IDLE_RANGE_S, "pulse_s": PULSE_S}
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Going upside down is the task, and the stage plays on after a fall: NaN guard + timeout only.
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    # Standing start, part-way round (reverse curriculum), or a moment of the reference cartwheel
    # (cartwheel_reference.npz: a cartwheel that lands standing, found by searching the Arena's own
    # physics, logs/cartwheel/search.py). cart1 and cart2 learned the turn but never found a way
    # back onto their feet; starts along the reference's landing show them one. Must run after
    # reset_robot_joints (dict insertion order): the spread lerps FROM the HOME pose it wrote.
    cfg.events["set_cartwheel_state"] = EventTermCfg(
        func=cartwheel.reset_cartwheel_state,
        mode="reset",
        params={
            "standing_prob":       0.3,
            "midturn_prob":        0.1,
            "reference_prob":      0.4,
            "reference":           "cartwheel_reference.npz",
            # cart2-4 ended their own cartwheels in a head-down tripod (pitched 26° forward, neck at
            # its limit) and, started there, never stood; started from the reference's landing they
            # do. So a fifth of the episodes start in those tripods, turn already complete.
            "stuck_prob":          0.2,
            "stuck":               "cartwheel_stuck.npz",
            "midturn_range":       (MIDTURN_MIN, MIDTURN_MAX),
            "midturn_omega_range": MIDTURN_OMEGA,
            "spread_overrides":    SPREAD_OVERRIDES,
            "spread_range":        (0.0, 1.0),
            "joint_noise_std":     0.08,
        },
    )

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    # The Arena solves every contact with 10/20 iterations; nconmax is a contact buffer, not
    # physics: 200 keeps a heap of limbs on the floor from overflowing it.
    cfg.sim.nconmax = 200
    cfg.sim.mujoco.iterations = 10
    cfg.sim.mujoco.ls_iterations = 20

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_KP_RANDOMIZATION or ENABLE_KD_RANDOMIZATION:
        kp_range = KP_RANDOMIZATION_RANGE if ENABLE_KP_RANDOMIZATION else (1.0, 1.0)
        kd_range = KD_RANDOMIZATION_RANGE if ENABLE_KD_RANDOMIZATION else (1.0, 1.0)
        cfg.events["randomize_motor_gains"] = EventTermCfg(
            func=microduck_mdp.randomize_delayed_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "operation": "scale",
                "kp_range": kp_range,
                "kd_range": kd_range,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    # Flat: the Arena's stage is an open floor.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curriculum ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # Reverse-curriculum mix (roulade's): heavy part-way spawns early, shifting to standing
    # starts once whole turns appear; never zero, so the second half stays practised.
    cfg.curriculum["spawn_mix"] = CurriculumTermCfg(
        func=microduck_mdp.event_param_curriculum,
        params={
            "event_name": "set_cartwheel_state",
            "param_stages": [
                {"step": 0,          "params": {"standing_prob": 0.30, "midturn_prob": 0.10, "reference_prob": 0.40, "stuck_prob": 0.20}},
                {"step": 3000 * 24,  "params": {"standing_prob": 0.40, "midturn_prob": 0.10, "reference_prob": 0.30, "stuck_prob": 0.20}},
                {"step": 6000 * 24,  "params": {"standing_prob": 0.50, "midturn_prob": 0.10, "reference_prob": 0.20, "stuck_prob": 0.20}},
            ],
        },
    )

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.005},
                    {"step": 1000 * 24, "range": 0.01},
                ],
            },
        )

    # Smoothness, tightened only once the turn exists (roulade's stages).
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "action_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": -0.1},
                {"step": 1500 * 24,  "weight": -0.2},
                {"step": 3000 * 24,  "weight": -0.4},
            ],
        },
    )
    cfg.curriculum["arrival_damping_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "arrival_damping",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -0.025},
                {"step": 3500 * 24,  "weight": -0.05},
            ],
        },
    )
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name":   "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,          "weight": 0.0},
                {"step": 2500 * 24,  "weight": -5e-4},
                {"step": 3500 * 24,  "weight": -1e-3},
            ],
        },
    )
    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            # POSITIVE weights: the function is self-negating (returns -|a_z|).
            "reward_name":   "gentle_landing",
            "weight_stages": [
                {"step": 0,          "weight": 0.002},
                {"step": 2500 * 24,  "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

RL_CFG = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # normalizer MUST be baked into ONNX by export.py
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="got_a_talent",  # logs/rsl_rl/got_a_talent/
    run_name="got_a_talent",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=15_000,
)
