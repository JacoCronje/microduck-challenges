"""The 2 m Roller Sprint's training environment: Microduck on roller skates.

A complete copy of microduck_rl's roller velocity task
(microduck_velocity_rollers_env_cfg.py), made the roller sprint's own: the two
rewards a novice tunes are read from challenge.toml ([params]). Everything here
is yours to change; the robot (MICRODUCK_WALK_ROLLERS_ROBOT_CFG) and the reward
functions (microduck_mdp) come from mjlab_microduck and are not edited here.

MIGRATED to mjlab 1.3.0 + canonical BAM (2026-07), matching the velocity env's
sim2real machinery, and updated for the NEW roller model:

  - `get_walk_rollers_spec` now loads `robot_groundcontact_rollers.xml`
    (it silently loaded the wheel-less standup model before): 14 actuated
    joints + 4 passive wheels (passive_{L,R}{F,R}wheel), two per blade,
    INTERSPERSED in the joint order (after each ankle) — everything resolves
    joints by NAME, never by index.
  - Legs run the canonical BAM actuator like every other variant (was a plain
    XML PD — an actuator-physics mismatch, and no joint-friction DR).
  - Obs migrated to the unified 61D layout (twist + zero-padded head/body
    command slots) so roller policies load through the runtime's
    --new-cmd-obs path. Symmetry was OFF then (SYMMETRY_CFG was hardcoded
    for the old 51D layout); it is on again since race1 (see ENABLE_SYMMETRY).
  - DR/noise/delays matched to the velocity env's FIXED (non-accumulating,
    per-env-verified) versions. Wheel-bearing frictionloss DR was REMOVED: the
    Arena's roller model's wheels are frictionless, and training against added
    bearing friction breaks the policy once raced there.

Race shaping (race1, rewards.py; the 2 m Sprint's race recipe on skates). The
Arena runs the policy through its 0.5 s settle with the clock stopped, then
times 2 m from wherever the duck is: the leaders skate through the start line
at 1.3 m/s and cross the finish at 2.4-2.5 m/s, the trunk pitched 15-20° (up to
50° at the line), hip_roll swinging between its ±0.38 rad stops, the neck thrown
forward. The source task's calm skate (wheel_speed's tanh saturating at
0.3 m/s, glide/single_support/air-time shaping, action_rate ramped to -2) gave
the reference's 0.65 m/s. Everything that held the duck to it is gone:
forward_speed pays the race's progress whatever the command, upright is a tilt
guard that only drains past 45°, the head and hips are free, action_rate stays
light, episodes are a 3 s race without pushes.
"""

import math
from copy import deepcopy

# Symmetry (the sprint's race9): a mirror loss makes the policy act as its own
# mirror image on mirrored observations, so its stroke cannot carry a built-in
# sideways lean. The actor cannot see its heading or lane position, so no reward
# can correct a drift once started. SYMMETRY_CFG covers the 61D layout since
# mjlab_microduck's 2026-08 migration.
ENABLE_SYMMETRY = True

# ── Domain randomisation toggles (matched to the velocity env) ────────────────
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True   # BAM friction budget per-env (legs)
ENABLE_ARMATURE_RANDOMIZATION        = True   # legs only — NOT the wheel bearings
ENABLE_VELOCITY_PUSHES               = False  # the Arena races without pushes; a 3 s race has no room to recover
# race2: off. The Arena's IMU is mounted true; a random 6° misalignment hid the
# trunk's pitch from the policy by as much, and race1 leaned 5-8° where the
# leaders lean 15-20°.
ENABLE_IMU_ORIENTATION_RANDOMIZATION = False  # obs-level per-env rotation
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to the velocity env unless roller-specific) ───────────────
COM_RANDOMIZATION_RANGE          = 0.003   # ±3mm initial, ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE     = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE = (0.95, 1.05)
JOINT_FRICTION_RANDOMIZATION_RANGE = (0.9, 1.1)
ARMATURE_RANDOMIZATION_RANGE     = (0.9, 1.1)
VELOCITY_PUSH_INTERVAL_S         = (3.0, 6.0)
VELOCITY_PUSH_RANGE              = (-0.2, 0.2)  # roller-specific: gentler than walk ±0.3
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0
ENCODER_BIAS_RANGE               = (-0.015, 0.015)

# The race: the Arena's 2 m take the leaders about 1 s after the 0.5 s settle,
# so an episode is a 3 s sprint from the stand, not a 20 s skate.
NUM_STEPS_PER_ENV = 24
RACE_EPISODE_S = 3.0
RACE_MAX_SPEED = 3.0  # m/s, forward_speed's cap: above the leaders' 2.5

# The head assembly's bodies, for the head CoM randomization. The head-roll
# body is named bottom_head_shell in the walk model and jaw_soft in the
# 2026-07 roller model, hence the alternation. NOTE: bearing_roll is NOT a
# head body — in both models it is the right-hip-yaw link (child of
# trunk_base); it has always been listed here by mistake and is kept only to
# preserve existing DR behavior.
HEAD_BODY_NAMES = (
    "neck",
    "neck_pitch",
    "yaw_roll_motion",
    "(bottom_head_shell|jaw_soft)",
    "bearing_roll",
)

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
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
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck import challenge
from mjlab_microduck.robot.microduck_constants import MICRODUCK_WALK_ROLLERS_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG

from . import rewards as roller_rewards


def heading_error(env: ManagerBasedRlEnv) -> torch.Tensor:
    """How far the trunk has turned from the heading heading_hold holds (rad, wrapped).

    Critic-only: the Arena never tells the actor its heading. Reads the spawn
    heading heading_hold_reward captures (env._heading_ref); zero on an
    episode's first observation, which comes before that reward captures it.
    """
    w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    reference = torch.where(env.episode_length_buf == 0, yaw, getattr(env, "_heading_ref", yaw))
    error = yaw - reference
    return torch.atan2(torch.sin(error), torch.cos(error)).unsqueeze(-1)


def make_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """The roller sprint's training environment."""

    # The knobs a novice tunes, from challenge.toml beside this file.
    knobs = challenge.params(__file__)

    # passive_.*: 999.0 → passive wheel joints are matched but effectively ignored
    std_standing = {
        r".*hip_yaw.*": 0.05,
        r".*hip_roll.*": 0.05,
        r".*hip_pitch.*": 0.05,
        r".*knee.*": 0.05,
        r".*ankle.*": 0.05,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_walking = {
        r".*hip_yaw.*": 0.3,
        r".*hip_roll.*": 0.6,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.4,
        r".*knee.*": 0.4,
        r".*ankle.*": 0.25,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    std_running = {
        r".*hip_yaw.*": 0.5,
        r".*hip_roll.*": 0.8,  # loosened: skating requires wide lateral push
        r".*hip_pitch.*": 0.8,
        r".*knee.*": 0.8,
        r".*ankle.*": 0.5,
        r".*neck.*": 0.05,
        r".*head.*": 0.05,
        r".*passive_.*": 999.0,
    }

    # 2026-07 model: the roller_blade bodies were merged into the ankles (blade
    # mesh is now a visual geom on ankle_{l,r}_v1); the tires hang directly off
    # the ankles. Each ankle subtree's only collision geoms are its two tires,
    # so this keeps the old per-foot semantics: 2 slots, left first.
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(ankle_l_v1|ankle_r_v1)$",
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

    cfg = make_velocity_env_cfg()

    # Robot setup
    cfg.scene.entities = {"robot": MICRODUCK_WALK_ROLLERS_ROBOT_CFG}
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"

    # Action configuration
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0
    # NOTE: an env-side action clip was tried here to bound the target, but the
    # deployment pipeline (infer_policy.py) does NOT clip → the clip would only
    # exist in sim, a train/deploy mismatch. The over-command deterrent lives
    # policy-side instead (action_over_limit reward below), baked into the network
    # so it transfers with the ONNX.

    # === REWARDS ===
    keep = {"pose", "upright", "body_ang_vel", "angular_momentum", "action_rate_l2"}
    for name in list(cfg.rewards.keys()):
        if name not in keep:
            del cfg.rewards[name]

    # Pose on the legs only, weight 2.0 -> 0.5 (the sprint's race value): a
    # motion-blocker for a race. The head is free (see the neck terms below).
    def legs(std):
        return {k: v for k, v in std.items() if not any(n in k for n in ("neck", "head", "passive_"))}

    cfg.rewards["pose"].params["std_standing"] = legs(std_standing)
    cfg.rewards["pose"].params["std_walking"] = legs(std_walking)
    cfg.rewards["pose"].params["std_running"] = legs(std_running)
    cfg.rewards["pose"].params["walking_threshold"] = 0.01
    cfg.rewards["pose"].params["running_threshold"] = 0.5
    cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=(r"^(?!passive_|.*neck.*|.*head.*).*",)
    )
    cfg.rewards["pose"].weight = 0.5

    # upright is a tilt guard: free up to 45°, drained to 0 at 60° (the Arena's
    # fall line); leaning back free only to 5°, drained at 20° (the sprint's
    # race3: leaning back is a braking posture). The old Gaussian held the trunk
    # level; the leaders skate pitched 15-20° and up to 50° at the line.
    cfg.rewards["upright"] = RewardTermCfg(
        func=roller_rewards.tilt_guard,
        weight=knobs["upright_reward"],
        params={"free_deg": 45.0, "zero_deg": 60.0, "back_free_deg": 5.0, "back_zero_deg": 20.0},
    )

    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    # Action smoothness held at -0.1 (the sprint's race value; was -1.0 ramped to
    # -2.0 for a calm skate). The sprint's race4 tried -0.02 from scratch: a
    # jittery gait, 3x the falls.
    cfg.rewards["action_rate_l2"].weight = -0.1

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": "self_collision"},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torques_l2, weight=-1e-3
    )
    # Deter OVER-COMMANDING a joint past its hard stop: hip_roll's ±0.38 rad
    # limit vs the ±10 rad ctrlrange let the low-kp servo be commanded far past
    # the stop and slam it with max torque. This penalises only the COMMAND
    # beyond (limit + 0.3 overshoot), so the joint keeps its full reachable range.
    cfg.rewards["action_over_limit"] = RewardTermCfg(
        func=microduck_mdp.action_over_limit_penalty,
        weight=-0.5,
        params={"action_name": "joint_pos", "overshoot": 0.3},
    )

    # The race itself: speed along the lane, whatever the command, from the
    # first step (the Arena's settle is a free run-up). At 2.5 m/s it pays
    # 25/step at the default weight, the bulk of the stack. The knob that set
    # wheel_speed's weight sets this one.
    cfg.rewards["forward_speed"] = RewardTermCfg(
        func=roller_rewards.forward_speed,
        weight=knobs["speed_reward"],
        params={"max_speed": RACE_MAX_SPEED},
    )

    # Sideways speed costs (the sprint's race7): off while a stroke is
    # discovered, ramped in by iter 1000 (curriculum lateral_drift_weight below).
    # Training has no walls; the Arena's lane is 0.8 m wide.
    cfg.rewards["lateral_drift"] = RewardTermCfg(
        func=roller_rewards.lateral_drift,
        weight=0.0,
    )

    # Skate straight. The Arena commands yaw 0 and never tells the policy its
    # heading: heading_hold prices the yaw ANGLE against the spawn heading
    # (corrective, unlike a yaw-rate penalty, which froze the yaw and made
    # drift worse). The critic sees the heading error (heading_error, below).
    # std 0.4 -> 0.2, weight 1 -> 2 (the sprint's race8): at 0.2 a 10° turn
    # keeps 47% (83% at 0.4).
    cfg.rewards["heading_hold"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=2.0,
        params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot")},
    )

    # Gone with the calm skate: wheel_speed (forward_speed pays the race),
    # braking (the Arena never brakes), skating_air_time, glide, single_support,
    # gait_symmetry (the mirror loss balances the legs), forward_lean (the tilt
    # guard leaves the lean free), com_height_target, feet_flat, hip_roll_neutral
    # (the leaders swing hip_roll stop to stop) and the neck terms (the leaders
    # throw the head forward as a counterweight; the Arena never judges it).

    # === TERMINATIONS ===
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # === EVENTS ===
    # BAM (mjlab_frictionloss branch) writes per-env dof_frictionloss/dof_damping
    # every step; this no-op event registers those fields for per-world expansion.
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )

    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )

    del cfg.events["foot_friction"]  # wheels roll; ground friction lives in the XML

    if ENABLE_VELOCITY_PUSHES:
        cfg.events["push_robot"] = EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=VELOCITY_PUSH_INTERVAL_S,
            params={
                "velocity_range": {"x": VELOCITY_PUSH_RANGE, "y": VELOCITY_PUSH_RANGE},
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

    cfg.events["reset_base"].params["pose_range"]["z"] = (0.1335, 0.1435)

    # No wheel-bearing friction DR: the Arena's roller model's wheels are
    # frictionless (frictionloss="0.0" in robot_allcollisions_rollers.xml's
    # passive_joint default). A 3000-iteration reference run that ramped bearing
    # frictionloss in from iteration 2000 stayed fast and consistent (2.99-3.22 s,
    # 5/5 seeds) through checkpoint 2000, then went erratic from checkpoint 2250
    # on (40 s 0/5, 17.7 s, 3.5 s, 24.2 s 3/5) once raced on the Arena's free wheels.

    # ── DR matched to the velocity env's FIXED versions ───────────────────────
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

    if ENABLE_ARMATURE_RANDOMIZATION:
        # Legs/head only — the wheel bearings' tiny armature is excluded; the
        # Arena's wheels are frictionless and get no DR at all (see above).
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    # === OBSERVATIONS (unified 61D layout) ===
    del cfg.observations["actor"].terms["base_lin_vel"]
    # 1.3.0 base template adds sensor-based foot_height + height_scan; the roller
    # env has no terrain-height sensor.
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel,
        scale=1.0,
    )

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )
    # IMU delay 0-1 control steps (matches velocity: the real dxl IMU path is fast)
    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    # Observation noise — matched to the velocity env
    cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.25, n_max=0.25)

    # IMU mounting-misalignment DR (obs-level, actor only — matches velocity)
    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    # 1-ctrl-step lag on joint_vel (Dynamixel present_velocity moving average)
    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    # Exclude the passive wheel joints from joint_pos/vel obs (obs dim 14, matches
    # the action space). Deepcopy per group so the encoder-bias `biased` flag
    # below applies to the actor only.
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

    # Privileged wheel speeds for the critic (4 wheels in the new model).
    wheel_cfg = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel",))
    cfg.observations["critic"].terms["wheel_vel"] = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        scale=1.0,
        params={"asset_cfg": wheel_cfg},
    )
    # Privileged too: the heading error heading_hold prices.
    cfg.observations["critic"].terms["heading_error"] = ObservationTermCfg(func=heading_error)

    # Command obs parity with the 61D family layout: head/body slots zero-padded
    # (the roller task drives heading through the twist slot instead).
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # === COMMANDS ===
    command: UniformVelocityCommandCfg = cfg.commands["twist"]
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.heading_command = False  # RelativeHeadingVelocityCommand handles heading internally
    command.ranges.heading = None    # must be None when heading_command=False
    # The ranges are the Arena event's own (arena/events/roller-sprint-2m.toml):
    # straight ahead, the event's sweep of commanded speeds. forward_speed pays
    # the same whatever the command, so the policy learns to ignore it, and
    # every command the sweep tries is one it has seen. History: wheel_speed
    # grew with the command, and training above 0.6 broke learning twice; the
    # calm skate's 0.6 top and its braking range (< 0) went with it.
    command.ranges.lin_vel_x = (0.30, 2.00)
    command.ranges.lin_vel_y = (0.0, 0.0)
    # ang_vel_z range is the clip limit for cmd[2] = heading error (rad).
    # Set to 0 → cmd[2] is always 0 → no turning demand (straight-line focus).
    command.ranges.ang_vel_z = (0.0, 0.0)
    command.viz.z_offset = 0.5
    cfg.commands["twist"] = microduck_mdp.RelativeHeadingVelocityCommandCfg(**vars(command))

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # === CURRICULUM ===
    del cfg.curriculum["terrain_levels"]
    del cfg.curriculum["command_vel"]

    # The race: a short episode from the stand (see RACE_EPISODE_S).
    cfg.episode_length_s = RACE_EPISODE_S

    # No action_rate ramp (it reached -2.0 by iter 500 for a calm skate).

    # lateral_drift: off until a stroke exists, then up to -6 (the sprint's race8).
    cfg.curriculum["lateral_drift_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "lateral_drift",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 500 * NUM_STEPS_PER_ENV, "weight": -2.0},
                {"step": 750 * NUM_STEPS_PER_ENV, "weight": -4.0},
                {"step": 1000 * NUM_STEPS_PER_ENV, "weight": -6.0},
            ],
        },
    )

    # CoM randomization curricula — velocity's ramp, capped lower for the
    # balance-sensitive skating task (audit lesson: ±30 mm forced a nervous
    # gait on the walker; skates are even less forgiving).
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * NUM_STEPS_PER_ENV,  "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
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
                    {"step": 500 * NUM_STEPS_PER_ENV,  "range": 0.005},
                    {"step": 1000 * NUM_STEPS_PER_ENV, "range": 0.01},
                ],
            },
        )

    return cfg


RL_CFG = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,  # matches the family; normalizer baked into ONNX by export.py
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
        entropy_coef=0.03,  # roller-specific: higher exploration than the walk envs
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
    experiment_name="roller_sprint_2m",  # logs/rsl_rl/roller_sprint_2m/
    run_name="roller_sprint_2m",
    save_interval=250,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=50_000,
)
