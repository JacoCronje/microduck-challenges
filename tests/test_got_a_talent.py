"""MicroDuck Got a Talent trains in the Arena's physics, and reads the command a timeline carries."""

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

import microduck_challenges  # noqa: F401  (registers every task)
from arena.assets import ASSET_ROOT
from arena.constants import CTRL_DT, DECIMATION, DEFAULT_POSE, JOINT_NAMES, TRACE_WIDTH
from arena.course import compile_course
from arena.manifest import load_event
from arena.timeline import Timeline, load_troupe
from mjlab_microduck.tasks import mdp as microduck_mdp


@pytest.fixture(scope="module")
def cfg():
    return importlib.import_module("microduck_challenges.got_a_talent.env").make_env_cfg()


@pytest.fixture(scope="module")
def arena():
    return compile_course(load_event("got-a-talent"))


def _training_model(entity):
    """The robot as training compiles it: its MJCF with the entity's collision
    settings applied (mjlab's Entity runs each CollisionCfg.edit_spec on the spec)."""
    spec = entity.spec_fn()
    for collision in entity.collisions:
        collision.edit_spec(spec)
    return spec.compile()


def _body(model, geom):
    return model.body(int(model.geom_bodyid[geom])).name


def _robot_geoms(model, skip):
    return [g for g in range(model.ngeom)
            if (model.geom_contype[g] or model.geom_conaffinity[g]) and _body(model, g) not in skip]


def _colliding(model, skip):
    """(body, geom type) of every geom that can collide, bodies in `skip` left out."""
    return sorted((_body(model, g), int(model.geom_type[g])) for g in _robot_geoms(model, skip))


# mjlab's flat terrain (terrain_entity._import_ground_plane) is a plane with
# MuJoCo's default contact: priority 0, condim 3, sliding friction 1.
FLOOR = (0, 3, 1.0)


def _floor_contacts(model, skip):
    """(body, condim, sliding friction) of each robot geom's contact with that floor.
    MuJoCo pairs two geoms by priority first; at equal priority it takes the larger
    condim and the larger friction."""
    out = []
    for g in _robot_geoms(model, skip):
        priority, condim, friction = int(model.geom_priority[g]), int(model.geom_condim[g]), float(model.geom_friction[g][0])
        if priority > FLOOR[0]:
            pair = (condim, friction)
        elif priority < FLOOR[0]:
            pair = FLOOR[1:]
        else:
            pair = (max(condim, FLOOR[1]), max(friction, FLOOR[2]))
        out.append((_body(model, g), *pair))
    return sorted(out)


def test_got_a_talent_collides_like_the_arena(cfg, arena):
    # A sit puts the trunk and hips on the floor: training must collide with the
    # same robot geoms as the Arena's stage, or a seated trunk sinks in one and
    # not the other.
    robot = _training_model(cfg.scene.entities["robot"])
    assert _colliding(robot, skip={"world"}) == _colliding(arena, skip={"world", "course"})


def test_got_a_talent_touches_the_floor_like_the_arena(cfg, arena):
    """Every robot geom meets the floor with the Arena's condim and friction.
    Robot-on-robot contacts are where the two differ: training gives its non-foot
    geoms condim 1 (FULL_COLLISION), so two of them touch without friction, and
    the Arena's are all condim 3."""
    floor = arena.geom("floor").id
    assert (int(arena.geom_priority[floor]), int(arena.geom_condim[floor]), float(arena.geom_friction[floor][0])) == FLOOR
    robot = _training_model(cfg.scene.entities["robot"])
    assert _floor_contacts(robot, skip={"world"}) == _floor_contacts(arena, skip={"world", "course"})


def test_got_a_talent_solves_contacts_like_the_arena(cfg, arena):
    assert cfg.sim.mujoco.iterations == arena.opt.iterations
    assert cfg.sim.mujoco.ls_iterations == arena.opt.ls_iterations
    assert math.isclose(cfg.sim.mujoco.timestep, arena.opt.timestep)
    assert cfg.decimation == DECIMATION


def test_got_a_talent_plays_on_after_a_fall_like_the_stage(cfg):
    assert "fell_over" not in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_got_a_talent_reads_what_a_timeline_carries(cfg):
    """twist[1] is the cartwheel trigger, head and body zeros: a keyframe's 13 numbers, in the
    61-D observation's command slot."""
    from microduck_challenges.got_a_talent import rewards

    assert isinstance(cfg.commands["twist"], rewards.CartwheelCommandCfg)
    actor = cfg.observations["actor"].terms
    assert list(actor)[-3:] == ["command", "head_command", "body_command"]
    assert actor["head_command"].func is microduck_mdp.zero_command_padding
    assert actor["head_command"].params == {"dim": 4}
    assert actor["body_command"].func is microduck_mdp.zero_command_padding
    assert actor["body_command"].params == {"dim": 6}


def test_a_left_cartwheel_mirrors_into_a_right_one():
    """The mirror loss negates twist[1] and the roll rate together, so one direction trains both."""
    from mjlab_microduck.tasks.symmetry import _OBS_SIGN

    assert _OBS_SIGN[49] == -1.0 and _OBS_SIGN[0] == -1.0


TIMELINE = Path(microduck_challenges.__file__).parent / "got_a_talent" / "timeline.json"


def test_the_reference_timeline_plays_on_the_arena():
    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    assert timeline.duration_s == 20.0
    assert len(timeline.times) == 7


def test_the_reference_timeline_stays_inside_what_training_saw():
    """Pulses of ±1 lasting the trained pulse, the rest zeros, each pulse after at least the
    shortest trained idle and a whole training episode after the one before."""
    from microduck_challenges.got_a_talent import check, env

    timeline = load_troupe(TIMELINE, load_event("got-a-talent")).ducks[0].timeline
    for command in timeline.commands:
        assert command[0] == 0.0 and command[2:] == (0.0,) * 11
        assert command[1] in (-1.0, 0.0, 1.0)
    marks = check.pulses(timeline)
    assert marks and marks[0][0] >= env.IDLE_RANGE_S[0]
    for t, _ in marks:
        i = timeline.times.index(t)
        assert math.isclose(timeline.times[i + 1] - t, env.PULSE_S)
    starts = [t for t, _ in marks]
    assert all(b - a >= env.EPISODE_LENGTH_S - env.IDLE_RANGE_S[1] + env.PULSE_S for a, b in zip(starts, starts[1:]))


STAND = (0.0,) * 13
LEFT = (0.0, 1.0) + (0.0,) * 11
EIGHT_S = Timeline(8.0, (0.0, 2.0, 2.5), (STAND, LEFT, STAND))


def _trace(turn_at=lambda t: 0.0, z_at=lambda t: 0.115, duration_s=8.0):
    """A synthetic performance: the trunk turned turn_at(t) degrees about its forward axis (a left
    cartwheel is negative), at height z_at(t)."""
    n = round(duration_s / CTRL_DT)
    trace = np.zeros((n, TRACE_WIDTH), dtype=np.float32)
    for i in range(n):
        t = i * CTRL_DT
        half = np.radians(turn_at(t)) / 2
        trace[i, 2] = z_at(t)
        trace[i, 3:7] = (np.cos(half), np.sin(half), 0.0, 0.0)
        trace[i, 7:] = DEFAULT_POSE
    return trace


def _cartwheel(start=2.2, length=1.5, degrees=-360.0):
    return lambda t: degrees * min(max((t - start) / length, 0.0), 1.0)


def _low_while_turning(start=2.2, length=1.5):
    return lambda t: 0.06 if start <= t < start + length + 0.3 else 0.115


def _report(trace):
    from microduck_challenges.got_a_talent import check

    return check.report(trace, EIGHT_S, load_event("got-a-talent").termination)


def test_a_cartwheel_that_follows_its_timeline_passes():
    lines, passed, done = _report(_trace(_cartwheel(), _low_while_turning()))
    assert passed, lines
    assert done == 1
    assert "left: turned 360°" in lines[1] and "held until 8.00 s" in lines[1]


def test_no_cartwheel_fails():
    lines, passed, done = _report(_trace())
    assert not passed and done == 0
    assert "turned 0°, not the 330° of a cartwheel" in lines[1]


def test_a_cartwheel_the_wrong_way_fails():
    lines, passed, _ = _report(_trace(_cartwheel(degrees=360.0), _low_while_turning()))
    assert not passed
    assert "not the 330° of a cartwheel" in lines[1]


def test_a_cartwheel_that_ends_lying_down_fails():
    lines, passed, _ = _report(_trace(_cartwheel(), lambda t: 0.115 if t < 2.2 else 0.05))
    assert not passed
    assert "not standing again within 5 s" in lines[1]


def test_leaving_the_stand_before_the_pulse_fails():
    lines, passed, _ = _report(_trace(_cartwheel(start=1.0), _low_while_turning(start=1.0)))
    assert not passed
    assert lines[0].startswith("t= 0.00  stand: left at")


def test_check_plays_the_stage_and_fails_a_walker(capsys):
    """End to end on the Arena: the reference walker does not cartwheel."""
    from microduck_challenges.got_a_talent import check

    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx")]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 + 3 + 2  # the stand, a line per pulse, the fall line, the verdict
    assert out[-1] == "FAIL: 0/3 cartwheels"


def test_check_reads_a_troupes_star_alone(tmp_path, capsys):
    """A troupe's other ducks are other policies: the check plays the star alone, on its own keyframes."""
    import json

    from microduck_challenges.got_a_talent import check

    reference = json.loads(TIMELINE.read_text())
    troupe = tmp_path / "timeline.json"
    troupe.write_text(json.dumps({"timeline_version": 2, "duration_s": reference["duration_s"], "ducks": [
        {"policy": "self", "keyframes": reference["keyframes"]},
        {"policy": "local:got_a_talent/first@1", "start": {"x": 0.0, "y": 0.6, "yaw": 0}, "keyframes": [{"t": 0.0}]},
    ]}))
    assert check.main([str(ASSET_ROOT / "policies" / "BEST_alpha_walking.onnx"), "--timeline", str(troupe)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 + 3 + 2


def test_check_refuses_a_malformed_timeline(tmp_path, capsys):
    """main() must fail cleanly, like the Arena's own CLI, instead of raising."""
    from microduck_challenges.got_a_talent import check

    bad = tmp_path / "timeline.json"
    bad.write_text("{nope")
    assert check.main(["policy.onnx", "--timeline", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "timeline.json" in err and "not a JSON file" in err


def test_the_manifest_says_what_the_policy_reads():
    """publish writes this into every got-a-talent manifest: the trigger, never 'unused'."""
    from mjlab_microduck import challenge as ch

    import microduck_challenges  # noqa: F401 — registers every challenge's task

    command = ch.for_task("Mjlab-GotATalent-MicroDuck").command
    assert command["twist"].startswith("[0, cartwheel trigger")
