"""Does a MicroDuck Got a Talent policy cartwheel on its timeline's pulses on the Arena's stage?

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.check policy.onnx [--timeline FILE]

Plays the stage the way the Arena scores an entry (arena.evaluate, with this folder's timeline.json
by default; for a troupe, the star alone on its own keyframes, since the other ducks are other
policies), then checks the performance against the star's pulses, twist[1] = +1 (a cartwheel to its
left) or -1 (to its right):

- before the first pulse: standing throughout;
- after each pulse: turned at least TURN_DEG about its forward axis in the pulse's direction, and
  standing again within RISE_S of the pulse, held until the next pulse (or the end).

Standing is the Arena's own reading (arena.stage). The event's fall rule fires during any cartwheel
(the duck is upside down), so a fall is reported but only the standing afterwards decides. Prints a
line per pulse; exits 1 when anything fails, including a performance that did not play to the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from arena import stage
from arena.cast import cast_of
from arena.constants import CTRL_DT
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.timeline import Timeline, TimelineError, load_troupe, solo

EVENT = "got-a-talent"
TIMELINE = Path(__file__).with_name("timeline.json")
TURN_DEG = 330.0
RISE_S = 5.0


def _frames(start_s: float, end_s: float) -> slice:
    """The trace frames of the control steps that start in [start_s, end_s)."""
    return slice(round(start_s / CTRL_DT), round(end_s / CTRL_DT))


def pulses(timeline: Timeline) -> list[tuple[float, float]]:
    """(time, direction) of each pulse: where twist[1] turns non-zero."""
    out, before = [], 0.0
    for t, command in zip(timeline.times, timeline.commands):
        if command[1] and not before:
            out.append((t, float(np.sign(command[1]))))
        before = command[1]
    return out


def turn_about_forward(trace) -> np.ndarray:
    """Cumulative rotation (rad) about the trunk's own forward axis, frame by frame; a cartwheel to
    the left is negative."""
    q = trace[:, 3:7].astype(np.float64)
    w0, v0 = q[:-1, 0], q[:-1, 1:]
    w1, v1 = q[1:, 0], q[1:, 1:]
    # r = conj(q0) ⊗ q1: the step's rotation in the body frame.
    w = w0 * w1 + (v0 * v1).sum(axis=1)
    v = w0[:, None] * v1 - w1[:, None] * v0 - np.cross(v0, v1)
    v = np.where(w[:, None] < 0, -v, v)
    w = np.abs(w)
    norm = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(norm, w)
    step_x = np.where(norm > 1e-12, angle * v[:, 0] / np.maximum(norm, 1e-12), 0.0)
    return np.concatenate([[0.0], np.cumsum(step_x)])


def report(trace, timeline: Timeline, termination) -> tuple[list[str], bool, int]:
    """A line per stretch then the fall line, whether everything passed, and the turns completed."""
    standing = stage.standing(trace)
    turn = turn_about_forward(trace)
    marks = pulses(timeline)
    ends = [t for t, _ in marks[1:]] + [timeline.duration_s]
    lines, passed, done = [], True, 0

    first = marks[0][0] if marks else timeline.duration_s
    still = standing[_frames(0.0, first)]
    if still.all():
        lines.append(f"t= 0.00  stand: held until {first:.2f} s")
    else:
        left = int(np.argmin(still)) * CTRL_DT
        lines.append(f"t= 0.00  stand: left at {left:.2f} s, before any pulse")
        passed = False

    for (t, direction), end in zip(marks, ends):
        name = "left" if direction > 0 else "right"
        window = _frames(t, min(t + RISE_S, end))
        start = round(t / CTRL_DT)
        turned = -direction * (turn[window] - turn[start])
        reached = np.nonzero(turned >= np.radians(TURN_DEG))[0]
        best = float(np.degrees(turned.max())) + 0.0 if len(turned) else 0.0  # + 0.0: never "-0°"
        if not len(reached):
            lines.append(f"t={t:5.2f}  {name}: turned {best:.0f}°, not the {TURN_DEG:.0f}° of a cartwheel")
            passed = False
            continue
        after = start + int(reached[0])
        up = standing[after:window.stop]
        if not up.any():
            lines.append(f"t={t:5.2f}  {name}: turned {best:.0f}°, not standing again within {RISE_S:g} s")
            passed = False
            continue
        stood = after + int(np.argmax(up))
        rest = standing[stood:round(end / CTRL_DT)]
        if not rest.all():
            left = (stood + int(np.argmin(rest))) * CTRL_DT
            lines.append(f"t={t:5.2f}  {name}: turned {best:.0f}°, standing after {stood * CTRL_DT - t:.2f} s, left at {left:.2f} s")
            passed = False
            continue
        done += 1
        lines.append(f"t={t:5.2f}  {name}: turned {best:.0f}°, standing after {stood * CTRL_DT - t:.2f} s, held until {end:.2f} s")
    fell = stage.fell_at(trace, termination)
    lines.append("no fall" if fell is None else f"fall rule first held at {fell:.2f} s (upside down mid-turn is expected)")
    return lines, passed, done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a MicroDuck Got a Talent policy's cartwheels on the Arena's stage.")
    parser.add_argument("onnx")
    parser.add_argument("--timeline", default=str(TIMELINE))
    args = parser.parse_args(argv)
    event = load_event(EVENT)
    try:
        # The workshop's own file may name checkpoints of this machine; only the star plays here.
        timeline = load_troupe(Path(args.timeline), event, allow_local=True).ducks[0].timeline
        policy = load_policy(Path(args.onnx))
        result = evaluate(event, policy, cast_of(event, solo(timeline), policy, None, {}).performers)
    except (TimelineError, PolicyError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    lines, passed, done = report(result.trace[0], timeline, event.termination)
    passed = passed and result.seeds_finished == 1
    print("\n".join(lines))
    print(f"{'PASS' if passed else 'FAIL'}: {done}/{len(pulses(timeline))} cartwheels")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
