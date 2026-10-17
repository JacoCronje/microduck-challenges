"""Does a MicroDuck Got a Talent policy perform its timeline on the Arena's stage?

    uv run --frozen --group workshop python -m microduck_challenges.got_a_talent.check policy.onnx [--timeline FILE]

Plays the stage the way the Arena scores an entry (arena.evaluate, with this
folder's timeline.json by default; for a troupe, the star alone on its own
keyframes, since the other ducks are other policies), then checks the
performance against the star's timeline, keyframe by keyframe:

- no fall, by the event's own rule;
- at the start, and after each change of the sit flag (twist.vx): seated
  within 1.5 s when it is now 1, or standing upright within 1.5 s when it is
  now 0, and held until the flag changes again;
- every keyframe held 2 s or more: each head joint within 0.25 rad of the
  default pose plus the keyframe's head command, over the hold's last 0.5 s.

Seated and standing are the Arena's own readings (arena.stage). Prints a line
per keyframe; exits 1 when anything fails, including a performance that did
not play to the end.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from arena import stage
from arena.constants import CTRL_DT, DEFAULT_POSE, JOINT_NAMES
from arena.cast import cast_of
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.policy import PolicyError, load_policy
from arena.timeline import Timeline, TimelineError, load_troupe, solo

EVENT = "got-a-talent"
TIMELINE = Path(__file__).with_name("timeline.json")
REACH_S = 1.5
HEAD_HOLD_S = 2.0
HEAD_WINDOW_S = 0.5
HEAD_TOLERANCE_RAD = 0.25
HEAD_JOINTS = [JOINT_NAMES.index(n) for n in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")]


def _frames(start_s: float, end_s: float) -> slice:
    """The trace frames of the control steps that start in [start_s, end_s)."""
    return slice(round(start_s / CTRL_DT), round(end_s / CTRL_DT))


def _flag_segments(timeline: Timeline) -> list[tuple[float, float, bool]]:
    """(start, end, sit) for each stretch where the sit flag keeps one value."""
    segments: list[tuple[float, float, bool]] = []
    for t, command in zip(timeline.times, timeline.commands):
        sit = command[0] >= 0.5
        if segments and segments[-1][2] == sit:
            continue
        if segments:
            segments[-1] = (segments[-1][0], t, segments[-1][2])
        segments.append((t, timeline.duration_s, sit))
    return segments


def _posture(trace, start_s: float, end_s: float, sit: bool) -> tuple[str, bool]:
    name = "sit" if sit else "stand"
    window = (stage.seated(trace) if sit else stage.standing(trace))[_frames(start_s, end_s)]
    reach = window[: round(REACH_S / CTRL_DT)]
    if not reach.any():
        return f"{name}: not reached within {REACH_S:g} s", False
    first = int(np.argmax(reach))
    if not window[first:].all():
        left = start_s + (first + int(np.argmin(window[first:]))) * CTRL_DT
        return f"{name}: reached after {first * CTRL_DT:.2f} s, left at {left:.2f} s", False
    return f"{name}: reached after {first * CTRL_DT:.2f} s, held until {end_s:.2f} s", True


def _head(trace, command, start_s: float, end_s: float) -> tuple[str, bool, float]:
    if end_s - start_s < HEAD_HOLD_S:
        return f"head: held under {HEAD_HOLD_S:g} s, not checked", True, 0.0
    angles = trace[_frames(end_s - HEAD_WINDOW_S, end_s)][:, [7 + j for j in HEAD_JOINTS]]
    if not len(angles):
        return "head: the performance ended before this hold", False, float("inf")
    target = DEFAULT_POSE[HEAD_JOINTS] + np.asarray(command[3:7])
    worst = float(np.abs(angles.mean(axis=0) - target).max())
    return f"head: off by {worst:.2f} rad", worst <= HEAD_TOLERANCE_RAD, worst


def report(trace, timeline: Timeline, termination) -> tuple[list[str], bool, float]:
    """A line per keyframe then the fall line, whether everything passed, and the largest head error."""
    starts = {start: (end, sit) for start, end, sit in _flag_segments(timeline)}
    ends = list(timeline.times[1:]) + [timeline.duration_s]
    lines, passed, head_error = [], True, 0.0
    for t, command, end in zip(timeline.times, timeline.commands, ends):
        parts = []
        if t in starts:
            text, ok = _posture(trace, t, *starts[t])
            parts.append(text)
            passed &= ok
        text, ok, worst = _head(trace, command, t, end)
        parts.append(text)
        passed &= ok
        head_error = max(head_error, worst)
        lines.append(f"t={t:5.2f}  " + "; ".join(parts))
    fell = stage.fell_at(trace, termination)
    lines.append("no fall" if fell is None else f"fell at {fell:.2f} s")
    return lines, passed and fell is None, head_error


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check a MicroDuck Got a Talent policy's performance on the Arena's stage.")
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
    lines, passed, head_error = report(result.trace[0], timeline, event.termination)
    passed = passed and result.seeds_finished == 1
    print("\n".join(lines))
    print(f"{'PASS' if passed else 'FAIL'}: largest head error {head_error:.2f} rad")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
