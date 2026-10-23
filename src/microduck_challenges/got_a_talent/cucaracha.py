"""Write "La Cucaracha", an 8-duck troupe timeline for MicroDuck Got a Talent.

    python3 src/microduck_challenges/got_a_talent/cucaracha.py [out.json]

The story is the song's own line, "la cucaracha ya no puede caminar" (the cockroach can't walk
anymore): a sprinter dashes across the front and wipes out; two walkers stroll in from the back and
stop dead on "ya no puede caminar"; a V of five sings, beats its heads in mirror symmetry, sits in
sympathy, sends Mexican waves across the V, and bows on the last chord.

Stage frame: x towards the audience, y to the audience's right, yaw 0 faces the audience.
head = [neck_pitch (+ beak up, neck back), head_pitch (+ nod down), head_yaw (+ looks to +y),
head_roll (+ tilts to +y)]. A duck at -y mirrors its partner at +y by negating yaw and roll.
"""

import json
import math
import sys
from pathlib import Path

SPRINT_REF = "pollen-robotics/microduck-sprint-2m-reference@62afcc461f1d4a5ac22bfbff55ce4900c1afba32"
RACE9 = "JacoCronje/microduck-sprint-race9@b4db96fde420503f1fca922485ec2a68d4f5104a"
DURATION = 20.0
# Where the sprinter starts (x, y, yaw) and where it lies after its wipe-out, read off the simulation.
CRASHER_START = (1.0, 1.6, -60)
CRASH_AT = (1.6, -0.3)
# The chorus's start heading: talent1 checkpoint 2750 turns about 14° to its right while it settles,
# so it starts turned the other way to face the audience.
CHORUS_YAW = 10

# The beat: 3/4, a quarter 0.4 s, a bar 1.2 s; bar downbeats at 3.6 + 1.2 k.
Q = 0.4
E = 0.2


def r(t):
    return round(t + 1e-9, 2)


class Duck:
    def __init__(self, policy, x, y, yaw, voice=None):
        self.policy, self.start, self.voice = policy, {"x": x, "y": y, "yaw": yaw}, voice
        self.y = y
        self.flag, self.head, self.walk, self.trig = {0.0: 0}, {0.0: (0, 0, 0, 0)}, {0.0: 0.0}, {0.0: (0, 0)}
        self.song = []

    # --- moves -------------------------------------------------------------------------------
    def sit(self, t):
        self.flag[r(t)] = 1

    def stand(self, t):
        self.flag[r(t)] = 0

    def look(self, t, neck=0.0, pitch=0.0, yaw=0.0, roll=0.0):
        self.head[r(t)] = (neck, pitch, yaw, roll)

    def go(self, t, vx):
        self.walk[r(t)] = vx

    def quack(self, t, length=0.16):
        self.trig[r(t)] = (1, 0)
        self.trig[r(t + length)] = (0, 0)

    def wheee(self, t, length):
        self.trig[r(t)] = (0, 1)
        self.trig[r(t + length)] = (0, 0)

    def sing(self, t, note, length, dynamic="mf", vowel="ah"):
        n = {"t": r(t), "note": note, "for": r(length)}
        if dynamic != "mf":
            n["dynamic"] = dynamic
        if vowel != "ah":
            n["vowel"] = vowel
        self.song.append(n)

    # --- the file ------------------------------------------------------------------------------
    def to_json(self, legs_walker=False):
        times = sorted(set(self.flag) | set(self.head) | set(self.walk) | set(self.trig))
        last = lambda track, t: track[max(k for k in track if k <= t)]
        keyframes = []
        for t in times:
            kf = {"t": t}
            vx = last(self.walk, t) if legs_walker else last(self.flag, t)
            if vx:
                kf["twist"] = [vx, 0, 0]
            if not legs_walker and any(last(self.head, t)):
                kf["head"] = [round(v, 3) for v in last(self.head, t)]
            if any(last(self.trig, t)):
                kf["triggers"] = list(last(self.trig, t))
            keyframes.append(kf)
        # Drop keyframes that repeat the one before (a release over a release, ...).
        kept = [keyframes[0]]
        for kf in keyframes[1:]:
            if {k: v for k, v in kf.items() if k != "t"} != {k: v for k, v in kept[-1].items() if k != "t"}:
                kept.append(kf)
        out = {"policy": self.policy, "start": self.start, "keyframes": kept}
        if self.voice:
            out["voice"] = self.voice
        if self.song:
            self.song.sort(key=lambda n: n["t"])
            out["song"] = self.song
        return out


def aim(duck, x, y):
    """The head yaw that points a duck facing the audience at the stage point (x, y)."""
    dx, dy = x - duck.start["x"], y - duck.start["y"]
    return max(-1.3, min(1.3, math.atan2(dy, dx)))


def build():
    # ---------------------------------------------------------------- the cast
    star = Duck("self", 0.55, 0.0, CHORUS_YAW, {"brightness": 0.6, "vibrato": 0.45, "pitch": 0.6})
    in_l = Duck("self", 0.20, -0.50, CHORUS_YAW, {"pitch": 0.45, "vibrato": 0.2})
    in_r = Duck("self", 0.20, +0.50, CHORUS_YAW, {"pitch": 0.5, "vibrato": 0.2})
    out_l = Duck("self", -0.15, -0.95, CHORUS_YAW, {"pitch": 0.15, "buzz": 0.7, "brightness": 0.2})
    out_r = Duck("self", -0.15, +0.95, CHORUS_YAW, {"pitch": 0.3, "buzz": 0.6})
    walk_l = Duck(SPRINT_REF, -0.95, -1.45, 10, {"pitch": 0.8})
    walk_r = Duck(SPRINT_REF, -0.95, +1.50, 10, {"pitch": 0.75})
    crasher = Duck(RACE9, *CRASHER_START, {"pitch": 0.95, "vibrato": 0.9})
    chorus = [out_l, in_l, star, in_r, out_r]  # audience's left to right: the wave's order
    pairs = [(in_l, in_r), (out_l, out_r)]
    crash_at = CRASH_AT  # where the sprinter ends up (checked in simulation)

    # ---------------------------------------------------------------- 0.0-2.4 the dash and the crash
    # Heads follow the sprinter as it passes in front of them, then flinch at the wipe-out.
    for t, y in ((0.0, 1.5), (0.5, 1.0), (0.9, 0.3), (1.2, -0.3)):
        for d in chorus:
            d.look(t, yaw=aim(d, 0.95, y))
    for d in chorus:
        d.look(1.6, neck=0.3, pitch=-0.2, yaw=aim(d, *crash_at))  # recoil, beak up: "¡ay!"
        d.look(2.2, pitch=0.35, yaw=aim(d, *crash_at))           # and peer down at it
    star.quack(1.55)
    for d in (in_l, in_r):
        d.quack(1.65)
    crasher.wheee(1.0, 0.5)  # the yell on the way down
    crasher.sing(1.5, "E5", 0.25, "f", "ee")
    crasher.sing(1.75, "C5", 0.25, "mf", "oh")
    crasher.sing(2.0, "A4", 0.5, "p", "oo")

    # ---------------------------------------------------------------- music
    def oompah(bar, root, fifth, low, high, bass=out_l, pah_low=in_l, pah_high=in_r):
        t = 3.6 + 1.2 * bar
        bass.sing(t, root if bar % 2 == 0 else fifth, 0.35, "f", "oh")
        for b in (1, 2):
            pah_low.sing(t + b * Q, low, 0.2, "mp", "mm")
            pah_high.sing(t + b * Q, high, 0.2, "mp", "mm")

    # One vamp bar before the tune (bar -1 = 2.4 s), then the chords under the six lines.
    F, C7 = ("F2", "C3", "A3", "C4"), ("C3", "G2", "G3", "Bb3")
    chords = [F] * 4 + [C7] * 2 + [C7] * 4 + [C7, F]  # bars 0-11 (3.6-18.0)
    for bar, chord in enumerate([F] + chords, start=-1):
        oompah(bar, *chord)

    def line(t0, notes, duck=star, dynamic="f"):
        t = t0
        for note, length in notes:
            if note:
                duck.sing(t, note, length - 0.02, dynamic, "ee" if length < 0.3 else "ah")
            t += length
        return t

    cucaracha = [("C5", E), ("C5", E), ("C5", E), ("F5", Q), ("A5", 2 * Q)]
    porque = [("C5", E), ("C5", E), ("C5", E), ("E5", Q), ("G5", 2 * Q)]
    line(3.0, cucaracha)
    line(5.4, cucaracha)
    line(8.0, [("F5", E), ("F5", E), ("E5", E), ("E5", E), ("D5", E), ("D5", E), ("C5", 0.9)])
    line(10.2, porque)
    line(12.6, porque)
    line(14.4, [("C6", E), ("D6", E), ("C6", E), ("Bb5", E), ("A5", E), ("G5", E), ("F5", 0.6)])
    # The finale: the whole V on the first line, in harmony, then the last chord.
    line(16.2, cucaracha, star, "ff")
    line(16.2, [("A4", E), ("A4", E), ("A4", E), ("C5", Q), ("F5", 2 * Q)], out_r, "f")
    for d, note in ((star, "F5"), (out_r, "C5"), (in_l, "A4"), (in_r, "C4"), (out_l, "F2")):
        d.sing(18.6, note, 1.3, "ff", "oh")
    # Walkers and the crasher answer the tune: "¡ay, ay!".
    for d, t in ((walk_l, 4.8), (walk_r, 7.2)):
        d.quack(t)
        d.quack(t + 0.2)

    # ---------------------------------------------------------------- 2.4-8.0 line 1 & 2: walk in, head beats
    for w in (walk_l, walk_r):
        w.go(3.6, 0.3)
        w.go(8.0, 0.0)  # "YA no puede caminar": they stop dead
        w.quack(8.0, 0.3)
    # Vamp bar: heads come back to the front one pair at a time, centre outwards.
    star.look(2.4)
    for i, (a, b) in enumerate(pairs):
        a.look(2.6 + 0.2 * i)
        b.look(2.6 + 0.2 * i)
    # Line 1 (3.6-6.0): mirror beats on the quarter, the wings opening and closing like a fan;
    # the star nods on every downbeat.
    for k in range(6):
        t = 3.6 + k * Q
        out = 0.55 if k % 2 == 0 else -0.25
        for a, b in pairs:
            a.look(t, yaw=-out)
            b.look(t, yaw=+out)
        star.look(t, pitch=0.35 if k % 3 == 0 else -0.15)
    # Line 2 (6.0-8.0): no mirror now: the whole V sways together, left, right, the star leading
    # a beat early, the outer pair a beat late (a ripple).
    for k in range(5):
        t = 6.0 + k * Q
        side = 0.6 if k % 2 == 0 else -0.6
        star.look(t - 0.2, yaw=side, roll=0.2 * math.copysign(1, side))
        for a, b in pairs[:1]:
            a.look(t, yaw=side, roll=0.2 * math.copysign(1, side))
            b.look(t, yaw=side, roll=0.2 * math.copysign(1, side))
        for a, b in pairs[1:]:
            a.look(t + 0.2, yaw=side, roll=0.2 * math.copysign(1, side))
            b.look(t + 0.2, yaw=side, roll=0.2 * math.copysign(1, side))

    # ---------------------------------------------------------------- 8.0-10.2 line 3: "ya no puede caminar"
    # Everyone looks at the fallen sprinter on "ya", sits down in sympathy, and hangs its head on "NAR".
    for d in chorus:
        d.look(8.0, yaw=aim(d, *crash_at))
        d.sit(8.4)
        d.look(9.2, pitch=0.45, yaw=0.6 * aim(d, *crash_at))
    star.look(9.6, pitch=0.45, yaw=-0.3)  # the star shakes its head: no, no
    star.look(9.8, pitch=0.45, yaw=0.3)
    star.look(10.0, pitch=0.45, yaw=-0.3)

    # ---------------------------------------------------------------- 10.2-14.4 lines 4 & 5: Mexican waves
    # Every sit turns this policy about 20° to its left (and a rise about 10° back), so the show sits
    # twice only. Line 4: the wave rises out of the sympathy sit, left to right, each duck springing
    # up with its beak to the sky and a quack; the left walker starts it, the right one finishes it.
    # Line 5: the wave runs back right to left standing, beaks only.
    def wave(t0, order, rise):
        for i, d in enumerate(order):
            t = t0 + 0.25 * i
            if rise:
                d.stand(t)
            d.look(t, neck=0.5, pitch=-0.35)
            d.quack(t + 0.05)
            d.look(t + 0.6, pitch=0.1)
    for d in chorus:
        d.look(10.2, pitch=0.1)
    walk_l.quack(10.55)
    wave(10.8, chorus, rise=True)
    walk_r.quack(12.05)
    # Between the waves the V nods on the downbeats, the wings in counter-phase to the centre.
    for t, s in ((12.4, 1), (12.8, -1)):
        star.look(t, pitch=0.3 * s)
        for a, b in pairs:
            a.look(t, pitch=-0.3 * s)
            b.look(t, pitch=-0.3 * s)
    walk_r.quack(12.95)
    wave(13.2, chorus[::-1], rise=False)
    walk_l.quack(14.45)

    # ---------------------------------------------------------------- 14.4-16.2 line 6: the run down
    # The V follows the descending run with its beaks, note by note, mirrored on the wings.
    for k in range(6):
        t = 14.6 + k * E
        p = -0.4 + 0.15 * k
        star.look(t, pitch=p)
        for a, b in pairs:
            yw = 0.5 if k % 2 == 0 else 0.1
            a.look(t, pitch=p, yaw=-yw)
            b.look(t, pitch=p, yaw=+yw)
    # "fu-MAR": beaks to the sky together on the downbeat.
    for d in chorus:
        d.look(15.6, neck=0.45, pitch=-0.3)
        d.quack(15.62)

    # ---------------------------------------------------------------- 16.2-20 finale
    # Walkers stroll down to the front line for the curtain call.
    for w in (walk_l, walk_r):
        w.go(16.2, 0.3)
        w.go(18.6, 0.0)
    # Double-time mirrored head beats while the tutti sings, outer pair in counter-phase.
    for k in range(12):
        t = 16.2 + k * E
        s = 1 if k % 2 == 0 else -1
        star.look(t, pitch=0.25 * s)
        in_l.look(t, yaw=-0.45 * s, roll=-0.15 * s)
        in_r.look(t, yaw=+0.45 * s, roll=+0.15 * s)
        out_l.look(t, yaw=+0.45 * s, roll=+0.15 * s)
        out_r.look(t, yaw=-0.45 * s, roll=-0.15 * s)
    # The bow: the last chord, then a ripple of curtsies from the centre outwards, and every duck
    # turns to the sprinter for the last "¡Olé!".
    star.sit(18.6)
    star.look(18.6, pitch=0.45)
    for i, (a, b) in enumerate(pairs, start=1):
        for d in (a, b):
            d.sit(18.6 + 0.2 * i)
            d.look(18.6 + 0.2 * i, pitch=0.45)
    for d in chorus:
        d.look(19.4, neck=0.5, pitch=-0.4, yaw=0.5 * aim(d, *crash_at))
    for d in chorus + [walk_l, walk_r]:
        d.quack(19.5, 0.3)
    crasher.sing(19.4, "F5", 0.5, "ff", "eh")  # the sprinter gets the last word from the floor

    ducks = [star, in_l, in_r, out_l, out_r, walk_l, walk_r, crasher]
    walkers = {id(walk_l), id(walk_r), id(crasher)}
    return {"timeline_version": 2, "duration_s": DURATION,
            "ducks": [d.to_json(legs_walker=id(d) in walkers) for d in ducks]}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("cucaracha.json"))
    with open(out, "w") as f:
        json.dump(build(), f, indent=1)
    print("wrote", out)
