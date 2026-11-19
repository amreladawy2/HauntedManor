#!/usr/bin/env python3
"""Synthesizes the Haunted Manor sound effects into assets/sounds/*.wav.

Pure standard library (wave, math, random, struct). Output: 16-bit PCM, mono, 22050 Hz,
peak-normalized, short fades on one-shots, seamless crossfaded/wrapped loops.
Run from anywhere:  py -3.10 tools/gen_sounds.py
"""
import math
import os
import random
import struct
import wave

SR = 22050
TAU = 2.0 * math.pi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "sounds")
rng = random.Random(1887)  # fixed seed: reproducible assets


# ----------------------------------------------------------------------------- building blocks
def n_of(sec):
    return int(round(sec * SR))


def zeros(sec):
    return [0.0] * n_of(sec)


def noise(n):
    return [rng.uniform(-1.0, 1.0) for _ in range(n)]


def add(dst, src, start=0.0, gain=1.0, wrap=False):
    """Mix src into dst at 'start' seconds. wrap=True folds overflow to the start (for loops)."""
    i0 = n_of(start)
    n = len(dst)
    for i, v in enumerate(src):
        j = i0 + i
        if j >= n:
            if not wrap:
                break
            j %= n
        dst[j] += v * gain
    return dst


def mul(x, fn):
    """Multiply each sample by fn(t)."""
    return [v * fn(i / SR) for i, v in enumerate(x)]


def scale(x, g):
    return [v * g for v in x]


def onepole_lp(x, fc):
    a = math.exp(-TAU * fc / SR)
    y = 0.0
    out = []
    for v in x:
        y = (1.0 - a) * v + a * y
        out.append(y)
    return out


def onepole_hp(x, fc):
    return [a - b for a, b in zip(x, onepole_lp(x, fc))]


def biquad(x, kind, fc, q=0.707):
    """RBJ biquad: 'lp', 'hp' or 'bp' (constant 0 dB peak)."""
    w = TAU * fc / SR
    cw, sw = math.cos(w), math.sin(w)
    al = sw / (2.0 * q)
    if kind == "lp":
        b0, b1, b2 = (1 - cw) / 2, 1 - cw, (1 - cw) / 2
    elif kind == "hp":
        b0, b1, b2 = (1 + cw) / 2, -(1 + cw), (1 + cw) / 2
    else:
        b0, b1, b2 = al, 0.0, -al
    a0, a1, a2 = 1 + al, -2 * cw, 1 - al
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    x1 = x2 = y1 = y2 = 0.0
    out = []
    for v in x:
        y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, v, y1, y
        out.append(y)
    return out


def svf(x, fcs, q, mode="bp"):
    """Chamberlin state-variable filter with a per-sample cutoff list (time-varying sweeps)."""
    low = band = 0.0
    damp = 1.0 / q
    out = []
    lim = SR / 6.5  # keep the Chamberlin SVF stable
    for v, fc in zip(x, fcs):
        f = 2.0 * math.sin(math.pi * min(fc, lim) / SR)
        low += f * band
        high = v - low - damp * band
        band += f * high
        out.append(band if mode == "bp" else low)
    return out


def tone(dur, freq_fn, amp_fn, harmonics=((1, 1.0),), phase=0.0):
    """Oscillator with time-varying frequency/amplitude and a fixed harmonic recipe."""
    n = n_of(dur)
    out = [0.0] * n
    ph = phase
    for i in range(n):
        t = i / SR
        ph += TAU * freq_fn(t) / SR
        a = amp_fn(t)
        if a < 1e-5:
            continue
        s = 0.0
        for k, g in harmonics:
            s += g * math.sin(k * ph)
        out[i] = a * s
    return out


def modes(dur, partials, attack=0.0008):
    """Sum of exponentially damped sines: partials = (freq, amp, decay_seconds)."""
    n = n_of(dur)
    out = [0.0] * n
    for f, a, d in partials:
        w = TAU * f / SR
        for i in range(n):
            t = i / SR
            e = math.exp(-t / d)
            if e < 1e-4:
                break
            out[i] += a * e * math.sin(w * i)
    na = max(1, n_of(attack))
    for i in range(min(na, n)):
        out[i] *= i / na
    return out


def burst(dur, decay, kind="bp", fc=2000.0, q=0.8):
    """Filtered noise burst with an exponential envelope."""
    x = noise(n_of(dur))
    x = biquad(x, kind, fc, q)
    return mul(x, lambda t: math.exp(-t / decay))


def smooth_rand(n, rate, period_n=None):
    """Cosine-interpolated random control signal in 0..1; periodic over period_n samples if given."""
    period_n = period_n or n
    pts = max(2, int(period_n / SR * rate))
    ctrl = [rng.random() for _ in range(pts)]
    out = []
    for i in range(n):
        p = (i % period_n) / period_n * pts
        k = int(p)
        fr = p - k
        a, b = ctrl[k % pts], ctrl[(k + 1) % pts]
        w = (1 - math.cos(math.pi * fr)) * 0.5
        out.append(a + (b - a) * w)
    return out


def reverb(x, mix=0.3, room=0.84, damp=0.35, tail=1.0):
    """Small Schroeder reverb (4 damped combs + 2 allpasses). Returns len(x)+tail samples."""
    x = x + [0.0] * n_of(tail)
    k = SR / 44100.0
    wet = [0.0] * len(x)
    for d in (1116, 1188, 1277, 1356):
        d = int(d * k)
        buf = [0.0] * d
        idx = 0
        lp = 0.0
        for i, v in enumerate(x):
            y = buf[idx]
            lp = y * (1 - damp) + lp * damp
            buf[idx] = v + lp * room
            idx = (idx + 1) % d
            wet[i] += y
    for d in (556, 441):
        d = int(d * k)
        buf = [0.0] * d
        idx = 0
        for i, v in enumerate(wet):
            b = buf[idx]
            buf[idx] = v + b * 0.5
            wet[i] = b - v
            idx = (idx + 1) % d
    return [a * (1 - mix) + b * mix * 0.25 for a, b in zip(x, wet)]


def make_loop(x, n, xf):
    """x has n+xf samples; equal-power crossfade the tail into the head -> seamless n-sample loop."""
    out = x[:n]
    for i in range(xf):
        t = i / xf
        out[i] = x[i] * math.sin(t * math.pi / 2) + x[n + i] * math.cos(t * math.pi / 2)
    return out


def fold(x, n):
    """Wrap everything past n back onto the start (reverb tails of loop content)."""
    out = x[:n] + [0.0] * max(0, n - len(x))
    for i in range(n, len(x)):
        out[i % n] += x[i]
    return out


def fades(x, fin=0.003, fout=0.03):
    n = len(x)
    a, b = max(1, n_of(fin)), max(1, n_of(fout))
    for i in range(min(a, n)):
        x[i] *= i / a
    for i in range(min(b, n)):
        x[n - 1 - i] *= i / b
    return x


def write(name, x, peak, loop=False):
    if not loop:
        x = fades(list(x))
    m = max(abs(v) for v in x) or 1.0
    g = peak / m
    data = struct.pack("<%dh" % len(x), *[int(round(max(-1.0, min(1.0, v * g)) * 32767)) for v in x])
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name + ".wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data)
    print("%-16s %5.2fs %7d bytes" % (name, len(x) / SR, os.path.getsize(path)))
    return path


def thump(dur, f_hi, f_lo, drop, decay, harmonics=((1, 1.0),)):
    """Low sine with a fast downward pitch glide: the body of footsteps, thuds and heartbeats.
    Extra harmonics keep sub-100 Hz bodies audible on laptop speakers."""
    return tone(dur, lambda t: f_lo + (f_hi - f_lo) * math.exp(-t / drop), lambda t: math.exp(-t / decay), harmonics)


def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


# ----------------------------------------------------------------------------- sounds
def footstep():
    x = zeros(0.25)
    heel = thump(0.25, 140, 70, 0.015, 0.035)
    board = modes(0.25, [(182, 0.35, 0.05), (311, 0.25, 0.035), (523, 0.15, 0.02), (871, 0.08, 0.012)])
    scuff = burst(0.05, 0.010, "bp", 700, 0.9)
    add(x, heel, 0.004)
    add(x, board, 0.004)
    add(x, scuff, 0.0, 0.5)
    add(x, heel, 0.065, 0.35)  # toe settles after the heel
    add(x, board, 0.065, 0.3)
    return onepole_lp(x, 3000)


def knock():
    k = modes(0.3, [(212, 1.0, 0.045), (397, 0.6, 0.03), (723, 0.35, 0.02), (1251, 0.15, 0.01)])
    return add(k, burst(0.02, 0.003, "bp", 1600, 0.7), 0.0, 0.4)


def creak_sound(dur=1.2, base=38.0, sweep=26.0, seed_gain=1.0):
    """Stick-slip friction: an irregular impulse train ringing resonant wood bandpasses."""
    n = n_of(dur)
    jit = smooth_rand(n, 9)
    env_ctrl = smooth_rand(n, 6)
    exc = [0.0] * n
    ph = 0.0
    for i in range(n):
        t = i / SR
        rate = base + sweep * math.sin(math.pi * t / dur) + 14 * (jit[i] - 0.5)
        ph += rate / SR
        if ph >= 1.0:
            ph -= 1.0
            env = min(1.0, t / 0.12) * min(1.0, (dur - t) / 0.3)
            exc[i] = env * (0.55 + 0.45 * env_ctrl[i]) * (0.7 + 0.3 * rng.random()) * seed_gain
    out = [0.0] * n
    for fc, q, g in ((340, 6, 0.6), (523, 12, 1.0), (1150, 10, 0.7), (1910, 8, 0.4), (2780, 9, 0.2)):
        add(out, biquad(exc, "bp", fc, q), 0.0, g)
    fr = mul(biquad(noise(n), "bp", 2500, 1.2), lambda t: 0.04 * min(1.0, t / 0.12) * max(0.0, min(1.0, (dur - t) / 0.3)))
    return add(out, fr)


def interact():
    x = zeros(0.5)
    add(x, knock(), 0.0, 1.0)
    add(x, knock(), 0.16, 0.7)
    add(x, creak_sound(0.28, 55, 20), 0.21, 0.35)
    return x


def bump():
    x = zeros(0.3)
    add(x, thump(0.3, 120, 45, 0.02, 0.07, ((1, 1.0), (2, 0.3), (3, 0.1))))
    add(x, onepole_lp(burst(0.15, 0.04, "lp", 400, 0.7), 300), 0.0, 0.6)
    add(x, modes(0.3, [(143, 0.25, 0.06), (262, 0.15, 0.04)]), 0.003)
    return onepole_lp(x, 900)


def click():
    x = modes(0.08, [(2100, 1.0, 0.006), (3400, 0.5, 0.004), (1100, 0.4, 0.01), (880, 0.25, 0.014)])
    return add(x, burst(0.01, 0.0015, "hp", 3000, 0.7), 0.0, 0.5)


def switch():
    d = 1.2
    n = n_of(d)
    fcs = [220 + 1900 * math.sin(math.pi * (i / n)) ** 2 for i in range(n)]
    whoosh = svf(noise(n), fcs, 4.0)
    env = lambda t: (t / 0.5) ** 2 if t < 0.5 else math.exp(-(t - 0.5) / 0.22)
    x = mul(whoosh, env)
    ghost = tone(d, lambda t: (392 - 110 * t) * (1 + 0.012 * math.sin(TAU * 5.5 * t)),
                 lambda t: 0.22 * math.sin(math.pi * t / d) ** 2, ((1, 1.0), (2, 0.15)))
    ghost2 = tone(d, lambda t: (415 - 118 * t) * (1 + 0.012 * math.sin(TAU * 4.7 * t)),
                  lambda t: 0.18 * math.sin(math.pi * t / d) ** 2)
    sub = tone(d, lambda t: 55, lambda t: 0.3 * math.sin(math.pi * t / d) ** 3)
    for s in (ghost, ghost2, sub):
        add(x, s)
    return reverb(x, 0.35, 0.82, tail=0.0)


def horror_on():
    d = 2.0
    saw = tuple((k, 1.0 / k) for k in range(1, 7))
    env = lambda t: min(1.0, t / 0.006) * math.exp(-t / 0.75)
    x = zeros(d)
    for f in (73.4, 77.8, 103.8, 155.6):  # D2, D#2, G#2 (tritone), D#3: grinding cluster
        add(x, tone(d, lambda t, f=f: f * (1 - 0.03 * t / d), env, saw), 0.0, 0.18)
    for f in (1480, 1568, 1661):  # shrieking string cluster
        add(x, tone(d, lambda t, f=f: f * (1 + 0.008 * math.sin(TAU * 6.3 * t + f)),
                    lambda t: min(1.0, t / 0.03) * math.exp(-t / 0.5)), 0.0, 0.07)
    add(x, thump(d, 85, 38, 0.06, 0.6), 0.0, 0.9)
    add(x, burst(0.4, 0.08, "lp", 2000, 0.7), 0.0, 0.6)
    y = reverb(x, 0.4, 0.86, tail=0.0)
    return y


def heartbeat():
    n = n_of(1.6)
    x = [0.0] * n

    def beat(f0):
        b = thump(0.3, f0 * 1.7, f0, 0.025, 0.055, ((1, 1.0), (2, 0.4), (3, 0.15)))
        b = [v * min(1.0, i / n_of(0.008)) for i, v in enumerate(b)]
        return add(b, onepole_lp(burst(0.1, 0.02, "lp", 150, 0.7), 120), 0.0, 0.5)

    add(x, beat(56), 0.04, 1.0, wrap=True)    # lub
    add(x, beat(47), 0.33, 0.75, wrap=True)   # dub
    return onepole_lp(x, 260)


def whisper():
    d = 2.0
    n = n_of(d)
    vowels = [(300, 870, 2240), (530, 1840, 2480), (730, 1090, 2440), (270, 2290, 3010), (570, 840, 2410)]
    syl = []
    t = 0.08
    while t < d - 0.2:
        ln = rng.uniform(0.12, 0.22)
        syl.append((t, ln, rng.choice(vowels), rng.uniform(0.5, 1.0)))
        t += ln + rng.uniform(0.02, 0.09)
    amp = [0.0] * n
    f1 = [500.0] * n
    f2 = [1500.0] * n
    f3 = [2500.0] * n
    for (s, ln, v, g) in syl:
        for i in range(n_of(s), min(n, n_of(s + ln))):
            p = (i / SR - s) / ln
            amp[i] = max(amp[i], g * max(0.0, math.sin(math.pi * p)) ** 1.5)
            f1[i], f2[i], f3[i] = v
    # glide formants between syllables so the breath does not jump
    f1, f2, f3 = (onepole_lp(f, 25) for f in (f1, f2, f3))
    src = onepole_hp(noise(n), 300)
    x = [0.0] * n
    for fcs, q, g in ((f1, 6, 1.0), (f2, 8, 0.7), (f3, 9, 0.35)):
        add(x, svf(src, fcs, q), 0.0, g)
    breath = mul(biquad(noise(n), "bp", 1200, 0.5), lambda t: 0.06)
    x = [(a + b) * (0.15 + e) for a, b, e in zip(x, breath, onepole_lp(amp, 30))]
    for s in (0.05, syl[len(syl) // 2][0] - 0.05, d - 0.45):  # sibilant "ss"/"sh" hisses
        hiss = mul(biquad(noise(n_of(0.16)), "hp", 3800, 0.7), lambda t: math.sin(math.pi * t / 0.16) ** 2)
        add(x, hiss, max(0.0, s), 0.35)
    y = reverb(x, 0.45, 0.85, tail=0.0)
    return mul(y, lambda t: 0.6 + 0.4 * math.sin(math.pi * t / d))


def ambient_library():
    L = 12.0
    n, xf = n_of(L), n_of(1.0)
    m = n + xf
    x = [0.0] * m
    lfo = lambda t, p: math.sin(TAU * t / p)
    # drone: every frequency has an integer number of cycles in 12 s -> periodic
    for f, a, p in ((55.0, 0.5, 12), (55.25, 0.35, 6), (82.5, 0.2, 4), (110.0, 0.1, 3), (77.75, 0.07, 12)):
        add(x, tone(L + 1.0, lambda t, f=f: f, lambda t, a=a, p=p: a * (0.75 + 0.25 * lfo(t, p)),
                    ((1, 1.0), (2, 0.25), (3, 0.08))))
    # fire: low roar + hiss with a slowly breathing intensity
    inten = smooth_rand(m, 1.5, n)
    roar = onepole_lp(onepole_lp(noise(m), 220), 220)
    hiss = biquad(noise(m), "bp", 2600, 0.7)
    for i in range(m):
        x[i] += roar[i] * (1.2 + 1.6 * inten[i]) + hiss[i] * 0.025 * (0.4 + inten[i])
    x = make_loop(x, n, xf)
    # crackles and pops, wrapped around the loop seam
    t = 0.0
    while t < L:
        t += rng.expovariate(7.0)
        c = burst(0.04, rng.uniform(0.002, 0.008), "bp", rng.uniform(1500, 5200), 1.5)
        add(x, c, t % L, rng.uniform(0.05, 0.25), wrap=True)
    t = 0.0
    while t < L:
        t += rng.expovariate(0.6)
        pop = add(burst(0.08, 0.015, "bp", rng.uniform(500, 900), 1.0), thump(0.08, 160, 90, 0.01, 0.02), 0.0, 0.5)
        add(x, pop, t % L, rng.uniform(0.2, 0.4), wrap=True)
    return x


def tine(dur, f, gain=1.0, decay=1.2):
    """Music-box comb tine: strong fundamental, weak overtones, short inharmonic ping."""
    x = modes(dur, [(f, 1.0, decay), (2 * f, 0.15, decay * 0.4), (3 * f, 0.08, decay * 0.2), (6.27 * f, 0.06, 0.04)])
    return scale(x, gain)


def ambient_bedroom():
    L = 12.0
    n, xf = n_of(L), n_of(1.0)
    m = n + xf
    gust = smooth_rand(m, 0.4, n)
    sweep = smooth_rand(m, 0.5, n)
    fcs = [320 + 520 * sweep[i] + 150 * math.sin(TAU * i / n) for i in range(m)]
    wind = svf(noise(m), fcs, 3.0)
    howl = svf(noise(m), [f * 2.1 + 200 for f in fcs], 18.0)
    rumble = onepole_lp(onepole_lp(noise(m), 300), 300)
    x = [wind[i] * (0.4 + 0.6 * gust[i] ** 1.5) + howl[i] * 0.5 * gust[i] ** 2.5 + rumble[i] * 0.9 for i in range(m)]
    x = make_loop(x, n, xf)
    # faint detuned music-box drone: beating pairs, all periodic in 12 s
    pad = [0.0] * n
    for f, a, p in ((440.0, 0.5, 12), (440.5, 0.5, 6), (523.25, 0.4, 12), (523.5833, 0.35, 4),
                    (659.25, 0.3, 6), (659.0, 0.3, 12)):
        add(pad, tone(L, lambda t, f=f: f, lambda t, a=a, p=p: a * (0.6 + 0.4 * math.sin(TAU * t / p)),
                      ((1, 1.0), (2, 0.12))))
    # occasional lonely plinks, reverb tails folded over the loop seam
    pl = [0.0] * n
    for t, mm in ((1.7, 88), (4.9, 84), (7.6, 81), (10.3, 83)):
        add(pl, tine(2.0, midi(mm) * 2 ** (-30 / 1200.0), 1.0, 0.9), t, wrap=True)
    pl = fold(reverb(pl, 0.5, 0.88, tail=2.0), n)
    for i in range(n):
        x[i] += pad[i] * 0.012 + pl[i] * 0.06
    return x


# Church-bell partial ratios (hum, prime, tierce, quint, nominal, ...): (ratio, amp, decay_seconds).
BELL_PARTIALS = [(0.5, 0.35, 2.8), (1.0, 0.6, 2.2), (1.2, 0.5, 1.8), (1.5, 0.3, 1.4), (2.0, 1.0, 1.6),
                 (2.5, 0.25, 0.9), (2.67, 0.2, 0.8), (3.0, 0.25, 0.7), (4.0, 0.15, 0.5), (5.33, 0.08, 0.3)]


def bell(dur, f0, decay=1.0, attack=0.0015):
    """Inharmonic bell strike on prime f0; 'decay' scales every partial's ring time."""
    p = []
    for r, a, dc in BELL_PARTIALS:
        p.append((f0 * r, a, dc * decay))
        p.append((f0 * r + 0.35 * r, a * 0.6, dc * 0.9 * decay))  # near-duplicate partial -> slow warble
    return modes(dur, p, attack)


def chime():
    d = 3.0
    x = bell(d, 196.0)  # G3 prime; gong-rod clock strike
    add(x, burst(0.03, 0.004, "bp", 3200, 0.9), 0.0, 0.5)
    y = reverb(x, 0.25, 0.85, tail=0.0)
    return mul(y, lambda t: 1.0 if t < d - 0.5 else (d - t) / 0.5)


def music_box():
    d = 4.0
    x = zeros(d)
    melody = [76, 81, 84, 83, 81, 76, 77, 76]  # E5 A5 C6 B5 A5 E5 F5 E5: A-minor lullaby
    t, step = 0.05, 0.28
    for k, mm in enumerate(melody):
        cents = -35 + rng.uniform(-25, 25) - 4 * k  # flat, wandering, drooping as the spring unwinds
        f = midi(mm) * 2 ** (cents / 1200.0)
        g = 0.9 - 0.04 * k
        add(x, tine(d - t, f, g), t)
        add(x, tine(d - t, f * 2 ** (12 / 1200.0), g * 0.5), t)  # detuned twin -> eerie beating
        add(x, burst(0.01, 0.001, "hp", 4000, 0.7), t, 0.08)    # mechanism tick
        t += step
        step *= 1.07  # winding down
    y = reverb(x, 0.35, 0.86, tail=0.0)
    return mul(y, lambda tt: 1.0 if tt < d - 0.6 else (d - tt) / 0.6)


def light_on():
    d = 0.8
    n = n_of(d)
    x = [0.0] * n
    # scratch along the box
    sc = biquad(noise(n_of(0.13)), "hp", 2200, 0.7)
    rough = smooth_rand(len(sc), 220)
    sc = [v * rough[i] ** 2 * min(1.0, i / n_of(0.02)) for i, v in enumerate(sc)]
    add(x, sc, 0.0, 0.9)
    # ignition pop
    add(x, burst(0.05, 0.008, "bp", 1200, 0.8), 0.12, 1.0)
    add(x, thump(0.06, 200, 110, 0.01, 0.02), 0.12, 0.5)
    # flare: lowpassed noise whose cutoff jumps then settles; slow decay into a faint hiss
    fl_n = n - n_of(0.12)
    fcs = [300 + 2700 * min(1.0, (i / SR) / 0.08) * math.exp(-(i / SR) / 0.3) + 900 for i in range(fl_n)]
    flare = svf(noise(fl_n), fcs, 0.8, "lp")
    flare = mul(flare, lambda t: min(1.0, t / 0.04) * (0.25 + 0.75 * math.exp(-t / 0.2)))
    add(x, flare, 0.12, 0.7)
    for _ in range(6):
        add(x, burst(0.02, 0.002, "bp", rng.uniform(2000, 5000), 1.5), rng.uniform(0.15, 0.6), rng.uniform(0.1, 0.3))
    return x


def creak():
    return creak_sound(1.2)


def flashlight():
    def clk(f):
        c = modes(0.03, [(f, 1.0, 0.004), (f * 1.64, 0.5, 0.002), (f * 0.46, 0.6, 0.006)])
        return add(c, burst(0.006, 0.0015, "hp", 3500, 0.7), 0.0, 0.5)
    x = zeros(0.15)
    add(x, clk(2800), 0.004, 1.0)
    add(x, clk(3300), 0.07, 0.55)  # release half of the slide switch
    add(x, modes(0.08, [(5200, 0.06, 0.03)]), 0.004)  # spring ting
    return x


# ----------------------------------------------------------------------------- quest extension (v2)
def drive(x, k):
    """Normalize to 1, then tanh-saturate: louder, harsher, never above full scale."""
    m = max(abs(v) for v in x) or 1.0
    return [math.tanh(k * v / m) for v in x]


def scream():
    """Piercing, distorted shriek: chorused band-limited saw voices with a scream pitch contour,
    'ah' formants, 70 Hz roughness (what makes real screams alarming) and tanh distortion."""
    d = 1.3
    n = n_of(d)
    jit = smooth_rand(n, 25)
    src = [0.0] * n
    for v, (det, gain) in enumerate(((1.0, 1.0), (1.034, 0.6), (0.968, 0.5), (1.5, 0.25))):
        ph = rng.random() * TAU
        for i in range(n):
            t = i / SR
            if t < 0.12:
                c = 650 + 650 * (t / 0.12) ** 0.6           # rising onset
            elif t < 0.95:
                c = 1300 + 90 * math.sin(TAU * 1.3 * t)     # quivering plateau
            else:
                c = 1300 - 500 * ((t - 0.95) / 0.35) ** 1.5  # voice cracks downward
            f = c * det * (1 + 0.03 * math.sin(TAU * 8.5 * t + v) + 0.04 * (jit[i] - 0.5))
            ph += TAU * f / SR
            s = 0.0
            for k in range(1, min(int(SR / 2 / f), 10) + 1):  # band-limited saw
                s += math.sin(k * ph) / k
            src[i] += s * gain
    x = [0.0] * n
    for fc, q, g in ((1000, 5, 1.0), (1700, 6, 0.8), (3100, 6, 0.5)):
        add(x, svf(src, [fc] * n, q), 0.0, g)
    add(x, src, 0.0, 0.12)  # raw buzz for bite
    add(x, biquad(noise(n), "hp", 2500, 0.7), 0.0, 0.15)  # breath
    env = lambda t: min(1.0, t / 0.02) * (1.0 if t < 1.0 else max(0.0, 1 - (t - 1.0) / 0.3))
    x = [v * env(i / SR) * (1 - 0.35 * (0.5 + 0.5 * math.sin(TAU * 70 * i / SR))) for i, v in enumerate(x)]
    return drive(x, 2.5)


def scare_hit():
    """Loud dissonant orchestral stab (brass/string cluster, closing filter) on a heavy impact and crash."""
    d = 0.9
    n = n_of(d)
    saw = tuple((k, 1.0 / k) for k in range(1, 9))
    env = lambda t: (min(1.0, t / 0.004) * (0.35 + 0.65 * math.exp(-t / 0.12))
                     * (1.0 if t < 0.55 else max(0.0, 1 - (t - 0.55) / 0.35)))
    stab = [0.0] * n
    for f in (65.4, 69.3, 92.5, 138.6, 196.0, 277.2, 370.0, 415.3):  # C/C#/F# clusters over octaves
        add(stab, tone(d, lambda t, f=f: f * (1 - 0.02 * t), env, saw), 0.0, 0.2)
    stab = svf(stab, [900 + 6000 * math.exp(-(i / SR) / 0.15) for i in range(n)], 1.2, "lp")  # brass "blat"
    for f in (1975.5, 2093.0, 2217.5):  # screeching violins
        add(stab, tone(d, lambda t, f=f: f * (1 + 0.01 * math.sin(TAU * 7 * t + f)),
                       lambda t: min(1.0, t / 0.01) * math.exp(-t / 0.4)), 0.0, 0.1)
    x = stab
    add(x, thump(d, 140, 38, 0.03, 0.18, ((1, 1.0), (2, 0.5), (3, 0.25))), 0.0, 0.6)
    add(x, burst(0.6, 0.15, "hp", 2500, 0.7), 0.0, 0.45)  # cymbal-like crash
    add(x, burst(0.2, 0.03, "lp", 1500, 0.7), 0.0, 0.6)   # body of the hit
    x = drive(x, 2.0)
    return reverb(x, 0.2, 0.8, tail=0.0)


def win():
    """Eerie music-box phrase that resolves to a warm A-major chord with a bell."""
    d = 3.5
    x = zeros(d)
    for t, m, c in ((0.05, 76, -30), (0.30, 84, -22), (0.55, 83, -14), (0.80, 80, -6)):  # E5 C6 B5 G#5, tuning in
        f = midi(m) * 2 ** (c / 1200.0)
        add(x, tine(d - t, f, 0.8, 1.0), t)
        add(x, tine(d - t, f * 2 ** (10 / 1200.0), 0.35, 1.0), t)
    t0 = 1.1
    for k, m in enumerate((81, 85, 88)):  # A5 C#6 E6, gently rolled
        add(x, tine(d - t0, midi(m), 0.6, 1.8), t0 + 0.03 * k)
    pad_env = lambda t: min(1.0, t / 0.35) * math.exp(-t / 1.4)
    for f in (220.0, 277.18, 329.63, 440.0):  # warm A-major pad, chorused pairs
        for df in (-0.6, 0.6):
            add(x, tone(d - t0, lambda t, f=f + df: f, pad_env, ((1, 1.0), (2, 0.3), (3, 0.1))), t0, 0.09)
    add(x, bell(d - t0, 440.0, 0.8), t0, 0.35)
    y = reverb(x, 0.35, 0.86, tail=0.0)
    return mul(y, lambda t: 1.0 if t < d - 0.6 else (d - t) / 0.6)


def riddle():
    """Low whisper swell with a dark drone, crowned by a small chime."""
    d = 1.5
    n = n_of(d)
    src = onepole_hp(noise(n), 200)
    f1 = [380 + 120 * math.sin(TAU * 1.1 * i / SR) for i in range(n)]
    f2 = [1100 + 350 * math.sin(TAU * 0.7 * i / SR + 1.0) for i in range(n)]
    x = svf(src, f1, 5)
    add(x, svf(src, f2, 7), 0.0, 0.6)
    flutter = smooth_rand(n, 9)
    swell = lambda t: math.sin(math.pi * min(1.0, t / d)) ** 2
    x = [v * swell(i / SR) * (0.6 + 0.4 * flutter[i]) for i, v in enumerate(x)]
    add(x, tone(d, lambda t: 73.4, lambda t: 0.25 * swell(t), ((1, 1.0), (2, 0.3))))
    add(x, bell(d - 0.45, 392.0, 0.6), 0.45, 0.35)
    y = reverb(x, 0.45, 0.85, tail=0.0)
    return mul(y, lambda t: 1.0 if t < d - 0.3 else (d - t) / 0.3)


def pickup():
    """Paper rustle followed by a small metallic jingle."""
    d = 0.6
    x = zeros(d)
    rn = n_of(0.38)
    rough = smooth_rand(rn, 60)
    ragged = [rough[i] ** 3 * math.sin(math.pi * i / rn) ** 0.5 for i in range(rn)]
    crinkle = biquad(noise(rn), "bp", 3200, 0.6)
    body = biquad(noise(rn), "bp", 900, 0.8)
    add(x, [(c + 0.3 * b) * e for c, b, e in zip(crinkle, body, ragged)])
    for _ in range(25):  # crisp paper ticks
        add(x, burst(0.01, 0.0015, "hp", 4000, 0.7), rng.uniform(0.0, 0.36), rng.uniform(0.1, 0.4))
    for t, f, g in ((0.30, 2600, 0.22), (0.37, 3100, 0.16), (0.43, 2850, 0.12)):
        add(x, modes(0.25, [(f, 1.0, 0.12), (f * 1.47, 0.6, 0.08), (f * 2.09, 0.4, 0.05), (f * 2.76, 0.25, 0.03)]), t, g)
    return x


def objective():
    """Soft, low chime: a quiet objective cue."""
    d = 0.9
    x = bell(d, 130.81, 0.45, attack=0.006)  # C3 prime, nominal C4
    add(x, modes(d, [(392.0, 0.25, 0.5)], attack=0.006))  # faint G4 shimmer
    x = onepole_lp(x, 2500)
    y = reverb(x, 0.3, 0.8, tail=0.0)
    return mul(y, lambda t: 1.0 if t < d - 0.3 else (d - t) / 0.3)


SOUNDS = [
    ("footstep", footstep, 0.55, False),
    ("interact", interact, 0.8, False),
    ("bump", bump, 0.8, False),
    ("click", click, 0.5, False),
    ("switch", switch, 0.75, False),
    ("horror_on", horror_on, 0.95, False),
    ("heartbeat", heartbeat, 0.9, True),
    ("whisper", whisper, 0.7, False),
    ("ambient_library", ambient_library, 0.6, True),
    ("ambient_bedroom", ambient_bedroom, 0.6, True),
    ("chime", chime, 0.85, False),
    ("music_box", music_box, 0.75, False),
    ("light_on", light_on, 0.75, False),
    ("creak", creak, 0.75, False),
    ("flashlight", flashlight, 0.6, False),
    # v2 quest sounds: appended last so the shared seeded RNG leaves the sounds above byte-identical
    ("scream", scream, 0.97, False),
    ("scare_hit", scare_hit, 0.97, False),
    ("win", win, 0.8, False),
    ("riddle", riddle, 0.75, False),
    ("pickup", pickup, 0.7, False),
    ("objective", objective, 0.7, False),
]

if __name__ == "__main__":
    total = 0
    for name, fn, peak, loop in SOUNDS:
        total += os.path.getsize(write(name, fn(), peak, loop))
    print("total %.2f MB" % (total / 1e6))
