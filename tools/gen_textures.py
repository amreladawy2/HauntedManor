#!/usr/bin/env python3
"""Procedural texture generator for Haunted Manor.

Writes the 35 texture ids listed in docs/CONTRACT.md to assets/textures/<id>.png
(512x512 RGB, sRGB). Pure Pillow, no numpy. Every random source is a seeded
random.Random, so the output is deterministic.

Tiling materials (floors, walls, wood, fabric, stone, metal, ...) are built from
periodic noise and wrap-around drawing, so they tile seamlessly. Pictures that are
mapped once (portrait, clock face, rugs, night sky, mirror, paper, world map) are
laid out as single images.

Usage:  py -3.10 tools/gen_textures.py            (all textures)
        py -3.10 tools/gen_textures.py wood_floor  (only the named ids)
"""
import math
import os
import random
import sys
import time

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

S = 512                      # texture size; the pixel loops below rely on S == 512 (x = i & 511, y = i >> 9)
N = S * S
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'assets', 'textures')
TAU = 2.0 * math.pi


# =============================================================================
# Basic helpers
# =============================================================================
def px(im):
    """Flat pixel list (Pillow 12 renamed getdata to get_flattened_data)."""
    f = getattr(im, 'get_flattened_data', None)
    return list(f() if f else im.getdata())


def clamp(v, lo=0.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


def sstep(a, b, x):
    t = (x - a) / (b - a)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def normalize(f):
    lo, hi = min(f), max(f)
    k = 1.0 / (hi - lo) if hi > lo else 0.0
    return [(v - lo) * k for v in f]


def uniform(f):
    """Rank-equalise a field to a uniform 0..1 distribution (threshold t covers 1-t of the area)."""
    order = sorted(range(len(f)), key=f.__getitem__)
    out = [0.0] * len(f)
    k = 1.0 / (len(f) - 1)
    for rank, i in enumerate(order):
        out[i] = rank * k
    return out


def tile3(im):
    w, h = im.size
    big = Image.new(im.mode, (w * 3, h * 3))
    for i in range(3):
        for j in range(3):
            big.paste(im, (i * w, j * h))
    return big


def noise(cx, cy=None, seed=0):
    """Periodic value noise in ~[0,1]: a cx x cy random lattice bicubic-upsampled to S x S.
    The lattice is tiled 3x3 before resampling and the centre cropped, so it wraps exactly."""
    cy = cy or cx
    rnd = random.Random(seed)
    small = Image.new('F', (cx, cy))
    small.putdata([rnd.random() for _ in range(cx * cy)])
    if cx == S and cy == S:
        return px(small)
    up = tile3(small).resize((S * 3, S * 3), Image.BICUBIC)
    return px(up.crop((S, S, 2 * S, 2 * S)))


def fbm(cx, cy=None, octaves=4, seed=0, gain=0.5):
    """Fractal sum of periodic value noise, normalised to 0..1."""
    cy = cy or cx
    acc = [0.0] * N
    amp = 1.0
    for o in range(octaves):
        n = noise(min(cx << o, S), min(cy << o, S), seed * 7919 + o * 104729 + 13)
        acc = [a + amp * v for a, v in zip(acc, n)]
        amp *= gain
    return normalize(acc)


def field_img(f, scale=255.0):
    im = Image.new('F', (S, S))
    im.putdata(f, scale, 0.5)
    return im.convert('L')


def img_field(im):
    return [v * (1.0 / 255.0) for v in px(im.convert('L'))]


def blur_wrap(im, r):
    """Gaussian blur that wraps around the tile edges."""
    w, h = im.size
    return tile3(im).filter(ImageFilter.GaussianBlur(r)).crop((w, h, 2 * w, 2 * h))


def blur_field(f, r):
    return img_field(blur_wrap(field_img(f), r))


def wrap9(fn, size=S):
    """Call fn(dx, dy) for the 9 periodic copies so drawn shapes wrap around the edges."""
    for dy in (-size, 0, size):
        for dx in (-size, 0, size):
            fn(dx, dy)


def gradient_lut(stops, n=1024):
    stops = sorted(stops, key=lambda s: s[0])
    luts = ([], [], [])
    for i in range(n):
        t = i / (n - 1)
        k = 0
        while k < len(stops) - 2 and t > stops[k + 1][0]:
            k += 1
        (p0, c0), (p1, c1) = stops[k], stops[k + 1]
        u = 0.0 if p1 == p0 else clamp((t - p0) / (p1 - p0))
        for ch in range(3):
            luts[ch].append(c0[ch] + (c1[ch] - c0[ch]) * u)
    return luts


def partition(total, lo, hi, rnd):
    """Random integer widths in [lo, hi] (last one may be slightly short) summing to total."""
    out, rem = [], total
    while rem > hi:
        w = rnd.randint(lo, hi)
        if rem - w < lo:
            w = rem - lo
        out.append(w)
        rem -= w
    out.append(rem)
    return out


def ellipse_bbox(cx, cy, rx, ry=None):
    ry = rx if ry is None else ry
    return [cx - rx, cy - ry, cx + rx, cy + ry]


def ellipse_poly(cx, cy, rx, ry, ang=0.0, n=36):
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for k in range(n):
        t = TAU * k / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        pts.append((cx + x * ca - y * sa, cy + x * sa + y * ca))
    return pts


def bezier(p0, p1, p2, p3, n=40):
    pts = []
    for k in range(n + 1):
        t = k / n
        u = 1.0 - t
        a, b, c, d = u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t
        pts.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return pts


def leaf_poly(path, width, power=0.8):
    """Closed polygon around a centre line with a pointed (sin-shaped) width profile."""
    n = len(path)
    left, right = [], []
    for k, (x, y) in enumerate(path):
        a, b = path[max(k - 1, 0)], path[min(k + 1, n - 1)]
        tx, ty = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / ln, tx / ln
        w = width * math.sin(math.pi * k / (n - 1)) ** power
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    return left + right[::-1]


def spiral(cx, cy, r0, r1, turns, a0=0.0, direction=1, n=60):
    pts = []
    for k in range(n + 1):
        t = k / n
        a = a0 + direction * TAU * turns * t
        r = r0 + (r1 - r0) * t
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def sc(pts, z, dx=0.0, dy=0.0):
    return [(x * z + dx, y * z + dy) for x, y in pts]


def speckles(seed, count, rmin, rmax, blur=0.0, value=255):
    """Mask of small random wrapped blobs (dust, specks, pits)."""
    rnd = random.Random(seed)
    m = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(m)
    for _ in range(count):
        x, y = rnd.uniform(0, S), rnd.uniform(0, S)
        rx, ry = rnd.uniform(rmin, rmax), rnd.uniform(rmin, rmax)
        v = int(value * rnd.uniform(0.5, 1.0))
        wrap9(lambda dx, dy: d.ellipse(ellipse_bbox(x + dx, y + dy, rx, ry), fill=v))
    if blur:
        m = blur_wrap(m, blur)
    return img_field(m)


def scratches(seed, count, lmin, lmax, width=1, bend=0.25, value=255, angle=None):
    """Mask of thin, slightly curved wrapped strokes."""
    rnd = random.Random(seed)
    m = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(m)
    for _ in range(count):
        x, y = rnd.uniform(0, S), rnd.uniform(0, S)
        a = rnd.uniform(0, TAU) if angle is None else angle + rnd.uniform(-0.35, 0.35)
        ln = rnd.uniform(lmin, lmax)
        steps = max(2, int(ln / 4))
        pts = [(x, y)]
        for _s in range(steps):
            a += rnd.uniform(-bend, bend)
            x += math.cos(a) * ln / steps
            y += math.sin(a) * ln / steps
            pts.append((x, y))
        v = int(value * rnd.uniform(0.45, 1.0))
        wrap9(lambda dx, dy: d.line([(p[0] + dx, p[1] + dy) for p in pts], fill=v, width=width))
    return img_field(m)


def cracks(seed, count, steps, step_len=2.5, jitter=0.45, branch=0.04, width=1):
    """Mask of branching random-walk cracks (plaster, enamel, paint)."""
    rnd = random.Random(seed)
    m = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(m)
    stack = [(rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(0, TAU), steps) for _ in range(count)]
    while stack:
        x, y, a, n = stack.pop()
        pts = [(x, y)]
        for _ in range(n):
            a += rnd.uniform(-jitter, jitter)
            x += math.cos(a) * step_len
            y += math.sin(a) * step_len
            pts.append((x, y))
            if rnd.random() < branch and n > 20:
                stack.append((x, y, a + rnd.choice((-1, 1)) * rnd.uniform(0.6, 1.3), int(n * rnd.uniform(0.25, 0.55))))
        v = rnd.randint(170, 255)
        wrap9(lambda dx, dy: d.line([(p[0] + dx, p[1] + dy) for p in pts], fill=v, width=width))
    return img_field(m)


def voronoi(cx, cy, seed, jitter=0.9):
    """Periodic Worley noise on a cx x cy cell grid. Returns (F1, F2, cell id) in pixels."""
    rnd = random.Random(seed)
    lo = (1.0 - jitter) * 0.5
    pts = [(lo + rnd.random() * jitter, lo + rnd.random() * jitter) for _ in range(cx * cy)]
    sx, sy = S / cx, S / cy
    f1 = [0.0] * N
    f2 = [0.0] * N
    ids = [0] * N
    i = 0
    for y in range(S):
        gy = (y + 0.5) / sy
        iy = int(gy)
        rows = []
        for dj in (-1, 0, 1):
            cj = iy + dj
            rows.append((cj, (cj % cy) * cx))
        for x in range(S):
            gx = (x + 0.5) / sx
            ix = int(gx)
            d1 = d2 = 1e18
            best = 0
            for cj, base in rows:
                for di in (-1, 0, 1):
                    ci = ix + di
                    k = base + ci % cx
                    ox, oy = pts[k]
                    ddx = (ci + ox - gx) * sx
                    ddy = (cj + oy - gy) * sy
                    dd = ddx * ddx + ddy * ddy
                    if dd < d1:
                        d2 = d1
                        d1 = dd
                        best = k
                    elif dd < d2:
                        d2 = dd
            f1[i] = math.sqrt(d1)
            f2[i] = math.sqrt(d2)
            ids[i] = best
            i += 1
    return f1, f2, ids


def masonry(rows, seed, warp, rc):
    """Seamless block layout. rows: [(height, [widths...], x offset)], heights and each
    row's widths sum to S. Returns (block id, distance to the block edge in px) per pixel."""
    row_of, rstart, rh = [0] * S, [], []
    y = 0
    for r, (h, _w, _o) in enumerate(rows):
        rstart.append(y)
        rh.append(h)
        for yy in range(y, y + h):
            row_of[yy] = r
        y += h
    assert y == S, 'row heights must sum to S'
    look, gid = [], 0
    for h, widths, off in rows:
        assert sum(widths) == S, 'row widths must sum to S'
        lut, x = [None] * S, 0
        for w in widths:
            ent = (gid, x, x + w)
            for xx in range(x, x + w):
                lut[xx] = ent
            x += w
            gid += 1
        look.append(lut)
    wx = fbm(10, 10, 3, seed)
    wy = fbm(10, 10, 3, seed + 1)
    offs = [r[2] for r in rows]
    bid = [0] * N
    dist = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        yw = (y + (wy[i] - 0.5) * warp) % S
        yi = int(yw)
        if yi >= S:
            yi, yw = 0, 0.0
        r = row_of[yi]
        ly = yw - rstart[r]
        h = rh[r]
        xx = (x + (wx[i] - 0.5) * warp - offs[r]) % S
        xi = int(xx)
        if xi >= S:
            xi, xx = 0, 0.0
        g, x0, x1 = look[r][xi]
        dx = min(xx - x0, x1 - xx)
        dy = min(ly, h - ly)
        if dx < rc and dy < rc:
            d = rc - math.hypot(rc - dx, rc - dy)
        else:
            d = dx if dx < dy else dy
        bid[i] = g
        dist[i] = d
    return bid, dist


# =============================================================================
# Float RGB canvas
# =============================================================================
class Tex:
    """Three float channel lists (0..255 sRGB) with blend helpers."""

    def __init__(self, r, g, b):
        self.r, self.g, self.b = r, g, b

    @classmethod
    def solid(cls, c):
        return cls([float(c[0])] * N, [float(c[1])] * N, [float(c[2])] * N)

    @classmethod
    def from_image(cls, im):
        r, g, b = im.convert('RGB').split()
        return cls([float(v) for v in px(r)], [float(v) for v in px(g)], [float(v) for v in px(b)])

    @classmethod
    def grad(cls, f, stops):
        lr, lg, lb = gradient_lut(stops)
        idx = [0 if v <= 0.0 else 1023 if v >= 1.0 else int(v * 1023.0) for v in f]
        return cls([lr[i] for i in idx], [lg[i] for i in idx], [lb[i] for i in idx])

    def copy(self):
        return Tex(self.r[:], self.g[:], self.b[:])

    def image(self):
        return Image.merge('RGB', [field_img(ch, 1.0) for ch in (self.r, self.g, self.b)])

    def lerp(self, c, m, k=1.0):
        """Blend towards colour/Tex c by mask m (list 0..1) scaled by k."""
        if isinstance(c, Tex):
            self.r = [a + (o - a) * t * k for a, o, t in zip(self.r, c.r, m)]
            self.g = [a + (o - a) * t * k for a, o, t in zip(self.g, c.g, m)]
            self.b = [a + (o - a) * t * k for a, o, t in zip(self.b, c.b, m)]
        else:
            cr, cg, cb = c
            self.r = [a + (cr - a) * t * k for a, t in zip(self.r, m)]
            self.g = [a + (cg - a) * t * k for a, t in zip(self.g, m)]
            self.b = [a + (cb - a) * t * k for a, t in zip(self.b, m)]
        return self

    def shade(self, f, lo, hi):
        """Multiply by lo + (hi - lo) * f."""
        d = hi - lo
        m = [lo + d * v for v in f]
        self.r = [a * k for a, k in zip(self.r, m)]
        self.g = [a * k for a, k in zip(self.g, m)]
        self.b = [a * k for a, k in zip(self.b, m)]
        return self

    def mul(self, m):
        self.r = [a * k for a, k in zip(self.r, m)]
        self.g = [a * k for a, k in zip(self.g, m)]
        self.b = [a * k for a, k in zip(self.b, m)]
        return self

    def tint(self, c):
        self.r = [a * c[0] for a in self.r]
        self.g = [a * c[1] for a in self.g]
        self.b = [a * c[2] for a in self.b]
        return self

    def add(self, c, m):
        self.r = [a + c[0] * t for a, t in zip(self.r, m)]
        self.g = [a + c[1] * t for a, t in zip(self.g, m)]
        self.b = [a + c[2] * t for a, t in zip(self.b, m)]
        return self

    def desat(self, m, k=1.0):
        """Pull towards grey by mask m * k."""
        lum = [0.3 * r + 0.59 * g + 0.11 * b for r, g, b in zip(self.r, self.g, self.b)]
        self.r = [a + (l - a) * t * k for a, l, t in zip(self.r, lum, m)]
        self.g = [a + (l - a) * t * k for a, l, t in zip(self.g, lum, m)]
        self.b = [a + (l - a) * t * k for a, l, t in zip(self.b, lum, m)]
        return self


def stain(tex, seed, coverage, tint, ring_col, cells=3, k=0.35, ring_k=0.45, rings=2):
    """Water/tea stains: blotches with darker tide-line rings (periodic, so seamless)."""
    f = uniform(fbm(cells, cells, 5, seed))
    t0 = 1.0 - coverage
    body = [sstep(t0, t0 + 0.03, v) for v in f]
    tex.lerp(tint, body, k)
    for j in range(rings):
        tj = t0 + 0.004 + j * 0.035
        ring = [max(0.0, 1.0 - abs(v - tj) * 260.0) for v in f]
        tex.lerp(ring_col, ring, ring_k * (1.0 - 0.35 * j))
    return tex


def grime(tex, seed, lo=0.8, hi=1.04, cells=4):
    return tex.shade(fbm(cells, cells, 5, seed), lo, hi)


# =============================================================================
# Wood
# =============================================================================
def wood_field(seed, K, knots=0, ring_pow=2.6, warp=2.4):
    """Long grain along V: wavy ring lines (periodic in x via integer K) plus fibres."""
    low = fbm(3, 2, 3, seed)
    mid = fbm(14, 3, 3, seed + 1)
    fib = noise(S, 26, seed + 2)
    fib2 = noise(110, 7, seed + 3)
    blot = fbm(4, 3, 4, seed + 4)
    rnd = random.Random(seed)
    kn = [(rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(9, 15)) for _ in range(knots)]
    v = [0.0] * N
    kk = K / S
    for i in range(N):
        x = i & 511
        y = i >> 9
        g = x * kk + warp * low[i] + 0.7 * mid[i]
        dark = 0.0
        for kx, ky, kr in kn:
            dx = x - kx
            dx -= S * round(dx / S)
            dy = y - ky
            dy -= S * round(dy / S)
            q = (dx * dx + dy * dy * 0.3) / (kr * kr)
            e = math.exp(-q)
            g += 2.6 * e
            dark += math.exp(-q * 5.0)
        ring = g - math.floor(g)
        v[i] = 0.55 * ring ** ring_pow + 0.17 * fib[i] + 0.13 * fib2[i] + 0.22 * blot[i] + 0.6 * dark
    return normalize(v)


def wood_tex(seed, K, stops, knots=0):
    v = wood_field(seed, K, knots)
    t = Tex.grad(v, stops)
    pores = [sstep(0.80, 0.97, p) for p in noise(256, 36, seed + 7)]
    t.lerp((stops[-1][1][0] * 0.7, stops[-1][1][1] * 0.7, stops[-1][1][2] * 0.7), pores, 0.45)
    return t, v


def gen_wood_dark():
    t, _ = wood_tex(201, 38, [(0.0, (156, 104, 66)), (0.45, (118, 74, 45)), (1.0, (66, 38, 22))], knots=1)
    t.lerp((150, 120, 95), scratches(202, 40, 8, 50, 1), 0.25)
    t.lerp((40, 24, 15), scratches(203, 25, 10, 40, 1), 0.35)
    stain(t, 204, 0.10, (92, 62, 40), (55, 35, 22), k=0.25, ring_k=0.3)
    return grime(t, 205, 0.82, 1.05).image()


def gen_wood_light():
    t, _ = wood_tex(211, 30, [(0.0, (198, 160, 112)), (0.45, (168, 126, 82)), (1.0, (114, 78, 46))], knots=2)
    t.lerp((205, 185, 160), scratches(212, 35, 8, 50, 1), 0.25)
    t.lerp((80, 55, 35), scratches(213, 25, 10, 40, 1), 0.3)
    stain(t, 214, 0.10, (150, 118, 80), (100, 70, 42), k=0.25, ring_k=0.35)
    return grime(t, 215, 0.8, 1.04).image()


def gen_wood_floor():
    """Dark oak boards along V, 4 per tile (~0.5 m each when the tile spans 2 m)."""
    rnd = random.Random(101)
    K = 44
    lows = [fbm(3, 2, 3, 1011), fbm(3, 2, 3, 1012)]
    mids = [fbm(14, 3, 3, 1013), fbm(14, 3, 3, 1014)]
    fib = noise(S, 28, 1015)
    fib2 = noise(96, 8, 1016)
    planks = []
    for _k in range(4):
        j1 = rnd.randrange(S)
        j2 = (j1 + rnd.randrange(170, 340)) % S
        planks.append((j1, j2, (rnd.uniform(0.84, 1.1), rnd.uniform(0.84, 1.1)),
                       (rnd.uniform(-1, 1), rnd.uniform(-1, 1))))
    val = [0.0] * N
    tone = [1.0] * N
    warm = [0.0] * N
    seam = [1.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        k = x >> 7
        lx = x & 127
        j1, j2, tones, warmth = planks[k]
        if j1 < j2:
            seg = 1 if j1 <= y < j2 else 0
        else:
            seg = 1 if (y >= j1 or y < j2) else 0
        sid = k * 2 + seg
        f = (k + seg) & 1
        j = (y << 9) | ((x + sid * 149) & 511)
        g = x * K / S + 2.4 * lows[f][j] + 0.7 * mids[f][j]
        ring = g - math.floor(g)
        val[i] = 0.55 * ring * ring * ring + 0.22 * fib[j] + 0.18 * fib2[j]
        tone[i] = tones[seg]
        warm[i] = warmth[seg]
        s = 1.0
        if lx == 0:
            s = 0.25
        elif lx == 1:
            s = 0.6
        elif lx == 127:
            s = 0.78
        elif lx == 2:
            s = 0.9
        if y == j1 or y == j2:
            s = min(s, 0.28)
        elif y == ((j1 + 1) & 511) or y == ((j2 + 1) & 511):
            s = min(s, 0.66)
        elif y == ((j1 - 1) & 511) or y == ((j2 - 1) & 511):
            s = min(s, 0.82)
        seam[i] = s
    t = Tex.grad(normalize(val), [(0.0, (160, 114, 72)), (0.45, (124, 85, 52)), (1.0, (70, 45, 27))])
    t.r = [a * tn * (1 + 0.05 * w) for a, tn, w in zip(t.r, tone, warm)]
    t.g = [a * tn for a, tn in zip(t.g, tone)]
    t.b = [a * tn * (1 - 0.08 * w) for a, tn, w in zip(t.b, tone, warm)]
    pores = [sstep(0.82, 0.97, p) for p in noise(256, 40, 1017)]
    t.lerp((48, 30, 18), pores, 0.4)
    # worn varnish paths and scuffs
    wear = uniform(fbm(3, 3, 4, 1018))
    t.desat([sstep(0.55, 0.95, w) for w in wear], 0.25)
    t.shade([sstep(0.5, 1.0, w) for w in wear], 1.0, 1.12)
    t.lerp((176, 146, 112), scratches(1019, 70, 10, 60, 1, 0.15), 0.3)
    t.lerp((172, 150, 124), scratches(1020, 14, 25, 70, 4, 0.1), 0.18)
    t.lerp((35, 22, 14), scratches(1021, 30, 10, 45, 1, 0.2), 0.35)
    # nail holes beside every end joint
    m = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(m)
    for k, (j1, j2, _t, _w) in enumerate(planks):
        for jy in (j1, j2):
            for nx in (k * 128 + 22, k * 128 + 106):
                for oy in (-7, 8):
                    wrap9(lambda dx, dy: d.ellipse(ellipse_bbox(nx + dx, jy + oy + dy, 1.6), fill=255))
    t.lerp((25, 16, 10), img_field(m), 0.85)
    t.mul(seam)
    stain(t, 1022, 0.07, (90, 66, 44), (50, 32, 20), cells=2, k=0.3, ring_k=0.35)
    t.lerp((110, 100, 88), speckles(1023, 90, 0.6, 1.6), 0.4)
    return grime(t, 1024, 0.74, 1.06, 5).image()


def gen_wood_painted_white():
    wood, grain = wood_tex(221, 34, [(0.0, (176, 136, 92)), (0.5, (140, 100, 64)), (1.0, (92, 62, 38))])
    paint = Tex.solid((212, 206, 188))
    brush = noise(160, 6, 222)
    paint.shade(brush, 0.93, 1.03)
    paint.shade(grain, 1.03, 0.9)                          # grain relief through the paint
    paint.lerp((196, 182, 150), fbm(3, 3, 4, 223), 0.55)   # yellowing
    chipn = uniform([0.7 * a + 0.3 * b for a, b in zip(fbm(9, 9, 5, 224), fbm(3, 3, 3, 225))])
    chip = [sstep(0.86, 0.875, c) for c in chipn]
    edge = [max(0.0, 1.0 - abs(c - 0.845) * 55.0) for c in chipn]
    lip = [max(0.0, 1.0 - abs(c - 0.862) * 120.0) for c in chipn]
    paint.lerp((120, 110, 92), edge, 0.55)
    paint.lerp((232, 228, 214), lip, 0.5)
    paint.lerp(wood, chip)
    paint.lerp((70, 62, 50), scratches(226, 30, 6, 30, 1), 0.3)
    paint.lerp((120, 110, 90), speckles(227, 160, 0.5, 1.4), 0.4)
    stain(paint, 228, 0.08, (180, 160, 120), (130, 112, 80), k=0.3, ring_k=0.35)
    return grime(paint, 229, 0.78, 1.02).image()


# =============================================================================
# Walls / ceiling
# =============================================================================
def damask_cell_masks(z):
    """Positive and cut-out masks of one 256x256 damask cell at scale z (mirror-symmetric)."""
    W = 256 * z
    pos = Image.new('L', (W, W), 0)
    cut = Image.new('L', (W, W), 0)
    dp, dc = ImageDraw.Draw(pos), ImageDraw.Draw(cut)

    def P(pts):
        return sc(pts, z)

    # central almond
    alm, alm_in = [], []
    for k in range(48):
        t = TAU * k / 48
        s = math.sin(t)
        sx = math.copysign(abs(s) ** 1.5, s)
        alm.append((128 + 22 * sx, 126 - 54 * math.cos(t)))
        alm_in.append((128 + 10 * sx, 126 - 36 * math.cos(t)))
    dp.polygon(P(alm), fill=255)
    dc.polygon(P(alm_in), fill=255)
    dp.ellipse(ellipse_bbox(128 * z, 126 * z, 5 * z), fill=255)
    for yy in (100, 152):
        dp.ellipse(ellipse_bbox(128 * z, yy * z, 3 * z), fill=255)
    # top finial: bud + flame tip
    dp.ellipse(ellipse_bbox(128 * z, 58 * z, 11 * z, 10 * z), fill=255)
    dp.polygon(P([(119, 54), (128, 22), (137, 54)]), fill=255)
    dc.ellipse(ellipse_bbox(128 * z, 58 * z, 4 * z), fill=255)
    # bottom palmette fan
    dp.pieslice(ellipse_bbox(128 * z, 214 * z, 34 * z, 26 * z), 180, 360, fill=255)
    dp.polygon(P([(124, 176), (132, 176), (131, 214), (125, 214)]), fill=255)
    for k in range(5):
        a = math.pi + math.pi * (k + 1) / 6
        dc.line(P([(128, 213), (128 + 30 * math.cos(a), 213 + 23 * math.sin(a))]), fill=255, width=int(2.2 * z))
    dp.ellipse(ellipse_bbox(128 * z, 220 * z, 9 * z, 6 * z), fill=255)
    # left half ornaments (mirrored later)
    leaf_a = bezier((116, 102), (92, 50), (58, 56), (50, 96))
    leaf_b = bezier((116, 152), (88, 196), (54, 192), (44, 158))
    for path, w in ((leaf_a, 13), (leaf_b, 12)):
        dp.polygon(P(leaf_poly(path, w)), fill=255)
        dc.line(P(path[6:-6]), fill=255, width=int(1.8 * z))
    for (cx, cy, r0, dirn, a0) in ((52, 100, 13, -1, -1.2), (45, 154, 12, 1, 1.0)):
        dp.line(P(spiral(cx, cy, r0, 3, 1.4, a0, dirn)), fill=255, width=int(4 * z), joint='curve')
    dp.line(P(bezier((110, 127), (96, 118), (84, 138), (70, 127))), fill=255, width=int(3 * z), joint='curve')
    for bx, by, br in ((64, 127, 5), (74, 115, 3.5), (74, 139, 3.5), (88, 108, 2.5), (88, 146, 2.5)):
        dp.ellipse(ellipse_bbox(bx * z, by * z, br * z), fill=255)
    # small upper leaves beside the finial
    dp.polygon(P(leaf_poly(bezier((124, 66), (110, 60), (100, 44), (104, 30)), 6)), fill=255)
    # ogee lattice lines linking to the corner rosettes
    dp.line(P(bezier((128, 20), (96, 4), (36, 34), (14, 12))), fill=255, width=int(3 * z), joint='curve')
    dp.line(P(bezier((128, 238), (96, 254), (36, 224), (14, 244))), fill=255, width=int(3 * z), joint='curve')
    # corner rosettes (wrapped)

    def rosette(dx, dy):
        for k in range(4):
            a = k * math.pi / 2
            tip = (dx + 24 * math.cos(a), dy + 24 * math.sin(a))
            path = [(dx + 4 * math.cos(a), dy + 4 * math.sin(a)), tip]
            pts = bezier(path[0], (dx + 12 * math.cos(a) - 6 * math.sin(a), dy + 12 * math.sin(a) + 6 * math.cos(a)),
                         (dx + 18 * math.cos(a) + 5 * math.sin(a), dy + 18 * math.sin(a) - 5 * math.cos(a)), tip, 20)
            dp.polygon(P(leaf_poly(pts, 6.5)), fill=255)
            b = a + math.pi / 4
            dp.ellipse(ellipse_bbox((dx + 17 * math.cos(b)) * z, (dy + 17 * math.sin(b)) * z, 2.8 * z), fill=255)
        dp.ellipse(ellipse_bbox(dx * z, dy * z, 7 * z), fill=255)
        dc.ellipse(ellipse_bbox(dx * z, dy * z, 3.2 * z), fill=255)

    for cx in (0, 256):
        for cy in (0, 256):
            rosette(cx, cy)
    pos = ImageChops.lighter(pos, ImageOps.mirror(pos))
    cut = ImageChops.lighter(cut, ImageOps.mirror(cut))
    return ImageChops.subtract(pos, cut)


def gen_wallpaper_damask():
    z = 2
    cell = damask_cell_masks(z)
    big = Image.new('L', (S * z, S * z))
    for i in range(2):
        for j in range(2):
            big.paste(cell, (i * 256 * z, j * 256 * z))
    m = img_field(big.resize((S, S), Image.LANCZOS))
    ms = img_field(ImageChops.offset(field_img(m), 1, 1))
    sheen = noise(200, 10, 302)
    paper = fbm(64, 64, 3, 303)
    base = Tex.solid((44, 60, 47)).shade(paper, 0.9, 1.08)
    motif = Tex.solid((102, 118, 82)).shade(sheen, 0.86, 1.1)
    base.lerp((26, 36, 28), ms, 0.55)                         # embossed shadow under the print
    base.lerp(motif, m)
    base.lerp((130, 142, 104), [max(0.0, a - b) for a, b in zip(m, ms)], 0.35)  # lit upper edge
    base.lerp((110, 100, 70), fbm(3, 3, 4, 304), 0.18)        # yellowed varnish/fading
    # vertical drip streaks from water damage
    drip = noise(60, 3, 305)
    region = fbm(3, 2, 4, 306)
    streak = [sstep(0.62, 0.8, a) * sstep(0.5, 0.75, b) for a, b in zip(drip, region)]
    base.lerp((70, 58, 36), streak, 0.5)
    stain(base, 307, 0.16, (88, 82, 52), (52, 42, 26), cells=3, k=0.4, ring_k=0.55, rings=3)
    # peeling paper: exposed plaster backing with a lifted, shadowed edge
    pn = uniform([0.6 * a + 0.4 * b for a, b in zip(fbm(10, 10, 4, 308), fbm(3, 3, 3, 309))])
    peel = [sstep(0.955, 0.962, v) for v in pn]
    lip = [max(0.0, 1.0 - abs(v - 0.951) * 160.0) for v in pn]
    shadow = img_field(ImageChops.offset(field_img(peel), 2, 3))
    base.lerp((14, 18, 14), [max(0.0, s - p) for s, p in zip(shadow, peel)], 0.7)
    base.lerp((150, 140, 112), peel)
    base.lerp((120, 128, 98), lip, 0.6)
    # mildew speckle clusters
    cluster = fbm(4, 4, 4, 310)
    spk = speckles(311, 900, 0.5, 1.6)
    base.lerp((18, 22, 16), [s * sstep(0.6, 0.8, c) for s, c in zip(spk, cluster)], 0.8)
    # strip seam at the tile edge
    seam = [1.0] * N
    for y in range(S):
        seam[y * S] = 0.55
        seam[y * S + 1] = 1.12
    base.mul(seam)
    return grime(base, 312, 0.78, 1.05).image()


def gen_wallpaper_faded():
    z = 2
    W = S * z
    rose, deep, cream, light = (184, 136, 130), (160, 110, 110), (206, 188, 170), (196, 158, 150)
    im = Image.new('RGB', (W, W), rose)
    d = ImageDraw.Draw(im)
    for s in range(4):
        x0 = s * 128
        d.rectangle([(x0 + 64) * z, 0, (x0 + 70) * z - 1, W], fill=cream)
        d.rectangle([(x0 + 70) * z, 0, (x0 + 82) * z - 1, W], fill=deep)
        d.rectangle([(x0 + 82) * z, 0, (x0 + 88) * z - 1, W], fill=cream)
        d.rectangle([(x0 + 88) * z, 0, (x0 + 128) * z - 1, W], fill=light)
        for yy in range(0, 512, 16):                       # tiny dots in the light band
            d.ellipse(ellipse_bbox((x0 + 108) * z, (yy + 8 + (s % 2) * 8) * z, 1.6 * z), fill=(176, 128, 124))
    bud, bud_dk, leaf = (156, 88, 94), (118, 60, 66), (124, 136, 106)

    def rosebud(cx, cy):
        for side in (-1, 1):
            d.polygon(sc(ellipse_poly(cx + side * 9, cy + 7, 9, 3.6, side * 0.6), z), fill=leaf)
        d.line(sc([(cx, cy + 8), (cx - 1, cy + 18), (cx + 2, cy + 26)], z), fill=leaf, width=int(1.6 * z))
        d.ellipse(ellipse_bbox(cx * z, cy * z, 8.5 * z, 8 * z), fill=bud)
        d.line(sc(spiral(cx + 0.5, cy, 6.5, 1.2, 1.6, 0.5), z), fill=bud_dk, width=int(1.4 * z))
        d.ellipse(ellipse_bbox((cx - 2) * z, (cy - 3) * z, 2.5 * z), fill=(176, 108, 112))

    for s in range(4):
        for j in range(5):
            cy = j * 128 + 64 + (s % 2) * 64 - 128
            for dy in (0, 512):
                if -40 < cy + dy < 552:
                    rosebud(s * 128 + 32, cy + dy)
    # small sprig between buds
    for s in range(4):
        for j in range(4):
            cy = j * 128 + (s % 2) * 64
            for dy in (0, 512, -512):
                cx, yy = s * 128 + 32, cy + dy
                d.polygon(sc(ellipse_poly(cx - 4, yy, 5, 2, 0.7), z), fill=leaf)
                d.polygon(sc(ellipse_poly(cx + 4, yy, 5, 2, -0.7), z), fill=leaf)
                d.ellipse(ellipse_bbox(cx * z, (yy - 3) * z, 2 * z), fill=bud)
    t = Tex.from_image(im.resize((S, S), Image.LANCZOS))
    t.shade(fbm(64, 64, 3, 321), 0.93, 1.05)                 # paper tooth
    t.lerp((206, 190, 172), fbm(3, 3, 4, 322), 0.45)        # sun fading
    t.lerp((170, 145, 110), fbm(4, 4, 4, 323), 0.2)
    stain(t, 324, 0.14, (176, 148, 106), (122, 92, 60), cells=3, k=0.45, ring_k=0.5, rings=3)
    # mould: dark green-grey blooms with speckled fringe
    mf = uniform(fbm(4, 4, 6, 325))
    spk = speckles(326, 1600, 0.5, 1.8)
    # kept faint and small: the wall tiles every 2 m, so big dark blooms read as a repeating pattern
    core = [sstep(0.95, 0.99, v) for v in mf]
    fringe = [s * sstep(0.88, 0.95, v) for s, v in zip(spk, mf)]
    t.lerp((66, 72, 54), core, 0.4)
    t.lerp((40, 44, 34), fringe, 0.45)
    drip = noise(70, 3, 327)
    t.lerp((120, 96, 70), [sstep(0.64, 0.82, a) * sstep(0.55, 0.8, b) for a, b in zip(drip, fbm(3, 2, 3, 328))], 0.4)
    seam = [1.0] * N
    for y in range(S):
        seam[y * S] = 0.7
        seam[y * S + 1] = 1.08
    t.mul(seam)
    return grime(t, 329, 0.78, 1.04).image()


def gen_plaster():
    t = Tex.solid((196, 190, 176))
    t.shade(fbm(16, 16, 5, 401), 0.9, 1.06)
    t.shade(noise(S, S, 402), 0.97, 1.02)
    t.lerp((178, 168, 148), fbm(3, 3, 4, 403), 0.4)
    stain(t, 404, 0.2, (178, 152, 102), (130, 100, 62), cells=2, k=0.5, ring_k=0.55, rings=3)
    stain(t, 405, 0.1, (168, 150, 110), (120, 96, 66), cells=4, k=0.35, ring_k=0.45, rings=2)
    # fine crazing network + a few long branching cracks
    f1, f2, _ = voronoi(9, 9, 406, 0.95)
    craze = [1.0 - sstep(0.4, 1.4, b - a) for a, b in zip(f1, f2)]
    t.lerp((120, 110, 96), [c * v for c, v in zip(craze, fbm(4, 4, 3, 407))], 0.6)
    cr = cracks(408, 6, 130, 2.6, 0.5, 0.05)
    hi = img_field(ImageChops.offset(field_img(cr), 1, 1))
    t.lerp((228, 224, 212), [max(0.0, h - c) for h, c in zip(hi, cr)], 0.35)
    t.lerp((78, 70, 60), cr, 0.85)
    spots = [s * sstep(0.6, 0.85, v) for s, v in zip(speckles(409, 700, 0.5, 1.5), fbm(3, 3, 4, 410))]
    t.lerp((70, 72, 58), spots, 0.7)
    return grime(t, 411, 0.8, 1.03).image()


# =============================================================================
# Masonry
# =============================================================================
def gen_stone():
    rnd = random.Random(501)
    heights = partition(S, 70, 118, rnd)
    rows = [(h, partition(S, 90, 210, rnd), rnd.randrange(S)) for h in heights]
    bid, dist = masonry(rows, 502, warp=9.0, rc=10.0)
    nb = sum(len(r[1]) for r in rows)
    tone = [rnd.uniform(0.8, 1.14) for _ in range(nb)]
    warm = [rnd.uniform(-1.0, 1.0) for _ in range(nb)]
    surf = fbm(12, 12, 5, 503)
    grit = noise(S, S, 504)
    M = 3.4
    r, g, b = [0.0] * N, [0.0] * N, [0.0] * N
    for i in range(N):
        d = dist[i]
        k = bid[i]
        tt = tone[k] * (0.76 + 0.42 * surf[i]) + 0.1 * (grit[i] - 0.5)
        w = warm[k] * 7.0
        sh = 0.58 + 0.42 * sstep(M, M + 13.0, d)
        sr, sg, sb = (136 + w) * tt * sh, 132 * tt * sh, (126 - w) * tt * sh
        mo = 1.0 - sstep(M - 1.0, M + 1.0, d)
        if mo > 0.0:
            mt = (0.62 + 0.3 * grit[i]) * (0.75 + 0.25 * sstep(0.0, M, d))
            sr += (104 * mt - sr) * mo
            sg += (98 * mt - sg) * mo
            sb += (88 * mt - sb) * mo
        r[i], g[i], b[i] = sr, sg, sb
    t = Tex(r, g, b)
    t.lerp((62, 60, 56), [sstep(0.86, 0.96, v) for v in noise(170, 170, 505)], 0.55)   # pitting
    t.lerp((176, 172, 164), scratches(506, 40, 6, 24, 1, 0.3), 0.3)                    # chisel marks
    soot = fbm(3, 3, 5, 507)
    t.lerp((40, 36, 33), [sstep(0.45, 0.95, v) for v in soot], 0.45)
    stain(t, 508, 0.08, (120, 112, 96), (80, 74, 62), k=0.25, ring_k=0.3)
    return grime(t, 509, 0.8, 1.05).image()


def gen_brick_soot():
    rnd = random.Random(601)
    heights = [43] * 8 + [42] * 4
    rnd.shuffle(heights)
    rows = [(h, [128] * 4, (64 if r % 2 else 0) + rnd.randint(-8, 8)) for r, h in enumerate(heights)]
    bid, dist = masonry(rows, 602, warp=2.5, rc=4.0)
    pal = [(128, 60, 42), (112, 52, 38), (136, 72, 48), (98, 48, 36), (120, 66, 52), (88, 44, 34)]
    cols = [pal[rnd.randrange(len(pal))] for _ in range(48)]
    btone = [rnd.uniform(0.8, 1.15) for _ in range(48)]
    bsoot = [rnd.uniform(0.0, 0.55) for _ in range(48)]
    surf = fbm(16, 16, 4, 603)
    grit = noise(S, S, 604)
    M = 3.0
    r, g, b, sootb = [0.0] * N, [0.0] * N, [0.0] * N, [0.0] * N
    for i in range(N):
        d = dist[i]
        k = bid[i]
        c = cols[k]
        tt = btone[k] * (0.8 + 0.35 * surf[i]) + 0.12 * (grit[i] - 0.5)
        sh = 0.66 + 0.34 * sstep(M, M + 6.0, d)
        sr, sg, sb = c[0] * tt * sh, c[1] * tt * sh, c[2] * tt * sh
        mo = 1.0 - sstep(M - 0.8, M + 0.8, d)
        if mo > 0.0:
            mt = 0.55 + 0.35 * grit[i]
            sr += (92 * mt - sr) * mo
            sg += (86 * mt - sg) * mo
            sb += (78 * mt - sb) * mo
        r[i], g[i], b[i] = sr, sg, sb
        sootb[i] = bsoot[k]
    t = Tex(r, g, b)
    soot = uniform(fbm(4, 3, 5, 605))
    t.lerp((20, 17, 15), [clamp(sstep(0.2, 0.95, v) * 0.75 + s) for v, s in zip(soot, sootb)], 0.85)
    t.lerp((70, 66, 62), [sstep(0.9, 0.97, v) for v in noise(150, 150, 606)], 0.4)       # ash flecks
    t.lerp((14, 12, 11), [sstep(0.88, 0.97, v) for v in noise(200, 200, 607)], 0.5)      # pits
    return grime(t, 608, 0.8, 1.06).image()


# =============================================================================
# Fabrics
# =============================================================================
def gen_velvet_red():
    """Diamond-tufted velvet: buttons on a 128 px diamond lattice, puffed cushions between."""
    P = 128.0
    nap = fbm(8, 8, 4, 701)
    fine = noise(256, 256, 702)
    lx, ly, lz = -0.45, -0.6, 0.66
    v = [0.0] * N
    btn = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        a = ((x + y) % P) / P
        b = ((x - y) % P) / P
        sa, sb = math.sin(math.pi * a), math.sin(math.pi * b)
        ca, cb = math.cos(math.pi * a), math.cos(math.pi * b)
        h = sa * sb
        dha, dhb = math.pi * ca * sb, math.pi * sa * cb
        k = 22.0
        dhx = (dha + dhb) / P * k
        dhy = (dha - dhb) / P * k
        nl = math.sqrt(dhx * dhx + dhy * dhy + 1.0)
        lam = max(0.0, (-dhx * lx - dhy * ly + lz) / nl)
        s = 0.3 + 0.55 * lam + 0.25 * h ** 0.6
        da = min(a, 1.0 - a) * P
        db = min(b, 1.0 - b) * P
        dist = math.sqrt((da * da + db * db) * 0.5)
        if dist < 26:
            s *= 0.55 + 0.45 * sstep(4.0, 26.0, dist)
            fold = min(da, db)
            s *= 0.75 + 0.25 * sstep(0.0, 4.0, fold)
        btn[i] = 1.0 - sstep(5.0, 6.5, dist)
        v[i] = s * (0.86 + 0.22 * nap[i]) * (0.95 + 0.1 * fine[i])
    t = Tex.grad([clamp(q * 0.95) for q in v],
                 [(0.0, (34, 4, 8)), (0.35, (92, 14, 22)), (0.65, (142, 26, 36)), (1.0, (190, 62, 66))])
    t.lerp((40, 8, 12), btn, 0.9)
    wear = uniform(fbm(5, 5, 4, 703))
    t.lerp((150, 82, 78), [sstep(0.82, 0.98, w) * clamp(q) for w, q in zip(wear, v)], 0.45)
    t.lerp((120, 108, 104), speckles(704, 260, 0.5, 1.3), 0.35)
    return grime(t, 705, 0.82, 1.05).image()


def gen_linen():
    rnd = random.Random(801)
    cols = [rnd.uniform(0.88, 1.06) for _ in range(256)]
    rowv = [rnd.uniform(0.88, 1.06) for _ in range(256)]
    for _ in range(18):                                   # slubs: thicker, brighter thread runs
        cols[rnd.randrange(256)] *= 1.07
        rowv[rnd.randrange(256)] *= 1.07
    along = noise(S, 24, 802)
    across = noise(24, S, 803)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        tx, ty = x >> 1, y >> 1
        if (tx + ty) & 1:
            q = cols[tx] * (0.94 + 0.1 * along[i]) * (1.0 if x & 1 else 0.93)
        else:
            q = rowv[ty] * (0.94 + 0.1 * across[i]) * (1.0 if y & 1 else 0.93)
        v[i] = q
    t = Tex.solid((198, 191, 174)).mul(v)
    t.lerp((206, 200, 186), fbm(4, 4, 4, 804), 0.3)
    stain(t, 805, 0.12, (196, 176, 128), (160, 136, 90), cells=3, k=0.4, ring_k=0.4)
    stain(t, 806, 0.05, (170, 150, 130), (130, 110, 96), cells=5, k=0.35, ring_k=0.35)
    t.lerp((140, 130, 110), speckles(807, 120, 0.5, 1.2), 0.3)
    return grime(t, 808, 0.84, 1.03).image()


def gen_quilt():
    rnd = random.Random(901)
    z = 2
    P = 64
    im = Image.new('RGB', (S * z, S * z))
    d = ImageDraw.Draw(im)
    fabrics = [(178, 124, 122), (140, 152, 120), (122, 140, 162), (184, 158, 96), (206, 194, 168),
               (152, 134, 162), (146, 112, 86), (170, 96, 88), (110, 128, 110)]
    kinds = ['solid', 'gingham', 'stripe', 'dots', 'floral', 'hst', 'floral', 'dots']

    def lighter(c, f):
        return tuple(int(clamp(ch * f, 0, 255)) for ch in c)

    for j in range(8):
        for i in range(8):
            x0, y0 = i * P * z, j * P * z
            x1, y1 = x0 + P * z - 1, y0 + P * z - 1
            c1 = fabrics[rnd.randrange(len(fabrics))]
            c2 = fabrics[rnd.randrange(len(fabrics))]
            kind = kinds[rnd.randrange(len(kinds))]
            d.rectangle([x0, y0, x1, y1], fill=c1)
            if kind == 'gingham':
                for k in range(0, P, 8):
                    d.rectangle([x0 + k * z, y0, x0 + (k + 4) * z - 1, y1], fill=lighter(c1, 0.8))
                    d.rectangle([x0, y0 + k * z, x1, y0 + (k + 4) * z - 1], fill=lighter(c1, 0.8))
                for k in range(0, P, 8):
                    for l in range(0, P, 8):
                        d.rectangle([x0 + k * z, y0 + l * z, x0 + (k + 4) * z - 1, y0 + (l + 4) * z - 1],
                                    fill=lighter(c1, 0.64))
            elif kind == 'stripe':
                for k in range(0, P, 10):
                    d.rectangle([x0 + k * z, y0, x0 + (k + 4) * z - 1, y1], fill=lighter(c2, 0.9))
            elif kind == 'dots':
                for k in range(4, P, 12):
                    for l in range(4, P, 12):
                        o = 6 if (l // 12) % 2 else 0
                        d.ellipse(ellipse_bbox(x0 + ((k + o) % P) * z, y0 + l * z, 2.2 * z), fill=lighter(c2, 1.12))
            elif kind == 'floral':
                for _ in range(10):
                    fx, fy = rnd.uniform(6, P - 6), rnd.uniform(6, P - 6)
                    for a in range(5):
                        ang = a * TAU / 5
                        d.ellipse(ellipse_bbox(x0 + (fx + 2.6 * math.cos(ang)) * z, y0 + (fy + 2.6 * math.sin(ang)) * z,
                                               1.8 * z), fill=lighter(c2, 1.15))
                    d.ellipse(ellipse_bbox(x0 + fx * z, y0 + fy * z, 1.2 * z), fill=(200, 180, 110))
                    d.polygon(sc(ellipse_poly(fx + 5, fy + 3, 3, 1.3, 0.6), z, x0, y0), fill=(118, 132, 100))
            elif kind == 'hst':
                d.polygon([(x0, y1), (x1, y0), (x1, y1)], fill=c2)
    t = Tex.from_image(im.resize((S, S), Image.LANCZOS))
    weave = noise(S, S, 902)
    sh = [0.0] * N
    stitch = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        lx, ly = x % P, y % P
        de = min(lx, P - 1 - lx, ly, P - 1 - ly)
        puff = math.sin(math.pi * (lx + 0.5) / P) * math.sin(math.pi * (ly + 0.5) / P)
        sh[i] = (0.6 + 0.4 * sstep(0.0, 9.0, de)) * (0.9 + 0.14 * puff) * (0.95 + 0.08 * weave[i])
        if de == 4:
            along = ly if (lx == 4 or lx == P - 5) else lx
            stitch[i] = 1.0 if (along // 3) % 2 == 0 else 0.0
    t.mul(sh)
    t.lerp((222, 210, 186), stitch, 0.7)
    t.lerp((190, 178, 158), fbm(3, 3, 4, 903), 0.35)      # fading
    stain(t, 904, 0.1, (170, 142, 100), (120, 94, 64), cells=3, k=0.35, ring_k=0.4)
    t.lerp((150, 140, 128), speckles(905, 200, 0.5, 1.3), 0.3)
    return grime(t, 906, 0.8, 1.04).image()


def gen_coat():
    fib = noise(S, S, 1001)
    fib2 = noise(256, 64, 1002)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        tw = 0.5 + 0.5 * math.cos(TAU * (x + y) / 8.0)       # 2/2 twill diagonal
        v[i] = 0.82 + 0.13 * tw + 0.1 * fib[i] + 0.06 * fib2[i]
    t = Tex.solid((56, 55, 60)).mul(v)
    t.shade(fbm(6, 6, 4, 1003), 0.9, 1.08)
    t.lerp((104, 102, 106), [sstep(0.975, 0.995, q) for q in fib], 0.6)   # heather fibres
    t.lerp((120, 116, 110), speckles(1004, 140, 0.5, 1.2), 0.3)          # lint
    return grime(t, 1005, 0.86, 1.04).image()


def gen_trousers():
    fib = noise(S, S, 1101)
    v = [0.0] * N
    pin = [0.0] * N
    gap = noise(4, 64, 1102)
    for i in range(N):
        x = i & 511
        y = i >> 9
        w = math.cos(TAU * x / 4.0) * math.cos(TAU * y / 4.0)
        v[i] = 0.9 + 0.06 * w + 0.08 * fib[i]
        if x % 32 == 16:
            pin[i] = 0.55 * sstep(0.25, 0.4, gap[i])
    t = Tex.solid((76, 75, 80)).mul(v)
    t.lerp((122, 120, 124), pin)
    t.shade(fbm(5, 5, 4, 1103), 0.86, 1.06)
    t.lerp((118, 112, 104), speckles(1104, 120, 0.5, 1.2), 0.3)
    return grime(t, 1105, 0.86, 1.04).image()


def gen_doll_dress():
    z = 2
    W = S * z
    lace = Image.new('L', (W, W), 0)
    holes = Image.new('L', (W, W), 0)
    dl, dh = ImageDraw.Draw(lace), ImageDraw.Draw(holes)
    for k in range(-64, 64):                                 # diagonal net
        o = k * 16
        dl.line([(o, 0), (o + W, W)], fill=150, width=2)
        dl.line([(o, W), (o + W, 0)], fill=150, width=2)

    def flower(cx, cy):
        for p in range(6):
            a = p * TAU / 6
            pts = ellipse_poly(cx + 13 * math.cos(a), cy + 13 * math.sin(a), 9, 5.5, a, 24)
            dl.polygon(sc(pts, z), fill=255)
            dh.ellipse(ellipse_bbox((cx + 13 * math.cos(a)) * z, (cy + 13 * math.sin(a)) * z, 2.6 * z), fill=255)
        dl.ellipse(ellipse_bbox(cx * z, cy * z, 6 * z), fill=255)
        dh.ellipse(ellipse_bbox(cx * z, cy * z, 2.5 * z), fill=255)
        dl.ellipse(ellipse_bbox(cx * z, cy * z, 26 * z), outline=230, width=int(1.6 * z))

    for j in range(8):
        for i in range(8):
            cx = i * 64 + 32 + (32 if j % 2 else 0)
            cy = j * 64 + 32
            for ox in (0, -512):
                flower(cx + ox, cy)
    for row in range(2):                                     # scalloped edges
        yy = row * 256 + 128
        for k in range(16):
            cx = k * 32 + 16
            dl.arc(ellipse_bbox(cx * z, yy * z, 16 * z, 12 * z), 0, 180, fill=255, width=int(3 * z))
            dl.ellipse(ellipse_bbox(cx * z, (yy + 16) * z, 2.5 * z), fill=255)
        dl.line([(0, (yy - 2) * z), (W, (yy - 2) * z)], fill=255, width=int(2.4 * z))
    m = img_field(ImageChops.subtract(lace, holes).resize((S, S), Image.LANCZOS))
    hl = img_field(holes.resize((S, S), Image.LANCZOS))
    shadow = img_field(blur_wrap(ImageChops.offset(field_img(m), 1, 2), 1.2))
    folds = [0.0] * N
    for i in range(N):
        x = i & 511
        folds[i] = 0.5 + 0.3 * math.cos(TAU * 3 * x / S) + 0.2 * math.cos(TAU * 7 * x / S + 1.3)
    t = Tex.solid((200, 150, 156))
    t.shade(folds, 0.82, 1.08)
    t.shade(noise(S, S, 1201), 0.95, 1.03)
    t.lerp((120, 80, 86), shadow, 0.35)
    t.lerp((120, 84, 88), hl, 0.6)
    t.lerp((232, 220, 212), m, 0.82)
    t.lerp((205, 185, 170), fbm(3, 3, 4, 1202), 0.3)            # fading
    stain(t, 1203, 0.12, (176, 140, 112), (128, 96, 76), cells=3, k=0.4, ring_k=0.45)
    t.lerp((110, 100, 96), speckles(1204, 200, 0.5, 1.3), 0.35)
    return grime(t, 1205, 0.8, 1.03).image()


def gen_curtain():
    rnd = random.Random(1301)
    terms = [(k, rnd.uniform(0.4, 1.0) / (k ** 0.6), rnd.uniform(0, TAU)) for k in (4, 6, 9, 13, 17)]
    wav = noise(4, 6, 1302)
    nap = fbm(10, 4, 4, 1303)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        xs = x + 0.0
        h = dh = 0.0
        u = TAU * xs / S
        sway = (wav[i] - 0.5) * 0.25
        for k, a, ph in terms:
            arg = k * u + ph + sway * k * 0.3
            h += a * math.cos(arg)
            dh -= a * k * math.sin(arg)
        v[i] = h * 0.5 - dh * 0.06 + (nap[i] - 0.5) * 0.25
    v = normalize(v)
    t = Tex.grad(v, [(0.0, (34, 4, 8)), (0.3, (78, 12, 18)), (0.6, (124, 22, 30)), (0.85, (162, 40, 46)),
                     (1.0, (196, 76, 78))])
    t.shade(noise(S, 64, 1304), 0.93, 1.05)                  # vertical pile streaks
    t.lerp((120, 92, 88), [sstep(0.6, 0.95, q) for q in fbm(4, 2, 4, 1305)], 0.18)   # dust bloom on folds
    t.lerp((116, 104, 100), speckles(1306, 220, 0.5, 1.4), 0.35)
    return grime(t, 1307, 0.82, 1.05).image()


# =============================================================================
# Rugs
# =============================================================================
def gen_carpet_persian():
    """Rug laid out on a 768x512 design (3:2) then squeezed to 512x512, so it reads with
    correct proportions once stretched over the 3 m x 2 m rug."""
    z = 2
    DW, DH = 768, 512
    rnd = random.Random(1401)
    navy, red, ivory, gold, rust = (36, 42, 74), (138, 34, 32), (206, 190, 156), (196, 150, 72), (160, 70, 40)
    green = (64, 92, 76)
    im = Image.new('RGB', (DW * z, DH * z), red)
    d = ImageDraw.Draw(im)

    def R(x0, y0, x1, y1):
        return [x0 * z, y0 * z, x1 * z - 1, y1 * z - 1]

    FR = 16                                                   # fringe length at the short ends
    for x0 in (0, DW - FR):
        d.rectangle(R(x0, 0, x0 + FR, DH), fill=(70, 60, 50))
        for k in range(0, DH, 3):
            d.line([(x0 * z, (k + 1) * z), ((x0 + FR) * z, (k + 1) * z + rnd.randint(-1, 1))],
                   fill=(214, 202, 176), width=z)
    bx0, by0, bx1, by1 = FR, 0, DW - FR, DH
    bands = [(5, navy), (4, ivory), (8, rust), (4, ivory), (52, navy), (4, ivory), (7, red), (3, gold), (5, navy)]
    inset = 0
    border_rect = None
    for w, c in bands:
        d.rectangle(R(bx0 + inset, by0 + inset, bx1 - inset, by1 - inset), outline=c, width=w * z)
        if w == 52:
            border_rect = (inset, w)
        inset += w
    fx0, fy0, fx1, fy1 = bx0 + inset, by0 + inset, bx1 - inset, by1 - inset
    # rust guard stripe dots
    o = 5 + 4 + 4
    for x in range(bx0 + 12, bx1 - 12, 10):
        for yy in (by0 + o, by1 - o):
            d.ellipse(ellipse_bbox(x * z, yy * z, 1.6 * z), fill=gold)
    for y in range(by0 + 12, by1 - 12, 10):
        for xx in (bx0 + o, bx1 - o):
            d.ellipse(ellipse_bbox(xx * z, y * z, 1.6 * z), fill=gold)
    # main border: vine + rosettes
    bi, bw = border_rect
    mid = bi + bw / 2.0

    def rosette(cx, cy, r):
        for k in range(8):
            a = k * TAU / 8
            d.polygon(sc(ellipse_poly(cx + r * 0.75 * math.cos(a), cy + r * 0.75 * math.sin(a), r * 0.42, r * 0.24, a, 16), z),
                      fill=ivory)
        d.ellipse(ellipse_bbox(cx * z, cy * z, r * 0.55 * z), fill=rust)
        d.ellipse(ellipse_bbox(cx * z, cy * z, r * 0.3 * z), fill=gold)
        d.ellipse(ellipse_bbox(cx * z, cy * z, r * 0.12 * z), fill=navy)

    top, bot = by0 + mid, by1 - mid
    left, right = bx0 + mid, bx1 - mid
    for (ax, ay, bx_, by_) in ((left, top, right, top), (left, bot, right, bot), (left, top, left, bot), (right, top, right, bot)):
        ln = math.hypot(bx_ - ax, by_ - ay)
        n = int(ln / 3)
        ux, uy = (bx_ - ax) / ln, (by_ - ay) / ln
        pts = []
        for k in range(n + 1):
            s = ln * k / n
            off = 11 * math.sin(TAU * s / 56.0)
            pts.append(((ax + ux * s - uy * off) * z, (ay + uy * s + ux * off) * z))
        d.line(pts, fill=gold, width=int(2.2 * z))
        cnt = int(ln / 56)
        for k in range(cnt + 1):
            s = ln * k / cnt
            rosette(ax + ux * s, ay + uy * s, 13)
            s2 = s + ln / cnt / 2
            if k < cnt:
                lx_, ly_ = ax + ux * s2, ay + uy * s2
                d.polygon(sc(ellipse_poly(lx_, ly_, 7, 3.5, math.atan2(uy, ux) + 0.8, 16), z), fill=green)
                d.polygon(sc(ellipse_poly(lx_, ly_, 7, 3.5, math.atan2(uy, ux) - 0.8, 16), z), fill=(150, 60, 48))
    # field all-over lattice of small motifs
    for j, y in enumerate(range(fy0 + 14, fy1 - 6, 26)):
        for x in range(fx0 + 14 + (13 if j % 2 else 0), fx1 - 6, 26):
            d.polygon(sc([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], z), fill=navy)
            d.ellipse(ellipse_bbox(x * z, y * z, 1.8 * z), fill=ivory)
            for a in range(4):
                ang = a * math.pi / 2 + math.pi / 4
                d.ellipse(ellipse_bbox((x + 9 * math.cos(ang)) * z, (y + 9 * math.sin(ang)) * z, 1.3 * z), fill=gold)
    # corner spandrels

    def lobed(cx, cy, rx, ry, lobes, depth, n=240):
        pts = []
        for k in range(n):
            t = TAU * k / n
            rr = 1.0 - depth * (0.5 - 0.5 * math.cos(lobes * t))
            pts.append(((cx + rx * rr * math.cos(t)) * z, (cy + ry * rr * math.sin(t)) * z))
        return pts

    for cx in (fx0, fx1):
        for cy in (fy0, fy1):
            d.polygon(lobed(cx, cy, 108, 84, 14, 0.08), fill=ivory)
            d.polygon(lobed(cx, cy, 102, 78, 14, 0.08), fill=navy)
            d.polygon(lobed(cx, cy, 60, 46, 10, 0.1), fill=rust)
            d.polygon(lobed(cx, cy, 34, 26, 8, 0.12), fill=gold)
            for k in range(14):
                a = TAU * k / 14
                d.ellipse(ellipse_bbox((cx + 82 * math.cos(a)) * z, (cy + 63 * math.sin(a)) * z, 3 * z), fill=ivory)
    # repaint the border frame over the spandrels' overflow
    inset = 0
    for w, c in bands:
        if w != 52:
            d.rectangle(R(bx0 + inset, by0 + inset, bx1 - inset, by1 - inset), outline=c, width=w * z)
        inset += w
    # central medallion with pendants
    cx, cy = DW / 2.0, DH / 2.0
    for px_ in (cx - 196, cx + 196):
        d.polygon(sc([(px_ - 22, cy), (px_, cy - 18), (px_ + 22, cy), (px_, cy + 18)], z), fill=ivory)
        d.polygon(sc([(px_ - 17, cy), (px_, cy - 13), (px_ + 17, cy), (px_, cy + 13)], z), fill=navy)
        d.ellipse(ellipse_bbox(px_ * z, cy * z, 5 * z), fill=gold)
    layers = [(178, 124, ivory), (172, 118, navy), (120, 84, ivory), (115, 80, rust), (72, 52, ivory), (68, 48, navy),
              (36, 28, gold), (20, 16, red)]
    for rx, ry, c in layers:
        pts = []
        for k in range(360):
            t = TAU * k / 360
            dia = 1.0 - 0.3 * (1.0 - abs(math.cos(2 * t)) ** 0.7)
            rr = dia * (1.0 - 0.05 * (0.5 - 0.5 * math.cos(24 * t)))
            pts.append(((cx + rx * rr * math.cos(t)) * z, (cy + ry * rr * math.sin(t)) * z))
        d.polygon(pts, fill=c)
    for k in range(16):
        a = TAU * k / 16
        d.ellipse(ellipse_bbox((cx + 140 * math.cos(a) * (1 - 0.3 * (1 - abs(math.cos(2 * a)) ** 0.7))) * z,
                               (cy + 97 * math.sin(a) * (1 - 0.3 * (1 - abs(math.cos(2 * a)) ** 0.7))) * z, 4 * z), fill=gold)
    for k in range(8):
        a = TAU * k / 8
        d.polygon(sc(ellipse_poly(cx + 92 * math.cos(a) * 0.75, cy + 64 * math.sin(a) * 0.75, 12, 5, a, 16), z), fill=ivory)
    t = Tex.from_image(im.resize((S, S), Image.LANCZOS))
    # knotted pile, abrash (dye-lot banding along the length), wear, dirt
    t.shade(noise(S, S, 1402), 0.9, 1.06)
    t.shade(noise(128, 128, 1403), 0.94, 1.04)
    abr = noise(24, 3, 1404)
    t.tint((1.0, 0.97, 0.97))
    t.shade(abr, 0.9, 1.06)
    wear = uniform([0.65 * a + 0.35 * b for a, b in zip(fbm(6, 4, 5, 1405), fbm(2, 2, 3, 1406))])
    wm = [sstep(0.78, 0.96, w) for w in wear]
    t.desat(wm, 0.6)
    t.lerp((168, 140, 112), wm, 0.45)
    t.lerp((150, 136, 112), [sstep(0.93, 0.99, w) * (0.6 + 0.4 * (((i & 511) // 2) % 2)) for i, w in enumerate(wear)], 0.4)
    stain(t, 1407, 0.06, (70, 46, 34), (50, 30, 22), cells=3, k=0.45, ring_k=0.4)
    t.lerp((120, 112, 100), speckles(1408, 200, 0.5, 1.2), 0.3)
    return grime(t, 1409, 0.78, 1.04).image()


def gen_carpet_round():
    z = 2
    W = S * z
    c = W / 2.0
    rnd = random.Random(1501)
    bind, border, ivory, rose, field, deep, gold = ((46, 50, 62), (64, 76, 98), (186, 180, 162), (150, 100, 96),
                                                    (104, 120, 138), (60, 72, 96), (170, 150, 104))
    im = Image.new('RGB', (W, W), bind)
    d = ImageDraw.Draw(im)

    def circ(r, col, outline=None, width=0):
        d.ellipse(ellipse_bbox(c, c, r * z), fill=col, outline=outline, width=width)

    circ(250, border)
    for k in range(240):                                     # vine ring
        a0, a1 = TAU * k / 240, TAU * (k + 1) / 240
        r0 = 236 + 5 * math.sin(a0 * 24)
        r1 = 236 + 5 * math.sin(a1 * 24)
        d.line([(c + r0 * z * math.cos(a0), c + r0 * z * math.sin(a0)), (c + r1 * z * math.cos(a1), c + r1 * z * math.sin(a1))],
               fill=gold, width=int(1.8 * z))
    for k in range(24):
        a = TAU * k / 24
        x, y = c / z + 236 * math.cos(a), c / z + 236 * math.sin(a)
        for p in range(6):
            b = p * TAU / 6
            d.ellipse(ellipse_bbox((x + 4.5 * math.cos(b)) * z, (y + 4.5 * math.sin(b)) * z, 2.8 * z), fill=ivory)
        d.ellipse(ellipse_bbox(x * z, y * z, 3 * z), fill=rose)
        a2 = a + TAU / 48
        x2, y2 = c / z + 236 * math.cos(a2), c / z + 236 * math.sin(a2)
        d.polygon(sc(ellipse_poly(x2, y2, 5, 2.2, a2 + 0.6, 12), z), fill=(120, 138, 116))
    circ(222, ivory)
    circ(218, rose)
    circ(214, field)
    for k in range(48):                                       # dotted guard inside field
        a = TAU * k / 48
        d.ellipse(ellipse_bbox(c + 206 * z * math.cos(a), c + 206 * z * math.sin(a), 2 * z), fill=ivory)
    for k in range(12):
        a = TAU * k / 12
        path = [(c / z + r * math.cos(a), c / z + r * math.sin(a)) for r in range(112, 199, 4)]
        d.polygon(sc(leaf_poly(path, 30, 0.7), z), fill=ivory)
        d.polygon(sc(leaf_poly(path, 26, 0.7), z), fill=(128, 144, 160))
        inner = [(c / z + r * math.cos(a), c / z + r * math.sin(a)) for r in range(124, 188, 4)]
        d.polygon(sc(leaf_poly(inner, 13, 0.7), z), fill=deep)
        d.line(sc(inner, z), fill=gold, width=int(1.5 * z))
        a2 = a + TAU / 24
        x, y = c / z + 186 * math.cos(a2), c / z + 186 * math.sin(a2)
        d.polygon(sc([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)], z), fill=rose)
        d.ellipse(ellipse_bbox(x * z, y * z, 1.6 * z), fill=ivory)
    circ(112, None, outline=ivory, width=int(2 * z))
    star = []
    for k in range(32):
        a = TAU * k / 32
        r = 102 if k % 2 == 0 else 76
        star.append((c + r * z * math.cos(a), c + r * z * math.sin(a)))
    d.polygon(star, fill=ivory)
    d.polygon([(c + (p[0] - c) * 0.94, c + (p[1] - c) * 0.94) for p in star], fill=deep)
    circ(56, ivory)
    circ(52, rose)
    for k in range(8):
        a = TAU * k / 8
        d.polygon(sc(ellipse_poly(c / z + 32 * math.cos(a), c / z + 32 * math.sin(a), 15, 7, a, 20), z), fill=field)
        d.polygon(sc(ellipse_poly(c / z + 32 * math.cos(a), c / z + 32 * math.sin(a), 10, 4, a, 20), z), fill=ivory)
    circ(13, gold)
    circ(6, deep)
    t = Tex.from_image(im.resize((S, S), Image.LANCZOS))
    t.shade(noise(S, S, 1502), 0.9, 1.06)
    t.shade(noise(128, 128, 1503), 0.94, 1.04)
    wear = uniform([0.65 * a + 0.35 * b for a, b in zip(fbm(6, 6, 5, 1504), fbm(2, 2, 3, 1505))])
    wm = [sstep(0.8, 0.97, w) for w in wear]
    t.desat(wm, 0.5)
    t.lerp((160, 156, 146), wm, 0.4)
    stain(t, 1506, 0.07, (92, 80, 66), (60, 50, 42), cells=3, k=0.4, ring_k=0.4)
    t.lerp((140, 134, 124), speckles(1507, 180, 0.5, 1.2), 0.3)
    # soft darkening towards the bound outer edge
    edge = [0.0] * N
    for i in range(N):
        x = (i & 511) - 255.5
        y = (i >> 9) - 255.5
        edge[i] = sstep(236.0, 256.0, math.hypot(x, y))
    t.shade(edge, 1.0, 0.78)
    return grime(t, 1508, 0.8, 1.04).image()


# =============================================================================
# Metals, glass, mirror
# =============================================================================
def gen_brass():
    brushed = noise(6, 300, 1601)
    t = Tex.grad([0.55 * a + 0.45 * b for a, b in zip(fbm(6, 6, 5, 1602), brushed)],
                 [(0.0, (118, 86, 40)), (0.45, (172, 134, 66)), (0.8, (206, 170, 96)), (1.0, (226, 196, 128))])
    tarn = uniform(fbm(5, 5, 6, 1603))
    tm = [sstep(0.55, 0.9, v) for v in tarn]
    t.lerp((86, 70, 38), tm, 0.6)
    t.lerp((60, 54, 34), [sstep(0.86, 0.97, v) for v in tarn], 0.5)
    verd = [s * sstep(0.8, 0.9, v) for s, v in zip(speckles(1604, 500, 0.6, 2.2, 0.6), tarn)]
    t.lerp((96, 128, 100), verd, 0.75)
    t.lerp((232, 206, 140), scratches(1605, 70, 8, 50, 1, 0.12), 0.4)
    t.lerp((70, 56, 30), scratches(1606, 30, 8, 40, 1, 0.12), 0.3)
    return grime(t, 1607, 0.86, 1.06).image()


def gen_iron():
    f1, _f2, _ = voronoi(26, 26, 1701, 0.85)
    cell = S / 26.0
    ham = [clamp(v / cell) for v in f1]
    t = Tex.solid((62, 62, 66))
    t.shade(ham, 1.08, 0.86)
    t.shade(fbm(16, 16, 4, 1702), 0.88, 1.1)
    t.shade(noise(S, S, 1703), 0.94, 1.05)
    rust = uniform(fbm(5, 5, 6, 1704))
    grain = noise(200, 200, 1705)
    rm = [sstep(0.8, 0.88, v) * (0.55 + 0.45 * g) for v, g in zip(rust, grain)]
    t.lerp((50, 36, 28), [max(0.0, 1.0 - abs(v - 0.79) * 40.0) for v in rust], 0.6)
    rcol = Tex.grad(grain, [(0.0, (96, 46, 24)), (0.5, (128, 66, 32)), (1.0, (156, 90, 44))])
    t.lerp(rcol, rm)
    streak = [sstep(0.6, 0.85, a) * sstep(0.6, 0.85, b) for a, b in zip(noise(50, 3, 1706), rust)]
    t.lerp((110, 62, 34), streak, 0.35)
    t.lerp((120, 120, 126), scratches(1707, 40, 6, 30, 1, 0.2), 0.35)
    return grime(t, 1708, 0.84, 1.05).image()


def gen_glass():
    streak = noise(48, 3, 1801)
    streak2 = noise(140, 6, 1802)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        band = math.exp(-((math.cos(TAU * (x - 150) / S) - 1.0) ** 2) * 8.0)
        band2 = math.exp(-((math.cos(TAU * (x - 330) / S) - 1.0) ** 2) * 30.0)
        ripple = 0.5 + 0.5 * math.sin(TAU * 9 * y / S + 3.0 * streak[i])
        v[i] = 0.42 + 0.22 * streak[i] + 0.1 * streak2[i] + 0.06 * ripple + 0.3 * band + 0.15 * band2
    t = Tex.grad([clamp(q) for q in v],
                 [(0.0, (110, 60, 22)), (0.4, (176, 112, 44)), (0.7, (214, 154, 72)), (1.0, (246, 214, 150))])
    rnd = random.Random(1803)
    m = Image.new('L', (S, S), 0)
    rim = Image.new('L', (S, S), 0)
    d, dr = ImageDraw.Draw(m), ImageDraw.Draw(rim)
    for _ in range(70):
        x, y, r = rnd.uniform(0, S), rnd.uniform(0, S), rnd.uniform(0.8, 3.0)
        wrap9(lambda dx, dy: dr.ellipse(ellipse_bbox(x + dx, y + dy, r), outline=200))
        wrap9(lambda dx, dy: d.ellipse(ellipse_bbox(x + dx - r * 0.3, y + dy - r * 0.3, r * 0.35), fill=255))
    t.lerp((120, 70, 30), img_field(rim), 0.5)
    t.lerp((255, 236, 190), img_field(m), 0.8)
    t.lerp((90, 70, 50), fbm(4, 4, 4, 1804), 0.15)
    return t.image()


def gen_mirror():
    base = fbm(2, 2, 4, 1902)
    blot = uniform(fbm(6, 6, 5, 1903))
    v = [0.0] * N
    edge = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        diag = (x + y) / (2.0 * S)
        streak = math.exp(-((diag - 0.42) / 0.09) ** 2) * 0.55 + math.exp(-((diag - 0.6) / 0.03) ** 2) * 0.3
        v[i] = 0.35 + 0.25 * base[i] + streak
        e = min(x, y, S - 1 - x, S - 1 - y) / (S * 0.5)
        edge[i] = 1.0 - sstep(0.0, 0.45, e)
    t = Tex.grad([clamp(q) for q in v], [(0.0, (70, 76, 82)), (0.5, (128, 134, 138)), (1.0, (196, 200, 200))])
    t.shade(fbm(24, 24, 3, 1904), 0.94, 1.04)
    ds = [0.55 * b + 0.6 * e for b, e in zip(blot, edge)]
    desil = [sstep(0.72, 0.78, q) for q in ds]
    halo = [max(0.0, 1.0 - abs(q - 0.7) * 20.0) for q in ds]
    t.lerp((120, 104, 78), halo, 0.5)
    dark = Tex.grad(noise(90, 90, 1905), [(0.0, (22, 20, 18)), (0.6, (48, 44, 40)), (1.0, (84, 76, 64))])
    t.lerp(dark, desil, 0.9)
    spk = speckles(1906, 900, 0.4, 1.6)
    t.lerp((20, 18, 16), [s * clamp(0.3 + e) for s, e in zip(spk, edge)], 0.8)
    fox = speckles(1907, 60, 2.0, 6.0, 1.5)
    t.lerp((110, 96, 76), fox, 0.35)
    t.lerp((210, 212, 214), scratches(1908, 40, 10, 60, 1, 0.05), 0.25)
    return t.image()


# =============================================================================
# Fire, candles
# =============================================================================
def gen_flame():
    flick = noise(14, 3, 2001)
    flick2 = noise(40, 5, 2002)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        u = x / S
        p = math.exp(-((u - 0.5) / 0.26) ** 2)
        p2 = math.exp(-((u - 0.5 + 1.0) / 0.26) ** 2) + math.exp(-((u - 0.5 - 1.0) / 0.26) ** 2)
        v[i] = clamp((p + p2) * (0.85 + 0.3 * flick[i]) + 0.12 * (flick2[i] - 0.5))
    t = Tex.grad(v, [(0.0, (236, 92, 18)), (0.3, (250, 140, 34)), (0.6, (255, 196, 70)), (0.85, (255, 236, 150)),
                     (1.0, (255, 252, 226))])
    return t.image()


def gen_embers():
    f1, f2, ids = voronoi(10, 8, 2101, 0.9)
    g1, g2, _ = voronoi(26, 22, 2102, 0.9)
    heat = fbm(5, 5, 4, 2103)
    grit = noise(S, S, 2104)
    rnd = random.Random(2105)
    ash_cell = [rnd.random() < 0.18 for _ in range(80)]
    glow = [0.0] * N
    char = [0.0] * N
    ash = [0.0] * N
    for i in range(N):
        e1 = f2[i] - f1[i]
        e2 = g2[i] - g1[i]
        h = heat[i]
        crack = 1.0 - sstep(0.6, 3.2, e1)
        halo = 1.0 - sstep(0.0, 16.0, e1)
        sub = (1.0 - sstep(0.3, 1.6, e2)) * sstep(0.35, 0.75, h)
        glow[i] = clamp(crack * (0.7 + 0.5 * h) + halo * 0.35 * h + sub * 0.55)
        char[i] = 0.7 + 0.3 * grit[i]
        ash[i] = (0.6 if ash_cell[ids[i]] else 0.0) * sstep(2.0, 10.0, e1) + 0.25 * sstep(0.82, 0.95, grit[i])
    t = Tex.solid((36, 31, 28)).mul(char)
    t.lerp((116, 110, 104), ash, 1.0)
    hot = Tex.grad(glow, [(0.0, (60, 14, 4)), (0.3, (160, 34, 6)), (0.6, (236, 98, 16)), (0.85, (255, 184, 60)),
                          (1.0, (255, 236, 160))])
    t.lerp(hot, [sstep(0.04, 0.4, g) for g in glow])
    return t.image()


def gen_wax():
    rnd = random.Random(2201)
    t = Tex.solid((226, 216, 192))
    t.lerp((214, 196, 160), fbm(3, 3, 4, 2202), 0.6)
    t.shade(fbm(12, 12, 3, 2203), 0.94, 1.04)
    drip = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(drip)
    for _ in range(26):
        x, y0 = rnd.uniform(0, S), rnd.uniform(0, S)
        ln, w = rnd.uniform(40, 200), rnd.uniform(4, 11)

        def draw_drip(dx, dy, x=x, y0=y0, ln=ln, w=w):
            pts = []
            for k in range(21):
                f = k / 20
                pts.append((x + dx + math.sin(f * 3.0 + x) * 2.0, y0 + dy + f * ln))
            for k in range(len(pts) - 1):
                ww = w * (1.0 - 0.45 * k / 20)
                d.line([pts[k], pts[k + 1]], fill=255, width=max(1, int(ww)))
            ex, ey = pts[-1]
            d.ellipse(ellipse_bbox(ex, ey, w * 0.62, w * 0.75), fill=255)
        wrap9(draw_drip)
    dm = img_field(blur_wrap(drip, 1.2))
    sh = img_field(ImageChops.offset(field_img(dm), -2, -1))
    t.lerp((176, 160, 130), [max(0.0, a - b) for a, b in zip(dm, sh)], 0.6)   # shadow side of drips
    t.lerp((246, 240, 222), [max(0.0, b - a) * 0.8 + a * 0.25 for a, b in zip(dm, sh)], 0.6)
    t.lerp((150, 140, 120), speckles(2204, 120, 0.5, 1.3), 0.3)
    return t.image()


# =============================================================================
# Characters
# =============================================================================
def gen_skin():
    t = Tex.solid((214, 192, 178))
    t.lerp((222, 178, 172), fbm(4, 4, 4, 2301), 0.45)
    t.lerp((198, 196, 196), fbm(3, 3, 4, 2302), 0.3)
    t.shade(noise(S, S, 2303), 0.975, 1.02)
    t.shade(noise(170, 170, 2304), 0.97, 1.02)
    t.lerp((180, 150, 140), [sstep(0.9, 0.98, v) for v in noise(120, 120, 2305)], 0.25)
    return t.image()


def gen_leather():
    f1, f2, _ = voronoi(64, 64, 2401, 0.9)
    cell = S / 64.0
    t = Tex.grad(fbm(5, 5, 5, 2402), [(0.0, (70, 42, 26)), (0.5, (102, 66, 42)), (1.0, (130, 88, 58))])
    dome = [1.0 - clamp(a / cell) for a in f1]
    t.shade(dome, 0.88, 1.06)
    t.lerp((44, 26, 16), [1.0 - sstep(0.0, 1.6, b - a) for a, b in zip(f1, f2)], 0.5)
    cr = cracks(2403, 14, 40, 3.0, 0.25, 0.02, 2)
    hi = img_field(ImageChops.offset(field_img(cr), 1, 1))
    t.lerp((150, 110, 78), [max(0.0, h - c) for h, c in zip(hi, cr)], 0.35)
    t.lerp((40, 24, 14), cr, 0.6)
    wear = uniform(fbm(4, 4, 5, 2404))
    t.lerp((150, 112, 80), [sstep(0.75, 0.97, w) for w in wear], 0.45)
    t.lerp((168, 134, 100), scratches(2405, 50, 6, 30, 1, 0.2), 0.35)
    return grime(t, 2406, 0.84, 1.05).image()


def gen_hair():
    warp = fbm(3, 4, 3, 2501)
    s1 = noise(320, 4, 2502)
    s2 = noise(160, 3, 2503)
    s3 = fbm(48, 2, 3, 2504)
    v = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        xs = int(x + (warp[i] - 0.5) * 60.0) & 511
        j = (y << 9) | xs
        v[i] = 0.45 * s1[j] + 0.3 * s2[j] + 0.25 * s3[j]
    v = normalize(v)
    t = Tex.grad(v, [(0.0, (16, 12, 10)), (0.4, (40, 30, 24)), (0.75, (72, 54, 42)), (1.0, (120, 94, 72))])
    t.shade(fbm(4, 2, 3, 2505), 0.85, 1.1)
    return t.image()


# =============================================================================
# Stone, marble, paper, sky
# =============================================================================
def gen_marble():
    turb = fbm(4, 4, 6, 2601, 0.55)
    turb2 = fbm(8, 8, 5, 2602, 0.5)
    v = [0.0] * N
    v2 = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        s = math.sin(TAU * (x + 2 * y) / S + turb[i] * 9.0)
        v[i] = 1.0 - sstep(0.0, 0.12, abs(s))
        s2 = math.sin(TAU * 3 * (x - y) / S + turb2[i] * 12.0)
        v2[i] = 1.0 - sstep(0.0, 0.05, abs(s2))
    t = Tex.solid((222, 219, 212))
    t.lerp((196, 194, 190), fbm(5, 5, 5, 2603), 0.7)
    t.lerp((132, 132, 138), [a * (0.4 + 0.6 * b) for a, b in zip(v, fbm(6, 6, 3, 2604))], 0.75)
    t.lerp((150, 150, 156), v2, 0.4)
    t.lerp((214, 200, 168), fbm(3, 3, 4, 2605), 0.25)
    t.lerp((100, 96, 90), speckles(2606, 120, 0.4, 1.2), 0.3)
    return grime(t, 2607, 0.86, 1.03).image()


def gen_paper():
    z = 2
    rnd = random.Random(2701)
    ink = Image.new('L', (S * z, S * z), 0)
    d = ImageDraw.Draw(ink)
    y = 52
    while y < S - 30:
        x = 40 + rnd.uniform(0, 30 if y > 60 else 90)
        x_end = S - 40 - rnd.uniform(0, 60)
        if rnd.random() < 0.12:
            x_end = x + rnd.uniform(60, 200)
        while x < x_end:
            wl = rnd.uniform(14, 52)
            pts = []
            t = 0.0
            ph = rnd.uniform(0, TAU)
            while t < wl:
                xx = x + t + 2.2 * math.cos(t * 1.1 + ph)
                yy = y - 3.2 * abs(math.sin(t * 0.55 + ph)) - (5.5 if rnd.random() < 0.03 else 0.0)
                pts.append((xx * z, yy * z))
                t += 0.7
            if len(pts) > 1:
                d.line(pts, fill=rnd.randint(150, 230), width=z)
            x += wl + rnd.uniform(6, 12)
        y += 22
    inkm = img_field(ink.resize((S, S), Image.LANCZOS))
    t = Tex.solid((216, 198, 158))
    t.lerp((190, 164, 118), fbm(4, 4, 5, 2702), 0.6)
    t.shade(fbm(64, 64, 3, 2703), 0.94, 1.04)
    t.shade(noise(S, 40, 2704), 0.97, 1.02)                 # laid-paper fibres
    rule = [0.0] * N
    edge = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        if y % 22 == 8 and 30 < x < S - 30:
            rule[i] = 1.0
        e = min(x, y, S - 1 - x, S - 1 - y)
        edge[i] = 1.0 - sstep(0.0, 70.0, e)
    t.lerp((150, 150, 160), rule, 0.15)
    t.lerp((86, 58, 40), [m * (0.55 + 0.45 * f) for m, f in zip(inkm, fbm(3, 3, 3, 2705))], 0.85)
    stain(t, 2706, 0.12, (186, 150, 96), (140, 100, 60), cells=2, k=0.4, ring_k=0.5)
    t.lerp((150, 108, 66), speckles(2707, 90, 0.8, 3.0, 0.8), 0.45)      # foxing
    t.lerp((150, 122, 82), edge, 0.5)
    crease = [0.0] * N
    for i in range(N):
        x = i & 511
        crease[i] = max(0.0, 1.0 - abs(x - 300) / 2.0)
    t.lerp((170, 150, 115), crease, 0.35)
    return grime(t, 2708, 0.84, 1.03).image()


def gen_night_sky():
    rnd = random.Random(2801)
    v = [0.0] * N
    for i in range(N):
        v[i] = (i >> 9) / (S - 1)
    t = Tex.grad(v, [(0.0, (20, 28, 64)), (0.6, (34, 46, 88)), (1.0, (56, 70, 112))])
    # stars
    st = Image.new('L', (S, S), 0)
    d = ImageDraw.Draw(st)
    MX, MY, MR = 362.0, 140.0, 44.0
    for _ in range(320):
        x, y = rnd.uniform(0, S), rnd.uniform(0, S * 0.92)
        if math.hypot(x - MX, y - MY) < MR * 2.2:
            continue
        b = rnd.random() ** 3
        r = 0.5 + 1.2 * b
        d.ellipse(ellipse_bbox(x, y, r), fill=int(120 + 135 * b))
        if b > 0.6:
            d.line([(x - 4, y), (x + 4, y)], fill=int(90 * b))
            d.line([(x, y - 4), (x, y + 4)], fill=int(90 * b))
    t.lerp((236, 238, 255), img_field(st.filter(ImageFilter.GaussianBlur(0.5))), 1.0)
    # moon glow + disc with maria
    glow = [0.0] * N
    disc = [0.0] * N
    maria = fbm(10, 10, 4, 2802)
    limb = [0.0] * N
    for i in range(N):
        x = (i & 511) - MX
        y = (i >> 9) - MY
        r = math.sqrt(x * x + y * y)
        glow[i] = math.exp(-max(0.0, r - MR) / 60.0) * 0.55 + math.exp(-max(0.0, r - MR) / 14.0) * 0.35
        disc[i] = 1.0 - sstep(MR - 1.2, MR + 0.6, r)
        limb[i] = math.sqrt(max(0.0, 1.0 - (r / MR) ** 2))
    t.add((70, 80, 100), glow)
    moon = Tex.solid((240, 236, 218))
    moon.lerp((172, 172, 168), [sstep(0.5, 0.75, m) for m in maria], 0.7)
    moon.shade(limb, 0.72, 1.0)
    t.lerp(moon, disc)
    # clouds: lit by the moon near it
    cl = uniform(fbm(3, 5, 6, 2803))
    cm = [sstep(0.55, 0.85, c) for c in cl]
    cloud = Tex.solid((58, 66, 96)).lerp((168, 170, 186), [clamp(g * 1.3) for g in glow])
    cloud.shade(fbm(12, 12, 3, 2804), 0.85, 1.1)
    t.lerp(cloud, cm, 0.75)
    return t.image()


# =============================================================================
# Pictures / props
# =============================================================================
def gen_book_spines():
    rnd = random.Random(2901)
    img = Image.new('RGB', (S, S), (26, 19, 14))
    shade = Image.new('L', (S, S), 120)
    d, ds = ImageDraw.Draw(img), ImageDraw.Draw(shade)
    gold = (198, 160, 82)
    pal = [(128, 40, 36), (92, 30, 28), (56, 80, 58), (44, 64, 54), (48, 56, 96), (40, 44, 70), (128, 88, 52),
           (104, 70, 44), (160, 130, 88), (72, 52, 40), (44, 38, 36), (116, 108, 72), (140, 76, 46), (82, 56, 76),
           (150, 140, 118)]
    RH = 128
    top_mask = Image.new('L', (S, S), 0)
    dt = ImageDraw.Draw(top_mask)
    for row in range(4):
        y0, y1 = row * RH, row * RH + RH
        widths = partition(S, 14, 40, rnd)
        x = 0
        for w in widths:
            hf = rnd.uniform(0.72, 0.96)
            top = int(y1 - 3 - (RH - 8) * hf)
            bot = y1 - 4
            col = pal[rnd.randrange(len(pal))]
            br = rnd.uniform(0.82, 1.15)
            if rnd.random() < 0.3:                              # sun-faded
                col = tuple(int(c * 0.6 + 70) for c in col)
            col = tuple(int(clamp(c * br, 0, 255)) for c in col)
            d.rectangle([x, top, x + w - 1, bot], fill=col)
            for k in range(w):
                u = (k + 0.5) / w
                s = 0.5 + 0.5 * math.sin(math.pi * u) ** 0.5 + 0.18 * math.exp(-((u - 0.36) / 0.13) ** 2)
                ds.line([(x + k, top), (x + k, bot)], fill=int(clamp(s * 120, 0, 255)))
            dt.line([(x + 1, top), (x + w - 2, top)], fill=255)
            h = bot - top
            if rnd.random() < 0.75:
                for fy in (0.07, 0.11, 0.89, 0.93):
                    yy = top + int(h * fy)
                    d.line([(x + 1, yy), (x + w - 2, yy)], fill=gold)
            if rnd.random() < 0.35 and w > 18:
                for fy in (0.25, 0.4, 0.55, 0.7):
                    yy = top + int(h * fy)
                    ds.line([(x, yy), (x + w - 1, yy)], fill=70)
                    ds.line([(x, yy + 1), (x + w - 1, yy + 1)], fill=170)
            if rnd.random() < 0.55:
                ly0 = top + int(h * rnd.uniform(0.15, 0.22))
                ly1 = ly0 + int(h * rnd.uniform(0.12, 0.2))
                lab = (34, 24, 20) if rnd.random() < 0.6 else (110, 30, 28)
                d.rectangle([x + 2, ly0, x + w - 3, ly1], fill=lab, outline=gold)
                for yy in range(ly0 + 3, ly1 - 2, 4):
                    xa = x + 4 + rnd.randint(0, 2)
                    xb = x + w - 5 - rnd.randint(0, 3)
                    if xb > xa:
                        d.line([(xa, yy), (xb, yy)], fill=(176, 142, 74))
            else:
                cx_ = x + w // 2
                for yy in range(top + int(h * 0.25), top + int(h * 0.62), 3):
                    if rnd.random() < 0.8:
                        d.line([(cx_, yy), (cx_, yy + 1)], fill=gold)
            x += w
        ds.rectangle([0, y1 - 4, S, y1 - 1], fill=40)
    t = Tex.from_image(img)
    t.mul([v / 120.0 for v in px(shade)])
    t.shade(fbm(48, 48, 3, 2902), 0.88, 1.08)                  # leather grain
    t.shade(noise(S, S, 2903), 0.94, 1.04)
    t.lerp((140, 128, 112), img_field(top_mask), 0.45)         # dust on the top edges
    shelf = [0.0] * N
    for i in range(N):
        ly = (i >> 9) % RH
        shelf[i] = 1.0 - 0.5 * (1.0 - sstep(0.0, 26.0, ly))
    t.mul(shelf)
    t.lerp((150, 136, 116), scratches(2904, 60, 3, 10, 1, 0.3), 0.3)
    return grime(t, 2905, 0.82, 1.05).image()


CONTINENTS = [
    [(-165, 63), (-166, 68), (-156, 71), (-141, 70), (-128, 70), (-110, 68), (-95, 68), (-85, 70), (-80, 63),
     (-94, 59), (-92, 57), (-82, 53), (-79, 56), (-76, 62), (-64, 60), (-56, 52), (-60, 47), (-66, 44), (-70, 42),
     (-74, 40), (-76, 35), (-81, 31), (-80, 25), (-82, 27), (-84, 30), (-89, 30), (-94, 29), (-97, 26), (-97, 21),
     (-91, 19), (-87, 21), (-88, 16), (-84, 15), (-83, 10), (-79, 9), (-77, 8), (-81, 8), (-86, 12), (-92, 15),
     (-105, 20), (-110, 24), (-112, 29), (-114, 31), (-110, 23), (-115, 28), (-117, 33), (-121, 35), (-124, 40),
     (-124, 47), (-123, 49), (-130, 54), (-135, 58), (-140, 60), (-148, 60), (-152, 58), (-158, 57), (-163, 55),
     (-158, 58), (-162, 60)],
    [(-73, 78), (-60, 82), (-30, 83), (-20, 80), (-20, 72), (-25, 70), (-40, 65), (-44, 60), (-50, 64), (-54, 68),
     (-58, 75)],
    [(-80, 9), (-77, 8), (-72, 12), (-62, 11), (-52, 5), (-50, 0), (-44, -2), (-35, -6), (-38, -13), (-40, -22),
     (-48, -26), (-53, -34), (-58, -38), (-65, -41), (-66, -47), (-69, -52), (-75, -50), (-74, -42), (-72, -30),
     (-71, -18), (-76, -14), (-81, -6), (-80, -1), (-77, 4)],
    [(-9, 37), (-9, 43), (-1, 44), (-4, 48), (0, 49), (4, 52), (8, 54), (8, 57), (11, 56), (14, 55), (21, 55),
     (21, 58), (24, 60), (29, 60), (25, 65), (21, 65), (17, 62), (18, 59), (16, 56), (13, 55), (12, 57), (10, 59),
     (5, 59), (5, 62), (14, 68), (20, 70), (28, 71), (40, 68), (44, 66), (54, 68), (60, 69), (68, 73), (80, 73),
     (90, 76), (105, 78), (115, 74), (130, 72), (140, 72), (160, 70), (170, 70), (180, 67), (180, 65), (178, 62),
     (163, 60), (156, 51), (155, 59), (143, 59), (138, 55), (141, 48), (135, 43), (130, 42), (129, 36), (126, 35),
     (126, 38), (122, 40), (119, 38), (122, 32), (121, 28), (117, 24), (110, 21), (108, 17), (109, 12), (105, 9),
     (103, 10), (100, 13), (99, 9), (104, 1), (100, 6), (98, 15), (94, 17), (92, 22), (87, 21), (80, 15), (77, 8),
     (73, 17), (72, 21), (67, 25), (57, 26), (59, 22), (52, 16), (43, 12), (39, 21), (35, 28), (35, 33), (36, 36),
     (30, 36), (27, 37), (26, 40), (23, 40), (23, 37), (21, 38), (19, 42), (13, 45), (12, 44), (16, 41), (18, 40),
     (16, 38), (15, 38), (12, 42), (10, 44), (7, 44), (3, 43), (0, 39), (-2, 37), (-6, 36)],
    [(-17, 21), (-17, 15), (-15, 11), (-11, 7), (-8, 4), (-2, 5), (4, 6), (9, 4), (9, -1), (13, -6), (12, -14),
     (15, -26), (18, -33), (20, -35), (26, -34), (32, -29), (35, -24), (40, -15), (40, -10), (39, -5), (43, 0),
     (51, 11), (44, 11), (39, 17), (34, 27), (32, 31), (25, 32), (20, 31), (11, 33), (10, 37), (3, 37), (-6, 36),
     (-10, 30), (-13, 27)],
    [(44, -25), (47, -25), (50, -15), (49, -12), (44, -17)],
    [(-5, 50), (1, 51), (2, 53), (-2, 56), (-3, 58), (-5, 58), (-6, 56), (-3, 54), (-5, 52)],
    [(-10, 52), (-6, 52), (-6, 55), (-8, 55)],
    [(-24, 64), (-14, 64), (-13, 66), (-23, 66.5)],
    [(130, 31), (132, 34), (136, 34), (140, 36), (142, 40), (141, 45), (144, 44), (141, 41), (140, 38), (139, 35),
     (135, 33), (131, 33)],
    [(113, -22), (114, -26), (115, -34), (118, -35), (124, -34), (131, -31), (135, -34), (138, -35), (140, -38),
     (147, -38), (150, -37), (153, -31), (153, -25), (146, -19), (142, -11), (141, -17), (136, -12), (130, -12),
     (126, -14), (122, -17)],
    [(172, -34), (175, -37), (178, -38), (175, -41), (171, -44), (167, -46), (170, -42), (173, -40)],
    [(109, 1), (111, -3), (116, -4), (119, 1), (117, 7), (113, 3)],
    [(95, 5), (98, 4), (104, -2), (106, -6), (101, -3)],
    [(131, -1), (141, -3), (150, -10), (143, -9), (138, -8)],
    [(106, -6), (114, -7), (114, -8), (106, -7)],
    [(120, 18), (122, 18), (126, 8), (124, 6), (121, 12)],
    [(80, 6), (82, 8), (80, 10), (79, 8)],
    [(15, 78), (24, 80), (28, 78), (18, 76)],
    [(-180, -78), (-150, -76), (-120, -73), (-90, -72), (-70, -66), (-60, -63), (-55, -66), (-40, -78), (-20, -72),
     (0, -70), (30, -69), (60, -67), (90, -66), (120, -66), (150, -68), (170, -72), (180, -78), (180, -90),
     (-180, -90)],
]
LAKES = [
    [(28, 42), (28, 46), (33, 46), (38, 47), (41, 42), (36, 41)],
    [(47, 45), (50, 47), (54, 45), (53, 40), (54, 37), (50, 37), (49, 40)],
]


def gen_world_map():
    """Equirectangular antique map drawn on a 2:1 design, squeezed to 512x512 for the globe."""
    DW, DH = 2048, 1024

    def P(lon, lat):
        return ((lon + 180.0) / 360.0 * DW, (90.0 - lat) / 180.0 * DH)

    land = Image.new('L', (DW, DH), 0)
    dl = ImageDraw.Draw(land)
    for poly in CONTINENTS:
        dl.polygon([P(a, b) for a, b in poly], fill=255)
    for poly in LAKES:
        dl.polygon([P(a, b) for a, b in poly], fill=0)
    lines = Image.new('L', (DW, DH), 0)
    dg = ImageDraw.Draw(lines)
    for lon in range(-180, 181, 15):
        x = P(lon, 0)[0]
        dg.line([(x, 0), (x, DH)], fill=150, width=2)
    for lat in range(-75, 76, 15):
        y = P(0, lat)[1]
        dg.line([(0, y), (DW, y)], fill=150, width=2)
    red = Image.new('L', (DW, DH), 0)
    dr = ImageDraw.Draw(red)
    for lat in (23.44, -23.44, 66.56, -66.56, 0.0):
        y = P(0, lat)[1]
        for x in range(0, DW, 24):
            dr.line([(x, y), (x + 14, y)], fill=230, width=3 if lat == 0 else 2)
    # compass rose in the south Pacific
    cx, cy = P(-130, -32)
    for k in range(16):
        a = TAU * k / 16
        r = 90 if k % 4 == 0 else 60 if k % 2 == 0 else 38
        tip = (cx + r * math.cos(a), cy + r * math.sin(a))
        l1 = (cx + 10 * math.cos(a + 1.2), cy + 10 * math.sin(a + 1.2))
        l2 = (cx + 10 * math.cos(a - 1.2), cy + 10 * math.sin(a - 1.2))
        dg.polygon([l1, tip, l2, (cx, cy)], outline=220, fill=110 if k % 2 else 200)
    dg.ellipse(ellipse_bbox(cx, cy, 70), outline=200, width=2)
    dg.ellipse(ellipse_bbox(cx, cy, 64), outline=150, width=1)
    # labels (old-style capitals); skipped if no scalable font is available
    try:
        font = ImageFont.load_default(size=30)
        small = ImageFont.load_default(size=22)
        for text, lon, lat, f in (('MARE PACIFICUM', -150, 8, font), ('OCEANUS ATLANTICUS', -42, 22, small),
                                  ('OCEANUS INDICUS', 75, -25, small), ('AFRICA', 18, 8, font), ('ASIA', 95, 50, font),
                                  ('EUROPA', 12, 50, small), ('AMERICA', -102, 42, font), ('TERRA AUSTRALIS', 20, -78, small)):
            x, y = P(lon, lat)
            dg.text((x, y), text, fill=210, font=f, anchor='mm')
    except (TypeError, OSError):
        pass
    sq = (S, S)
    landm = img_field(land.resize(sq, Image.LANCZOS))
    gridm = img_field(lines.resize(sq, Image.LANCZOS))
    redm = img_field(red.resize(sq, Image.LANCZOS))

    def hblur(f, r):                       # wraps east-west only (the globe seam)
        im = field_img(f)
        big = Image.new('L', (S * 3, S))
        for k in range(3):
            big.paste(im, (k * S, 0))
        return img_field(big.filter(ImageFilter.GaussianBlur(r)).crop((S, 0, 2 * S, S)))

    coastn = fbm(16, 16, 4, 3001)
    lb = hblur(landm, 1.4)
    lm = [sstep(0.46, 0.54, a + (n - 0.5) * 0.4) for a, n in zip(lb, coastn)]
    c1 = hblur(lm, 0.8)
    coast = [max(0.0, 1.0 - abs(v - 0.5) * 3.2) for v in c1]
    wide = hblur(lm, 5.0)
    ripple = [(max(0.0, 1.0 - abs(w - 0.14) * 45.0) + max(0.0, 1.0 - abs(w - 0.3) * 45.0)) * (1.0 - l)
              for w, l in zip(wide, lm)]
    t = Tex.solid((182, 156, 110))
    t.lerp((166, 138, 96), fbm(5, 5, 4, 3002), 0.6)
    landc = Tex.solid((214, 192, 142))
    landc.lerp((172, 160, 104), fbm(6, 6, 4, 3003), 0.55)
    landc.lerp((188, 146, 96), fbm(4, 4, 4, 3004), 0.35)
    landc.shade(wide, 0.9, 1.04)
    t.lerp(landc, lm)
    t.lerp((120, 90, 58), ripple, 0.35)
    t.lerp((140, 104, 66), gridm, 0.55)
    t.lerp((150, 60, 40), redm, 0.6)
    t.lerp((78, 50, 30), coast, 0.85)
    t.shade(fbm(64, 64, 3, 3005), 0.94, 1.05)
    stain(t, 3006, 0.12, (170, 130, 80), (120, 84, 50), cells=3, k=0.4, ring_k=0.45)
    t.lerp((130, 92, 56), speckles(3007, 120, 0.8, 2.5, 0.6), 0.4)
    gore = [0.0] * N
    for i in range(N):
        x = i & 511
        gore[i] = 1.0 if x % 43 == 0 else 0.0
    t.shade(gore, 1.0, 0.85)
    t.tint((1.0, 0.96, 0.88))
    return grime(t, 3008, 0.84, 1.04).image()


def gen_clock_face():
    z = 2
    W = S * z
    c = W / 2.0
    R = W * 0.495
    img = Image.new('RGB', (W, W), (34, 24, 16))
    d = ImageDraw.Draw(img)
    # corner filigree
    for (qx, qy) in ((0, 0), (W, 0), (0, W), (W, W)):
        sx, sy = (1 if qx == 0 else -1), (1 if qy == 0 else -1)
        for k in range(3):
            pts = spiral(qx + sx * (60 + 30 * k), qy + sy * (60 + 30 * k), 34 - 8 * k, 6, 1.3, 0.3 * k, 1, 50)
            d.line([(qx + (x - qx), qy + (y - qy)) for x, y in pts], fill=(120, 92, 48), width=4)
        d.line([(qx + sx * 20, qy + sy * 150), (qx + sx * 150, qy + sy * 20)], fill=(110, 84, 44), width=5)
    d.ellipse(ellipse_bbox(c, c, R), fill=(150, 116, 60))
    d.ellipse(ellipse_bbox(c, c, R * 0.975), fill=(196, 160, 90))
    d.ellipse(ellipse_bbox(c, c, R * 0.955), fill=(92, 68, 36))
    d.ellipse(ellipse_bbox(c, c, R * 0.945), fill=(226, 214, 182))
    ink = (34, 28, 26)
    for r in (0.905, 0.895, 0.86, 0.69):
        d.ellipse(ellipse_bbox(c, c, R * r), outline=ink, width=3 if r != 0.905 else 2)
    for m in range(60):
        a = TAU * m / 60
        r1, r2 = R * 0.86, R * 0.895
        dx, dy = math.sin(a), -math.cos(a)
        d.line([(c + dx * r1, c + dy * r1), (c + dx * r2, c + dy * r2)], fill=ink, width=7 if m % 5 == 0 else 3)
        if m % 5 == 0:
            d.ellipse(ellipse_bbox(c + dx * R * 0.925, c + dy * R * 0.925, 6), fill=ink)

    # roman numerals from strokes, oriented radially (bases towards the centre)
    def glyph_strokes(ch):
        th, tn, sw = 0.16, 0.06, 0.34      # thick stroke, thin stroke, serif width (em units)
        if ch == 'I':
            return 0.22, [((0, -0.5), (0, 0.5), th), ((-sw / 2, 0.5), (sw / 2, 0.5), tn), ((-sw / 2, -0.5), (sw / 2, -0.5), tn)]
        if ch == 'V':
            return 0.62, [((-0.26, 0.5), (0, -0.5), th), ((0.26, 0.5), (0, -0.5), tn),
                          ((-0.26 - 0.13, 0.5), (-0.26 + 0.13, 0.5), tn), ((0.26 - 0.12, 0.5), (0.26 + 0.12, 0.5), tn)]
        return 0.62, [((-0.24, 0.5), (0.24, -0.5), th), ((0.24, 0.5), (-0.24, -0.5), tn),
                      ((-0.37, 0.5), (-0.11, 0.5), tn), ((0.11, 0.5), (0.37, 0.5), tn),
                      ((-0.37, -0.5), (-0.11, -0.5), tn), ((0.11, -0.5), (0.37, -0.5), tn)]

    names = ['XII', 'I', 'II', 'III', 'IIII', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI']
    em = R * 0.135
    for h, name in enumerate(names):
        a = TAU * h / 12
        up = (math.sin(a), -math.cos(a))
        right = (math.cos(a), math.sin(a))
        cen = (c + up[0] * R * 0.775, c + up[1] * R * 0.775)
        gl = [glyph_strokes(ch) for ch in name]
        total = sum(g[0] for g in gl) + 0.08 * (len(gl) - 1)
        x = -total / 2
        for wdt, strokes in gl:
            gx = x + wdt / 2
            for (p0, p1, th) in strokes:
                ax, ay = gx + p0[0], p0[1]
                bx, by = gx + p1[0], p1[1]
                ln = math.hypot(bx - ax, by - ay) or 1.0
                nx, ny = -(by - ay) / ln * th / 2, (bx - ax) / ln * th / 2
                quad = [(ax + nx, ay + ny), (bx + nx, by + ny), (bx - nx, by - ny), (ax - nx, ay - ny)]
                d.polygon([(cen[0] + (qx * right[0] + qy * up[0]) * em, cen[1] + (qx * right[1] + qy * up[1]) * em)
                           for qx, qy in quad], fill=ink)
            x += wdt + 0.08
    try:
        font = ImageFont.load_default(size=26)
        d.text((c, c - R * 0.36), 'J. HARGRAVE', fill=(70, 56, 46), font=font, anchor='mm')
        d.text((c, c - R * 0.29), 'LONDON', fill=(70, 56, 46), font=ImageFont.load_default(size=20), anchor='mm')
    except (TypeError, OSError):
        pass

    def hand(angle, pts, col):
        up = (math.sin(angle), -math.cos(angle))
        right = (math.cos(angle), math.sin(angle))
        return d.polygon([(c + (sx_ * right[0] + sy_ * up[0]) * R, c + (sx_ * right[1] + sy_ * up[1]) * R) for sx_, sy_ in pts],
                         fill=col)

    hcol = (22, 20, 24)
    ha = TAU * (11 + 50 / 60) / 12
    ma = TAU * 50 / 60
    hand(ha, [(-0.03, -0.14), (-0.018, 0.0), (-0.014, 0.33), (-0.06, 0.37), (-0.055, 0.42), (0.0, 0.5),
              (0.055, 0.42), (0.06, 0.37), (0.014, 0.33), (0.018, 0.0), (0.03, -0.14)], hcol)
    hand(ma, [(-0.022, -0.18), (-0.011, 0.0), (-0.008, 0.6), (-0.03, 0.66), (0.0, 0.8), (0.03, 0.66), (0.008, 0.6),
              (0.011, 0.0), (0.022, -0.18)], hcol)
    for ang, rr in ((ha, 0.39), (ma, 0.66)):
        hx, hy = c + math.sin(ang) * R * rr, c - math.cos(ang) * R * rr
        d.ellipse(ellipse_bbox(hx, hy, R * 0.016), fill=(226, 214, 182))
    d.ellipse(ellipse_bbox(c, c, R * 0.045), fill=(170, 130, 64))
    d.ellipse(ellipse_bbox(c, c, R * 0.02), fill=(60, 46, 26))
    t = Tex.from_image(img.resize((S, S), Image.LANCZOS))
    # ageing: rim darkening, crazing, stains, a hairline crack
    rad = [0.0] * N
    for i in range(N):
        x = (i & 511) - 255.5
        y = (i >> 9) - 255.5
        rad[i] = math.hypot(x, y) / 255.0
    dial = [1.0 - sstep(0.93, 0.945, r) for r in rad]
    t.shade([sstep(0.5, 0.94, r) * dm for r, dm in zip(rad, dial)], 1.0, 0.8)
    f1, f2, _ = voronoi(12, 12, 3101, 0.95)
    t.lerp((140, 120, 90), [(1.0 - sstep(0.3, 1.1, b - a)) * dm for a, b, dm in zip(f1, f2, dial)], 0.35)
    blot = uniform(fbm(3, 3, 5, 3102))
    t.lerp((176, 150, 104), [sstep(0.8, 0.84, v) * dm for v, dm in zip(blot, dial)], 0.35)
    t.lerp((130, 104, 70), [max(0.0, 1.0 - abs(v - 0.8) * 200) * dm for v, dm in zip(blot, dial)], 0.4)
    crk = Image.new('L', (S, S), 0)
    dk = ImageDraw.Draw(crk)
    rnd = random.Random(3103)
    x, y, a = 256 + 230 * math.cos(2.2), 256 + 230 * math.sin(2.2), 2.2 + math.pi
    pts = [(x, y)]
    for _ in range(70):
        a += rnd.uniform(-0.3, 0.3)
        x += math.cos(a) * 2.2
        y += math.sin(a) * 2.2
        pts.append((x, y))
    dk.line(pts, fill=220, width=1)
    t.lerp((60, 46, 36), img_field(crk), 0.7)
    t.lerp((120, 110, 96), speckles(3104, 160, 0.5, 1.3), 0.3)
    return grime(t, 3105, 0.84, 1.03).image()


def gen_portrait():
    """Dark oil portrait of a pale Victorian figure. Painted on a 1.3:1 tall canvas, then squeezed
    to 512x512 so it un-squeezes inside the tall frame."""
    W, H = 1024, 1332
    rnd = random.Random(3201)
    img = Image.new('RGB', (W, H), (16, 13, 10))
    # background: warm halo behind the head fading to black
    halo = Image.radial_gradient('L').resize((1500, 1500)).crop((750 - 512, 750 - 560, 750 + 512, 750 + 772))
    bg = Image.composite(Image.new('RGB', (W, H), (14, 11, 9)), Image.new('RGB', (W, H), (88, 72, 50)), halo)
    img.paste(bg)

    def layer(draw_fn, col, blur, opacity=1.0, base=None, mask_only=False):
        m = Image.new('L', (W, H), 0)
        draw_fn(ImageDraw.Draw(m))
        if blur:
            m = m.filter(ImageFilter.GaussianBlur(blur))
        if opacity < 1.0:
            m = m.point(lambda v: int(v * opacity))
        if mask_only:
            return m
        tgt = base if base is not None else img
        tgt.paste(Image.composite(Image.new('RGB', (W, H), col), tgt, m))
        return m

    # loose brush strokes in the background
    for _ in range(220):
        x, y = rnd.uniform(0, W), rnd.uniform(0, H)
        a = rnd.uniform(-0.6, 0.6) + math.pi / 2 * rnd.choice((0, 1))
        ln = rnd.uniform(40, 160)
        v = rnd.uniform(-1, 1)
        col = (int(50 + 30 * v), int(40 + 24 * v), int(28 + 16 * v))
        layer(lambda dd: dd.line([(x, y), (x + math.cos(a) * ln, y + math.sin(a) * ln)], fill=255,
                                 width=int(rnd.uniform(8, 22))), col, 4, 0.18)
    # coat & shoulders
    body = [(40, H), (70, 1060), (160, 940), (320, 860), (430, 800), (594, 800), (704, 860), (864, 940), (954, 1060),
            (984, H)]
    layer(lambda dd: dd.polygon(body, fill=255), (24, 21, 24), 2)
    layer(lambda dd: dd.line([(120, 1000), (300, 880), (420, 815)], fill=255, width=26), (58, 54, 58), 18, 0.7)
    layer(lambda dd: dd.line([(904, 1000), (724, 880), (604, 815)], fill=255, width=20), (44, 40, 44), 18, 0.5)
    layer(lambda dd: dd.line([(428, 820), (476, 1160)], fill=255, width=6), (8, 7, 8), 2)
    layer(lambda dd: dd.line([(596, 820), (548, 1160)], fill=255, width=6), (8, 7, 8), 2)
    layer(lambda dd: dd.line([(440, 830), (484, 1150)], fill=255, width=4), (50, 46, 50), 3, 0.6)
    # neck
    layer(lambda dd: dd.polygon([(452, 640), (572, 640), (580, 810), (444, 810)], fill=255), (150, 146, 130), 4)
    layer(lambda dd: dd.ellipse(ellipse_bbox(512, 690, 120, 60), fill=255), (80, 74, 64), 24, 0.8)
    # shirt front, collar wings, cravat
    layer(lambda dd: dd.polygon([(440, 780), (584, 780), (560, 960), (512, 1060), (464, 960)], fill=255), (170, 164, 148), 3)
    layer(lambda dd: dd.polygon([(446, 760), (512, 812), (468, 860), (430, 800)], fill=255), (206, 200, 184), 2)
    layer(lambda dd: dd.polygon([(578, 760), (512, 812), (556, 860), (594, 800)], fill=255), (196, 190, 174), 2)
    layer(lambda dd: dd.ellipse(ellipse_bbox(512, 846, 38, 26), fill=255), (30, 24, 30), 2)
    layer(lambda dd: dd.polygon([(490, 860), (534, 860), (548, 990), (512, 1010), (476, 990)], fill=255), (26, 22, 28), 2)
    layer(lambda dd: dd.line([(500, 870), (506, 990)], fill=255, width=6), (70, 64, 74), 4, 0.6)
    # ears (under the hair)
    for ex in (360, 664):
        layer(lambda dd: dd.ellipse(ellipse_bbox(ex, 530, 22, 44), fill=255), (150, 140, 124), 3)
    # face on its own layer so shading stays inside the head
    face = Image.new('RGB', (W, H), (204, 200, 182))
    fm = layer(lambda dd: dd.ellipse(ellipse_bbox(512, 520, 150, 200), fill=255), None, 3, mask_only=True)
    layer(lambda dd: dd.ellipse(ellipse_bbox(640, 540, 90, 220), fill=255), (118, 112, 98), 60, 0.85, base=face)     # shadow side
    layer(lambda dd: dd.ellipse(ellipse_bbox(512, 720, 140, 60), fill=255), (130, 122, 108), 40, 0.7, base=face)     # jaw
    for cx in (430, 600):
        layer(lambda dd: dd.ellipse(ellipse_bbox(cx, 610, 38, 62), fill=255), (140, 134, 116), 26, 0.6, base=face)  # hollow cheeks
    layer(lambda dd: dd.ellipse(ellipse_bbox(482, 380, 80, 46), fill=255), (232, 230, 214), 34, 0.55, base=face)    # forehead light
    layer(lambda dd: dd.ellipse(ellipse_bbox(430, 545, 34, 24), fill=255), (226, 222, 206), 18, 0.45, base=face)    # cheekbone light
    for cx in (452, 574):
        layer(lambda dd: dd.ellipse(ellipse_bbox(cx, 478, 54, 36), fill=255), (58, 48, 44), 16, 0.92, base=face)    # sockets
        layer(lambda dd: dd.ellipse(ellipse_bbox(cx, 505, 46, 18), fill=255), (90, 72, 70), 14, 0.6, base=face)     # dark rings
        layer(lambda dd: dd.ellipse(ellipse_bbox(cx, 476, 28, 17), fill=255), (5, 4, 4), 4, 1.0, base=face)         # hollow eyes
        layer(lambda dd: dd.ellipse(ellipse_bbox(cx - 6, 471, 3, 3), fill=255), (190, 186, 170), 1.5, 0.35, base=face)
        layer(lambda dd: dd.line([(cx - 2, 500), (cx - 5, 560), (cx - 3, 620)], fill=255, width=5), (70, 40, 40), 3, 0.35,
              base=face)                                                                                   # tear stains
        layer(lambda dd: dd.arc(ellipse_bbox(cx, 452, 50, 22), 200, 340, fill=255, width=9), (40, 32, 28), 3, 0.9, base=face)
    layer(lambda dd: dd.line([(522, 490), (532, 590), (520, 612)], fill=255, width=12), (120, 112, 98), 8, 0.8, base=face)
    layer(lambda dd: dd.line([(500, 500), (496, 590)], fill=255, width=10), (226, 222, 208), 8, 0.4, base=face)
    for nx in (496, 530):
        layer(lambda dd: dd.ellipse(ellipse_bbox(nx, 614, 9, 5), fill=255), (60, 46, 42), 3, 0.85, base=face)
    layer(lambda dd: dd.line([(466, 684), (490, 678), (512, 680), (536, 678), (558, 688)], fill=255, width=6),
          (48, 30, 32), 2.5, 0.95, base=face)
    layer(lambda dd: dd.ellipse(ellipse_bbox(512, 698, 30, 8), fill=255), (172, 162, 146), 6, 0.5, base=face)
    layer(lambda dd: dd.ellipse(ellipse_bbox(512, 740, 60, 22), fill=255), (110, 102, 90), 16, 0.5, base=face)
    img.paste(Image.composite(face, img, fm))
    # hair: dark mass with a side parting, face cut out below the hairline
    hair = Image.new('L', (W, H), 0)
    dh = ImageDraw.Draw(hair)
    dh.ellipse(ellipse_bbox(512, 488, 168, 196), fill=255)
    cut = Image.new('L', (W, H), 0)
    dc = ImageDraw.Draw(cut)
    dc.ellipse(ellipse_bbox(512, 540, 146, 196), fill=255)
    dc.polygon([(300, 0), (724, 0), (724, 380), (620, 360), (540, 350), (470, 372), (400, 400), (300, 440)], fill=0)
    hair = ImageChops.subtract(hair, cut)
    dh = ImageDraw.Draw(hair)
    for sx_ in (360, 664):
        dh.polygon([(sx_ - 14, 470), (sx_ + 14, 470), (sx_ + 10, 580), (sx_ - 8, 580)], fill=255)
    hair = hair.filter(ImageFilter.GaussianBlur(3))
    img.paste(Image.composite(Image.new('RGB', (W, H), (22, 17, 15)), img, hair))
    for k in range(18):
        x0 = 380 + k * 15
        layer(lambda dd: dd.line([(540, 352), (x0, 300 + abs(x0 - 540) * 0.2), (x0 - 40, 420)], fill=255, width=3),
              (60, 48, 40), 2, 0.45)
    t = Tex.from_image(img.resize((S, S), Image.LANCZOS))
    # oil paint: brush texture, canvas weave, craquelure, yellowed varnish, vignette
    t.shade(fbm(24, 12, 4, 3202), 0.9, 1.08)
    weave = [0.0] * N
    vig = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        weave[i] = 0.5 + 0.5 * math.cos(TAU * x / 4.0) * math.cos(TAU * y / 4.0)
        dx, dy = (x - 255.5) / 256.0, (y - 255.5) / 256.0
        vig[i] = sstep(0.55, 1.35, math.sqrt(dx * dx + dy * dy))
    t.shade(weave, 0.95, 1.04)
    f1, f2, _ = voronoi(22, 22, 3203, 0.95)
    crq = [1.0 - sstep(0.4, 1.2, b - a) for a, b in zip(f1, f2)]
    t.lerp((20, 16, 12), crq, 0.4)
    t.lerp((40, 30, 20), cracks(3204, 4, 80, 2.4, 0.4, 0.04), 0.5)
    t.tint((1.0, 0.94, 0.8))
    t.shade(vig, 1.0, 0.45)
    t.lerp((150, 130, 96), speckles(3205, 80, 0.5, 1.2), 0.25)
    return grime(t, 3206, 0.82, 1.04).image()


# =============================================================================
# Quest extension (v2): map_parchment, porcelain_cracked
# =============================================================================
def load_font(size, names):
    """shortcut: prefers fonts from the Windows font folder, else Pillow's default font, else None
    (callers then skip the lettering); upgrade by bundling a font if the generator must run elsewhere."""
    fdir = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
    for n in names:
        try:
            return ImageFont.truetype(os.path.join(fdir, n), size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):
        return None


def paint(img, draw_fn, col, blur=0.0, opacity=1.0):
    """Composite a solid colour onto img through a mask drawn by draw_fn (soft when blurred)."""
    m = Image.new('L', img.size, 0)
    draw_fn(ImageDraw.Draw(m))
    if blur:
        m = m.filter(ImageFilter.GaussianBlur(blur))
    if opacity < 1.0:
        m = m.point(lambda v: int(v * opacity))
    img.paste(Image.composite(Image.new('RGB', img.size, col), img, m))
    return m


def wobble(pts, rnd, amp, step=4.0):
    """Resample a polyline with a smooth perpendicular hand tremor."""
    out = []
    ph1, ph2 = rnd.uniform(0, TAU), rnd.uniform(0, TAU)
    s = 0.0
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        ln = math.hypot(bx - ax, by - ay)
        if ln == 0.0:
            continue
        nx, ny = -(by - ay) / ln, (bx - ax) / ln
        n = max(1, int(ln / step))
        for k in range(n):
            t = k / n
            u = s + t * ln
            o = amp * (math.sin(u / 23.0 + ph1) + 0.5 * math.sin(u / 7.0 + ph2))
            out.append((ax + (bx - ax) * t + nx * o, ay + (by - ay) * t + ny * o))
        s += ln
    out.append(pts[-1])
    return out


def jagged_poly(cx, cy, rx, ry, rnd, amp, n=48):
    ph = [rnd.uniform(0, TAU) for _ in range(3)]
    pts = []
    for k in range(n):
        t = TAU * k / n
        r = 1.0 + amp * (0.5 * math.sin(3 * t + ph[0]) + 0.3 * math.sin(7 * t + ph[1]) + 0.2 * math.sin(13 * t + ph[2])
                         + 0.35 * rnd.uniform(-1, 1))
        pts.append((cx + rx * r * math.cos(t), cy + ry * r * math.sin(t)))
    return pts


# Where the red X sits on the inked plan (Nursery world X, Z): bed_key, half under the bed rail.
MAP_KEY_XZ = (-0.9, -3.7)  # must match bed_key in Scene.cpp


def gen_map_parchment():
    """Hand-inked plan of the Nursery on old parchment, drawn on a 1.4:1 design (the map prop is
    0.45 x 0.32 m) and squeezed to 512x512. Layout follows Scene.cpp buildBedroom: north (-Z, bed
    wall) up, east (+X) right. The curled ends of the prop take the outer margins, so the plan stays central."""
    z = 2
    DW, DH = 716, 512
    rnd = random.Random(3301)
    ink = Image.new('L', (DW * z, DH * z), 0)
    red = Image.new('L', (DW * z, DH * z), 0)
    di, dr = ImageDraw.Draw(ink), ImageDraw.Draw(red)
    K, OX, OY = 30.0, 282.0, 276.0

    def P(X, Zw):
        return (OX + X * K, OY + Zw * K)

    def line(pts, w=1.3, v=235, amp=0.7, d=None, closed=False):
        pts = list(pts) + ([pts[0]] if closed else [])
        (d or di).line(sc(wobble(pts, rnd, amp), z), fill=v, width=max(1, int(round(w * z))), joint='curve')

    def foot(X, Zw, w, dd, rot=0.0):
        a = math.radians(rot)
        ca, sa = math.cos(a), math.sin(a)
        return [P(X + lx * ca + lz * sa, Zw - lx * sa + lz * ca)
                for lx, lz in ((-w / 2, -dd / 2), (w / 2, -dd / 2), (w / 2, dd / 2), (-w / 2, dd / 2))]

    def arc(cx, cy, r, a0, a1, n=40):
        return [(cx + r * math.cos(a0 + (a1 - a0) * k / n), cy + r * math.sin(a0 + (a1 - a0) * k / n)) for k in range(n + 1)]

    # walls: double line with hatching; window gap on the west wall, door gap on the south wall
    T = 0.16
    walls = [((-5, -4.5), (5, -4.5), 'h'), ((5, -4.5), (5, 4.5), 'v'),
             ((-5, -4.5), (-5, -2.0), 'v'), ((-5, -0.4), (-5, 4.5), 'v'),
             ((-5, 4.5), (-0.7, 4.5), 'h'), ((0.7, 4.5), (5, 4.5), 'h')]
    for (x0, z0), (x1, z1), o in walls:
        if o == 'h':
            sgn = 1 if z0 < 0 else -1
            line([P(x0, z0), P(x1, z1)], 1.8)
            line([P(x0, z0 + sgn * T), P(x1, z1 + sgn * T)], 1.1)
            x = x0
            while x < x1 - 0.1:
                a, b = P(x, z0), P(x + 0.14, z0 + sgn * T)
                di.line(sc([a, b], z), fill=160, width=z)
                x += 0.22
        else:
            sgn = 1 if x0 < 0 else -1
            line([P(x0, z0), P(x1, z1)], 1.8)
            line([P(x0 + sgn * T, z0), P(x1 + sgn * T, z1)], 1.1)
            zz = z0
            while zz < z1 - 0.1:
                a, b = P(x0, zz), P(x0 + sgn * T, zz + 0.14)
                di.line(sc([a, b], z), fill=160, width=z)
                zz += 0.22
    for zz in (-2.0, -0.4):                                    # window jambs + glazing lines
        line([P(-5, zz), P(-5 + T, zz)], 1.4)
    for f in (0.3, 0.5, 0.7):
        line([P(-5 + T * f, -2.0), P(-5 + T * f, -0.4)], 0.8, 200, 0.3)
    for xx in (-0.7, 0.7):                                     # door jambs + swing
        line([P(xx, 4.5), P(xx, 4.5 - T)], 1.4)
    line([P(-0.7, 4.5 - T), P(-0.7, 4.5 - T - 1.3)], 1.0, 200)
    line(arc(*P(-0.7, 4.5 - T), 1.3 * K, -math.pi / 2, 0.0), 0.8, 150, 0.3)

    # furniture
    line(foot(0.0, -3.38, 1.8, 2.2), 1.4, closed=True)                                   # bed
    for px_, pz in ((-0.86, -4.44), (0.86, -4.44), (-0.86, -2.32), (0.86, -2.32)):
        x, y = P(px_, pz)
        di.ellipse(ellipse_bbox(x * z, y * z, 3.2 * z), outline=235, width=z)
    for px_ in (-0.42, 0.42):
        line(foot(px_, -4.12, 0.7, 0.38), 1.0, 220, closed=True)
    line([P(-0.9, -3.75), P(0.9, -3.7)], 1.0, 200)
    for k in range(1, 6):                                                                   # quilt patches
        line([P(-0.9 + k * 0.3, -3.7), P(-0.9 + k * 0.3, -2.28)], 0.7, 120, 0.4)
    for zz in (-3.25, -2.8):
        line([P(-0.9, zz), P(0.9, zz)], 0.7, 120, 0.4)
    line(foot(1.3, -4.27, 0.5, 0.42), 1.1, closed=True)                                   # nightstand + lamp
    x, y = P(1.3, -4.25)
    di.ellipse(ellipse_bbox(x * z, y * z, 3.0 * z), outline=235, width=z)
    for k in range(8):
        a = TAU * k / 8
        di.line(sc([(x + 4.5 * math.cos(a), y + 4.5 * math.sin(a)), (x + 7.5 * math.cos(a), y + 7.5 * math.sin(a))], z),
                fill=170, width=z)
    line(foot(0.0, -1.95, 0.9, 0.5), 1.2, closed=True)                                    # toy chest + doll
    for xx in (-0.3, 0.3):
        line([P(xx, -2.2), P(xx, -1.7)], 0.8, 190, 0.3)
    x, y = P(0.18, -1.95)
    di.ellipse(ellipse_bbox(x * z, y * z, 2.4 * z), fill=235)
    cx, cy = P(-3.4, -0.4)                                                                 # round rug
    line(arc(cx, cy, 1.2 * K, 0, TAU, 72), 1.0, 210, 0.5)
    line(arc(cx, cy, 0.95 * K, 0, TAU, 60), 0.7, 150, 0.4)
    for k in range(24):
        a = TAU * k / 24
        di.ellipse(ellipse_bbox((cx + 1.08 * K * math.cos(a)) * z, (cy + 1.08 * K * math.sin(a)) * z, 0.8 * z), fill=170)
    line(foot(-3.7, -0.6, 0.6, 0.55, 65.0), 1.1, closed=True)                             # rocking chair
    a = math.radians(65.0)
    for lx in (-0.3, 0.3):
        pts = []
        for k in range(9):
            lz = -0.5 + k / 8.0
            bow = 0.05 * (1 - (2 * lz) ** 2)
            xx, zz = lx + bow * (1 if lx > 0 else -1), lz
            pts.append(P(-3.7 + xx * math.cos(a) + zz * math.sin(a), -0.6 - xx * math.sin(a) + zz * math.cos(a)))
        line(pts, 1.0, 220, 0.3)
    line(foot(4.73, -2.2, 1.2, 0.5, -90.0), 1.3, closed=True)                             # dresser + music box
    for k in (1, 2):
        line([P(4.48 + 0.5 * k / 3, -2.8), P(4.48 + 0.5 * k / 3, -1.6)], 0.7, 150, 0.3)
    line(foot(4.61, -1.9, 0.26, 0.18, -90.0), 0.9, 220, 0.2, closed=True)
    line(foot(4.67, 1.3, 1.4, 0.6, -90.0), 1.4, closed=True)                              # wardrobe
    line([P(4.37, 1.3), P(4.97, 1.3)], 0.8, 190, 0.3)
    for zz, a0, a1 in ((0.6, math.pi / 2, math.pi), (2.0, math.pi, 1.5 * math.pi)):
        pts = arc(*P(4.37, zz), 0.7 * K, a0, a1, 16)
        for k in range(0, len(pts) - 1, 2):
            di.line(sc([pts[k], pts[k + 1]], z), fill=140, width=z)

    # red X with a hasty circle, and an arrow from the notes
    kx, ky = P(*MAP_KEY_XZ)
    for (ax, ay, bx, by) in ((-8, -8, 8, 8), (8, -8, -8, 8)):
        line([(kx + ax, ky + ay), (kx + bx, ky + by)], 3.2, 255, 0.6, dr)
    line(arc(kx, ky, 15, -0.4, TAU - 0.1, 50), 1.4, 220, 1.2, dr)
    arrow = bezier((466, 324), (440, 330), (452, 384), (kx + 22, ky + 2), 30)
    line(arrow, 1.1, 210, 0.6)
    line([(kx + 30, ky - 5), (kx + 22, ky + 2), (kx + 32, ky + 7)], 1.1, 210, 0.3)

    # compass rose
    cx, cy, R = 556.0, 132.0, 44.0
    line(arc(cx, cy, R * 0.72, 0, TAU, 60), 0.9, 200, 0.4)
    for k in range(8):
        a = TAU * k / 8 - math.pi / 2
        r = R if k % 2 == 0 else R * 0.55
        tip = (cx + r * math.cos(a), cy + r * math.sin(a))
        l1 = (cx + 7 * math.cos(a + math.pi / 2), cy + 7 * math.sin(a + math.pi / 2))
        l2 = (cx + 7 * math.cos(a - math.pi / 2), cy + 7 * math.sin(a - math.pi / 2))
        line([l1, tip, l2], 1.0, 230, 0.3)
        di.polygon(sc([(cx, cy), tip, l1], z), fill=200 if k % 2 == 0 else 120)

    # lettering
    title = load_font(34 * z, ['FRSCRIPT.TTF', 'MISTRAL.TTF', 'LHANDW.TTF'])
    hand = load_font(int(10.5 * z), ['LHANDW.TTF', 'segoesc.ttf', 'ITCKRIST.TTF'])
    small = load_font(int(8.5 * z), ['LHANDW.TTF', 'segoesc.ttf', 'ITCKRIST.TTF'])
    if title:
        di.text((282 * z, 52 * z), 'The Nursery', fill=240, font=title, anchor='mm')
        line(bezier((200, 74), (250, 82), (320, 68), (366, 76), 30), 1.0, 200, 0.3)
        di.text((cx * z, (cy - R - 10) * z), 'N', fill=230, font=hand, anchor='mm')
    if small:
        for text, X, Zw in (('bed', 0.0, -3.0), ('lamp', 1.3, -3.85), ('toy chest', 0.0, -1.45), ('rug', -3.4, 0.55),
                            ('chair', -3.0, -1.55), ('window', -4.1, -2.35), ('dresser', 3.85, -2.2),
                            ('wardrobe', 3.65, 1.3), ('door', 0.0, 4.1)):
            x, y = P(X, Zw)
            di.text((x * z, y * z), text, fill=215, font=small, anchor='mm')
    if hand:
        notes = ['she sits at the foot', 'of the bed and waits', '', 'the key lies where', 'no light can reach',
                 '', 'do not wake her']
        for k, text in enumerate(notes):
            if text:
                di.text((470 * z, (232 + k * 20) * z), text, fill=rnd.randint(190, 235), font=hand, anchor='lm')
    # faded older scribbles in the margins (these end up on the curled ends)
    for x0, x1 in ((18, 96), (624, 700)):
        y = 70.0
        while y < DH - 60:
            x = x0 + rnd.uniform(0, 10)
            while x < x1:
                wl = rnd.uniform(10, 30)
                pts, t, ph = [], 0.0, rnd.uniform(0, TAU)
                while t < wl:
                    pts.append(((x + t + 1.6 * math.cos(t * 1.2 + ph)) * z, (y - 2.6 * abs(math.sin(t * 0.6 + ph))) * z))
                    t += 0.7
                di.line(pts, fill=rnd.randint(60, 120), width=z)
                x += wl + rnd.uniform(5, 9)
            y += rnd.uniform(15, 19)

    inkm = img_field(ink.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.4)))
    redm = img_field(red.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.6)))
    t = Tex.solid((212, 186, 134))
    t.lerp((178, 142, 92), fbm(4, 4, 5, 3302), 0.6)
    t.shade(fbm(64, 64, 3, 3303), 0.93, 1.05)
    t.shade(noise(S, 36, 3304), 0.97, 1.02)
    stain(t, 3305, 0.1, (180, 142, 88), (128, 92, 54), cells=2, k=0.4, ring_k=0.5)
    edge_n = fbm(8, 8, 4, 3306)
    burn = [0.0] * N
    rim = [0.0] * N
    fold = [0.0] * N
    crease_hi = [0.0] * N
    crease_lo = [0.0] * N
    for i in range(N):
        x = i & 511
        y = i >> 9
        e = min(x, y, S - 1 - x, S - 1 - y) + (edge_n[i] - 0.5) * 36.0
        burn[i] = 1.0 - sstep(2.0, 52.0, e)
        rim[i] = 1.0 - sstep(-2.0, 7.0, e)
        dx, dy = abs(x - 255.5), abs(y - 255.5)
        fold[i] = max(1.0 - sstep(0.0, 9.0, dx), 1.0 - sstep(0.0, 9.0, dy))
        crease_hi[i] = 1.0 if (x == 255 or y == 255) else 0.0
        crease_lo[i] = 1.0 if (x == 257 or y == 257) else 0.0
    fade = fbm(5, 5, 4, 3307)
    t.lerp((44, 30, 20), [m * (0.6 + 0.4 * f) * (1.0 - 0.45 * fo) for m, f, fo in zip(inkm, fade, fold)], 0.92)
    t.lerp((146, 22, 18), [clamp(m * 1.2) * (0.8 + 0.2 * f) for m, f in zip(redm, fade)], 0.92)
    t.lerp((150, 116, 72), fold, 0.25)
    t.lerp((226, 206, 160), crease_hi, 0.45)
    t.lerp((110, 80, 48), crease_lo, 0.4)
    t.lerp((140, 100, 60), speckles(3308, 110, 0.8, 2.6, 0.7), 0.45)            # foxing
    t.lerp((96, 60, 30), burn, 0.75)
    t.lerp((36, 22, 12), rim, 0.85)
    return grime(t, 3309, 0.84, 1.04).image()


def gen_porcelain_cracked():
    """Jump-scare doll face filling the image: hollow black eyes, finely painted lashes and brows,
    greyed blush, tiny painted lips with a thin red cut smeared out to the cheeks, a knocked-out chip
    and hairline cracks through crazed, grimy glaze. Light comes from the upper left."""
    z = 2
    W = S * z
    rnd = random.Random(3401)
    img = Image.new('RGB', (W, W), (238, 233, 224))

    def Z(pts):
        return sc(pts, z)

    def E(cx, cy, rx, ry=None):
        return ellipse_bbox(cx * z, cy * z, rx * z, (rx if ry is None else ry) * z)

    def soft(draw_fn, col, blur, op):
        paint(img, draw_fn, col, blur * z, op)

    # ---- form: head rounding off at the edges, brow ridge, sockets, nose, cheeks, mouth, chin
    edge = Image.new('L', (W, W), 255)
    ImageDraw.Draw(edge).ellipse(E(250, 262, 236, 282), fill=0)
    edge = edge.filter(ImageFilter.GaussianBlur(46 * z))
    img.paste(Image.composite(Image.new('RGB', (W, W), (150, 142, 138)), img, edge))
    soft(lambda d: d.ellipse(E(215, 110, 120, 60), fill=255), (250, 248, 242), 34, 0.7)            # lit forehead
    eyes = [(180.0, 232.0, 1.0), (334.0, 238.0, 1.07)]                                    # right eye sags, larger
    for cx, cy, s in eyes:
        soft(lambda d: d.ellipse(E(cx, cy - 4, 64 * s, 52 * s), fill=255), (168, 156, 152), 16, 0.7)  # socket hollow
        soft(lambda d: d.ellipse(E(cx, cy - 46 * s, 58 * s, 16), fill=255), (248, 246, 240), 10, 0.45)  # brow ridge light
        soft(lambda d: d.ellipse(E(cx, cy + 46 * s, 34 * s, 10), fill=255), (140, 128, 122), 7, 0.45)   # grime under the eye
    for cx, cy in ((158.0, 334.0), (354.0, 340.0)):
        soft(lambda d: d.ellipse(E(cx, cy, 60, 48), fill=255), (194, 134, 138), 24, 0.75)             # rosy cheeks (greyed later)
        soft(lambda d: d.ellipse(E(cx - 8, cy - 28, 24, 12), fill=255), (252, 246, 240), 9, 0.5)      # cheekbone gloss
    soft(lambda d: d.line(Z([(252, 232), (250, 300)]), fill=255, width=int(9 * z)), (250, 248, 244), 5, 0.6)  # nose bridge
    soft(lambda d: d.line(Z([(272, 244), (274, 314)]), fill=255, width=int(10 * z)), (160, 148, 144), 7, 0.55)
    soft(lambda d: d.ellipse(E(256, 314, 15, 12), fill=255), (228, 196, 190), 5, 0.6)
    soft(lambda d: d.ellipse(E(252, 309, 6, 5), fill=255), (255, 254, 250), 3, 0.8)                 # nose tip gloss
    soft(lambda d: d.ellipse(E(258, 330, 22, 6), fill=255), (150, 134, 130), 4, 0.6)                # under the nose
    for nx in (245.0, 268.0):
        soft(lambda d: d.ellipse(E(nx, 329, 5, 3.4), fill=255), (104, 66, 64), 1, 0.95)
    for px_ in (250.0, 263.0):
        soft(lambda d: d.line(Z([(px_, 336), (px_ - 1, 360)]), fill=255, width=int(3 * z)), (184, 172, 168), 3, 0.5)
    for (a, b) in (((236, 326), (220, 378)), ((278, 326), (294, 378))):
        soft(lambda d: d.line(Z(bezier(a, (a[0] - 10 * (1 if a[0] < 256 else -1), 344), b, b, 16)), fill=255,
                              width=int(5 * z)), (168, 156, 152), 6, 0.45)
    soft(lambda d: d.ellipse(E(256, 398, 22, 6), fill=255), (150, 136, 132), 5, 0.5)                # under the lower lip
    soft(lambda d: d.ellipse(E(250, 438, 44, 26), fill=255), (250, 248, 244), 16, 0.45)             # chin gloss
    # ---- painted brows: short feathered hairs along a high, uneven arc
    for (x0, y0, xm, ym, x1, y1) in ((122, 176, 176, 144, 230, 168), (284, 160, 340, 128, 392, 170)):
        arcp = bezier((x0, y0), (xm - 24, ym), (xm + 24, ym), (x1, y1), 40)
        soft(lambda d: d.line(Z(arcp), fill=255, width=int(2 * z), joint='curve'), (120, 92, 78), 1.2, 0.3)
        for k in range(2, 40, 2):
            (ax, ay), (bx, by) = arcp[k - 1], arcp[k]
            ang = math.atan2(by - ay, bx - ax) - 0.5
            ln = rnd.uniform(6, 10) * (0.6 + 0.4 * math.sin(math.pi * k / 40))
            x, y = bx + rnd.uniform(-1, 1), by + rnd.uniform(-1, 1)
            soft(lambda d: d.line(Z([(x, y), (x + math.cos(ang) * ln, y + math.sin(ang) * ln)]), fill=255, width=z),
                 (96, 70, 58), 0.4, 0.75)
    # ---- eyes: lid line, fine lashes, painted lower lashes, broken-out hollow with a lit bisque edge
    voids = Image.new('L', (W, W), 0)                                   # the hollows get no glaze effects
    dv = ImageDraw.Draw(voids)
    for cx, cy, s in eyes:
        rx, ry = 42 * s, 34 * s
        lid = [(cx + (rx + 3) * math.cos(a), cy + (ry + 4) * math.sin(a))
               for a in [math.radians(186 + 168 * k / 40) for k in range(41)]]
        soft(lambda d: d.line(Z(lid), fill=255, width=int(2.4 * z), joint='curve'), (52, 34, 30), 0.5, 0.95)
        for k in range(16):
            a = math.radians(196 + 148 * k / 15)
            bx, by = cx + (rx + 4) * math.cos(a), cy + (ry + 5) * math.sin(a)
            ln = rnd.uniform(9, 15) * s * (0.7 + 0.3 * math.sin(math.pi * k / 15))
            ox, oy = math.cos(a) * ln, math.sin(a) * ln
            side = 4 if math.cos(a) > 0 else -4
            tip = (bx + ox + side, by + oy - 2)
            soft(lambda d: d.line(Z([(bx, by), (bx + ox * 0.55, by + oy * 0.55), tip]), fill=255, width=z),
                 (40, 28, 26), 0.35, 0.9)
        low = [(cx + (rx + 5) * math.cos(a), cy + (ry + 5) * math.sin(a))
               for a in [math.radians(25 + 130 * k / 30) for k in range(31)]]
        soft(lambda d: d.line(Z(low), fill=255, width=z, joint='curve'), (120, 86, 80), 0.6, 0.5)
        for k in range(9):
            a = math.radians(38 + 104 * k / 8)
            bx, by = cx + (rx + 6) * math.cos(a), cy + (ry + 6) * math.sin(a)
            ln = rnd.uniform(5, 8) * s
            soft(lambda d: d.line(Z([(bx, by), (bx + math.cos(a) * ln, by + math.sin(a) * ln)]), fill=255, width=z),
                 (98, 66, 62), 0.35, 0.75)
        soft(lambda d: d.ellipse(E(cx, cy, rx + 4, ry + 4), fill=255), (110, 96, 94), 4, 0.7)       # dirty socket rim
        hole = jagged_poly(cx, cy + 1, rx, ry, rnd, 0.07)
        inner = [(cx + (x - cx) * 0.95, cy + (y - cy) * 0.93 - 2.8) for x, y in hole]
        soft(lambda d: d.polygon(Z(hole), fill=255), (236, 230, 220), 0.4, 1.0)           # bisque thickness
        soft(lambda d: d.polygon(Z(inner), fill=255), (4, 3, 3), 0.6, 1.0)
        dv.polygon(Z(inner), fill=255)
        soft(lambda d: d.ellipse(E(cx, cy + 4, 15 * s, 11 * s), fill=255), (44, 4, 4), 8, 0.5)      # something red, deep inside
        # grime tears running from the sockets
        for k in range(2):
            x = cx + rnd.uniform(-18, 12) * s
            y = cy + ry * 0.9
            pts = [(x, y)]
            drift = rnd.uniform(-0.6, 0.6)
            for _ in range(rnd.randint(7, 15)):
                y += rnd.uniform(6, 11)
                drift += rnd.uniform(-0.5, 0.5)
                x += drift
                pts.append((x, y))
            soft(lambda d: d.line(Z(pts), fill=255, width=int(6 * z), joint='curve'), (140, 128, 124), 3, 0.35)
            for j in range(len(pts) - 1):
                w = max(1, int((2.6 - 1.8 * j / len(pts)) * z))
                soft(lambda d: d.line(Z([pts[j], pts[j + 1]]), fill=255, width=w), (84, 70, 68),
                     0.6, 0.6 * (1.0 - 0.5 * j / len(pts)))
            soft(lambda d: d.ellipse(E(pts[-1][0], pts[-1][1] + 2, 2.6, 3.6), fill=255), (78, 64, 62), 0.4, 0.6)
    # ---- tiny painted cupid's-bow lips, flaking
    lips = (bezier((232, 373), (241, 362), (250, 362), (256, 368), 12)
            + bezier((256, 368), (262, 362), (271, 362), (280, 373), 12)
            + bezier((280, 373), (272, 388), (240, 388), (232, 373), 16))
    soft(lambda d: d.polygon(Z(lips), fill=255), (150, 34, 40), 0.7, 0.92)
    soft(lambda d: d.line(Z(bezier((234, 374), (248, 378), (264, 378), (278, 374), 16)), fill=255, width=int(1.6 * z)),
         (80, 14, 18), 0.4, 0.9)
    soft(lambda d: d.ellipse(E(252, 381, 8, 2.4), fill=255), (214, 150, 150), 1.5, 0.6)
    for _ in range(6):                                                  # paint flaking off the lips
        x, y = rnd.uniform(238, 276), rnd.uniform(367, 383)
        r = rnd.uniform(0.6, 1.3)
        soft(lambda d: d.ellipse(E(x, y, r * 1.6, r), fill=255), (204, 168, 164), 0.3, 0.7)
    # ---- the smile cut too wide: from the lip corners up into the cheeks, smeared, uneven, running
    cutl = wobble(bezier((233, 375), (206, 380), (170, 368), (142, 324), 30), rnd, 1.6, 3.0)
    cutr = wobble(bezier((279, 375), (310, 381), (350, 366), (376, 316), 30), rnd, 1.6, 3.0)
    for cut, dx in ((cutl, -5), (cutr, 6)):
        soft(lambda d: d.line(Z([(x + dx, y + 7) for x, y in cut]), fill=255, width=int(9 * z), joint='curve'),
             (166, 96, 98), 6, 0.4)                                                           # wiped smear
        n = len(cut)
        for j in range(n - 1):
            w = max(1, int((3.0 - 2.2 * j / n) * z))
            soft(lambda d: d.line(Z([cut[j], cut[j + 1]]), fill=255, width=w), (128, 10, 16), 0.5, 0.95)
    slip = wobble(bezier((300, 384), (322, 386), (344, 376), (356, 360), 12), rnd, 1.0, 3.0)  # where the blade slipped
    soft(lambda d: d.line(Z(slip), fill=255, width=z, joint='curve'), (136, 20, 24), 0.4, 0.8)
    for _ in range(10):
        cut = cutl if rnd.random() < 0.5 else cutr
        x, y = cut[rnd.randrange(3, len(cut) - 3)]
        soft(lambda d: d.line(Z([(x, y), (x + rnd.uniform(-10, 10), y + rnd.uniform(3, 9))]), fill=255,
                              width=int(2.5 * z)), (140, 30, 34), 2, 0.45)
    for x, y, ln in ((cutl[9][0], cutl[9][1], 40), (cutr[7][0], cutr[7][1], 62), (cutr[20][0], cutr[20][1], 24),
                     (262, 386, 30)):
        run = wobble([(x, y), (x + 1.0, y + ln * 0.5), (x - 0.5, y + ln)], rnd, 0.8, 3.0)
        soft(lambda d: d.line(Z(run), fill=255, width=int(2.2 * z), joint='curve'), (118, 10, 16), 0.4, 0.92)
        soft(lambda d: d.ellipse(E(run[-1][0], run[-1][1] + 2, 2.4, 3.2), fill=255), (118, 10, 16), 0.4, 0.92)
    # ---- knocked-out chip on the brow: bright broken edge around a dark hollow
    chip_c = (214.0, 106.0)
    rim_p = jagged_poly(chip_c[0], chip_c[1], 21, 15, rnd, 0.4)
    hole = [(chip_c[0] + (x - chip_c[0]) * 0.8, chip_c[1] + (y - chip_c[1]) * 0.8 + 1.5) for x, y in rim_p]
    soft(lambda d: d.polygon(Z([(x + 2, y + 3) for x, y in rim_p]), fill=255), (130, 118, 114), 3, 0.6)
    soft(lambda d: d.polygon(Z(rim_p), fill=255), (226, 222, 214), 0.4, 1.0)
    soft(lambda d: d.polygon(Z(hole), fill=255), (22, 17, 17), 0.4, 1.0)
    soft(lambda d: d.polygon(Z([(x, y - 3) for x, y in hole]), fill=255), (8, 6, 6), 1.5, 0.6)
    # ---- crack network
    cr = Image.new('L', (W, W), 0)
    dc = ImageDraw.Draw(cr)

    def crack(pts, width=1.0, jit=2.0):
        out = []
        for (ax, ay), (bx, by) in zip(pts, pts[1:]):
            n = max(2, int(math.hypot(bx - ax, by - ay) / 4.5))
            for k in range(n):
                t = k / n
                out.append((ax + (bx - ax) * t + rnd.uniform(-jit, jit), ay + (by - ay) * t + rnd.uniform(-jit, jit)))
        out.append(pts[-1])
        dc.line(Z(out), fill=255, width=max(1, int(width * z)))
        return out

    main = crack([(196, -4), (204, 50), (210, 92), (198, 160), (186, 200), (176, 262), (166, 300), (148, 352),
                  (158, 430), (132, 520)], 1.2)
    crack([(326, 276), (322, 318), (304, 356), (286, 372)], 1.0)
    crack([(354, -4), (360, 40), (374, 72), (404, 98), (420, 150), (446, 182)], 0.9)
    for k in range(7):
        a = -math.pi / 2 + (k - 3) * 0.6 + rnd.uniform(-0.2, 0.2)
        x, y = chip_c[0] + 21 * math.cos(a), chip_c[1] + 15 * math.sin(a)
        pts = [(x, y)]
        for _ in range(rnd.randint(6, 18)):
            a += rnd.uniform(-0.45, 0.45)
            x += math.cos(a) * 5.0
            y += math.sin(a) * 5.0
            pts.append((x, y))
        dc.line(Z(pts), fill=rnd.randint(150, 240), width=z)
    for _ in range(12):
        x, y = main[rnd.randrange(len(main))]
        a = rnd.uniform(0, TAU)
        pts = [(x, y)]
        for _s in range(rnd.randint(3, 8)):
            a += rnd.uniform(-0.5, 0.5)
            x += math.cos(a) * 4.5
            y += math.sin(a) * 4.5
            pts.append((x, y))
        dc.line(Z(pts), fill=rnd.randint(120, 200), width=z)

    t = Tex.from_image(img.resize((S, S), Image.LANCZOS))
    bare = t.copy()
    voidm = img_field(voids.resize((S, S), Image.LANCZOS).filter(ImageFilter.GaussianBlur(0.8)))
    crs = cr.resize((S, S), Image.LANCZOS)
    crm = img_field(crs)
    seep = img_field(crs.filter(ImageFilter.GaussianBlur(2.2)))
    hi = img_field(ImageChops.offset(crs, 1, 1))
    # ---- glaze: blush gone grey, crazing in patches, grime in the cracks and recesses
    t.desat([1.0] * N, 0.35)
    t.lerp((198, 194, 190), fbm(4, 4, 5, 3402), 0.3)
    t.shade(fbm(48, 48, 3, 3403), 0.96, 1.03)
    f1, f2, _ = voronoi(40, 40, 3404, 0.95)
    patch = uniform(fbm(3, 3, 4, 3405))
    t.lerp((150, 140, 132), [(1.0 - sstep(0.2, 0.8, b - a)) * sstep(0.55, 0.9, g) for a, b, g in zip(f1, f2, patch)], 0.28)
    t.lerp((118, 104, 98), seep, 0.45)
    t.lerp((255, 252, 246), [max(0.0, h - c) for h, c in zip(hi, crm)], 0.4)
    t.lerp((36, 28, 26), crm, 0.88)
    t.lerp((120, 108, 102), speckles(3406, 260, 0.5, 1.4), 0.35)
    t.lerp(bare, voidm)
    return grime(t, 3407, 0.86, 1.03).image()


# =============================================================================
TEXTURES = {
    'wood_floor': gen_wood_floor,
    'wallpaper_damask': gen_wallpaper_damask,
    'wallpaper_faded': gen_wallpaper_faded,
    'plaster': gen_plaster,
    'wood_dark': gen_wood_dark,
    'wood_light': gen_wood_light,
    'wood_painted_white': gen_wood_painted_white,
    'stone': gen_stone,
    'brick_soot': gen_brick_soot,
    'velvet_red': gen_velvet_red,
    'linen': gen_linen,
    'quilt': gen_quilt,
    'carpet_persian': gen_carpet_persian,
    'carpet_round': gen_carpet_round,
    'brass': gen_brass,
    'iron': gen_iron,
    'book_spines': gen_book_spines,
    'world_map': gen_world_map,
    'portrait': gen_portrait,
    'clock_face': gen_clock_face,
    'mirror': gen_mirror,
    'glass': gen_glass,
    'wax': gen_wax,
    'flame': gen_flame,
    'embers': gen_embers,
    'skin': gen_skin,
    'coat': gen_coat,
    'trousers': gen_trousers,
    'leather': gen_leather,
    'hair': gen_hair,
    'doll_dress': gen_doll_dress,
    'curtain': gen_curtain,
    'marble': gen_marble,
    'night_sky': gen_night_sky,
    'paper': gen_paper,
    'map_parchment': gen_map_parchment,
    'porcelain_cracked': gen_porcelain_cracked,
}


def main(argv):
    assert S == 512
    ids = argv or list(TEXTURES)
    unknown = [i for i in ids if i not in TEXTURES]
    if unknown:
        sys.exit('unknown texture id(s): ' + ', '.join(unknown))
    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()
    for name in ids:
        t0 = time.time()
        im = TEXTURES[name]()
        assert im.size == (S, S) and im.mode == 'RGB', name
        im.save(os.path.join(OUT_DIR, name + '.png'), optimize=True)
        print('%-20s %5.1fs' % (name, time.time() - t0), flush=True)
    print('done: %d textures in %.1fs -> %s' % (len(ids), time.time() - t_all, OUT_DIR))


if __name__ == '__main__':
    main(sys.argv[1:])
