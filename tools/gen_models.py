"""Procedural model generator for Haunted Manor.

Builds every furniture / prop / character model listed in docs/CONTRACT.md out of a small
modelling toolkit (boxes, chamfered boxes, lathes, sweeps, extrusions, superellipsoids,
wavy sheets) and writes them as textured Wavefront OBJ + MTL files into assets/models/.

Run:  py -3.10 tools/gen_models.py            (writes all models, then verifies them)
      py -3.10 tools/gen_models.py lib_desk   (only the listed ids)

Conventions (see CONTRACT.md): metres, +Y up, fronts face +Z, origin bottom-centre;
[wall] models have their back at z = 0, [hang] models hang below y = 0.
UVs use the OBJ convention (v = 0 is the bottom row of the image).
Tiling materials get world-scale UVs (repeats per metre from TEX_SCALE); one-shot images
(portrait, clock face, rugs, globe, night sky, mirror, paper, book spines) are mapped explicitly.
"""
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "models")

# --------------------------------------------------------------------------------------------
# Vector / matrix helpers
# --------------------------------------------------------------------------------------------
def v_add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def v_sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def v_mul(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def v_dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def v_cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def v_len(a): return math.sqrt(v_dot(a, a))
def v_lerp(a, b, t): return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def v_norm(a):
    l = v_len(a)
    return (0.0, 1.0, 0.0) if l < 1e-12 else (a[0] / l, a[1] / l, a[2] / l)


def m_mul_v(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def m_det(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def m_inv_t(m):
    """Inverse-transpose (normal matrix) of a 3x3 matrix."""
    d = m_det(m)
    c = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for k in range(3):
            r1, r2 = [i for i in range(3) if i != r]
            k1, k2 = [i for i in range(3) if i != k]
            minor = m[r1][k1] * m[r2][k2] - m[r1][k2] * m[r2][k1]
            c[r][k] = ((-1) ** (r + k)) * minor / d
    return c  # cofactor matrix / det == inverse transpose


def rot_x(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return ((1, 0, 0), (0, c, -s), (0, s, c))


def rot_y(deg):  # positive angle turns +Z towards +X
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return ((c, 0, s), (0, 1, 0), (-s, 0, c))


def rot_z(deg):  # positive angle turns +X towards +Y
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def rot_axis(axis, deg):
    x, y, z = v_norm(axis)
    a = math.radians(deg); c, s = math.cos(a), math.sin(a); t = 1 - c
    return ((t * x * x + c, t * x * y - s * z, t * x * z + s * y),
            (t * x * y + s * z, t * y * y + c, t * y * z - s * x),
            (t * x * z - s * y, t * y * z + s * x, t * z * z + c))


# --------------------------------------------------------------------------------------------
# Part: an indexed triangle mesh with per-vertex position / normal / uv
# --------------------------------------------------------------------------------------------
class Part:
    def __init__(self, uv01=False):
        self.p, self.n, self.t, self.f = [], [], [], []
        self.uv01 = uv01  # True: UVs are final (one-shot image); False: UVs are metres, scaled per texture

    def vert(self, p, n, t):
        self.p.append(tuple(p)); self.n.append(tuple(n)); self.t.append(tuple(t))
        return len(self.p) - 1

    def tri(self, a, b, c):
        pa, pb, pc = self.p[a], self.p[b], self.p[c]
        if v_len(v_cross(v_sub(pb, pa), v_sub(pc, pa))) > 1e-12:  # drop degenerate triangles
            self.f.append((a, b, c))

    def merge(self, other):
        o = len(self.p)
        self.p += other.p; self.n += other.n; self.t += other.t
        self.f += [(a + o, b + o, c + o) for a, b, c in other.f]
        return self

    def copy(self):
        q = Part(self.uv01)
        q.p, q.n, q.t, q.f = list(self.p), list(self.n), list(self.t), list(self.f)
        return q

    def xform(self, m, tr=(0, 0, 0)):
        nm = m_inv_t(m)
        self.p = [v_add(m_mul_v(m, p), tr) for p in self.p]
        self.n = [v_norm(m_mul_v(nm, n)) for n in self.n]
        if m_det(m) < 0:  # mirroring flips winding
            self.f = [(a, c, b) for a, b, c in self.f]
        return self

    def t_(self, x=0.0, y=0.0, z=0.0):
        self.p = [(p[0] + x, p[1] + y, p[2] + z) for p in self.p]
        return self

    def rx(self, d): return self.xform(rot_x(d))
    def ry(self, d): return self.xform(rot_y(d))
    def rz(self, d): return self.xform(rot_z(d))
    def rot(self, axis, d): return self.xform(rot_axis(axis, d))
    def s(self, sx, sy=None, sz=None):
        sy = sx if sy is None else sy; sz = sx if sz is None else sz
        return self.xform(((sx, 0, 0), (0, sy, 0), (0, 0, sz)))

    def flip(self):
        self.f = [(a, c, b) for a, b, c in self.f]
        self.n = [v_mul(n, -1) for n in self.n]
        return self

    def map_uv(self, fn):
        self.t = [fn(*t) for t in self.t]
        return self

    def bounds(self):
        xs = [p[0] for p in self.p]; ys = [p[1] for p in self.p]; zs = [p[2] for p in self.p]
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def merged(*parts):
    out = Part(parts[0].uv01)
    for q in parts:
        out.merge(q)
    return out


# --------------------------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------------------------
def _planar_uv(p, n):
    """Planar projection (metres) on the dominant axis, oriented so the image reads upright."""
    ax, ay, az = abs(n[0]), abs(n[1]), abs(n[2])
    if ax >= ay and ax >= az:
        return (-p[2] if n[0] > 0 else p[2], p[1])
    if az >= ay:
        return (p[0] if n[2] > 0 else -p[0], p[1])
    return (p[0], -p[2] if n[1] > 0 else p[2])


def add_poly(part, pts, normal=None, uvs=None, sort=False):
    """Adds a planar convex polygon. With sort=True the points are ordered CCW around `normal`."""
    if normal is None:
        acc = (0.0, 0.0, 0.0)
        for i in range(1, len(pts) - 1):
            acc = v_add(acc, v_cross(v_sub(pts[i], pts[0]), v_sub(pts[i + 1], pts[0])))
        normal = acc
    normal = v_norm(normal)
    if sort:
        c = v_mul(pts[0], 0)
        for q in pts:
            c = v_add(c, q)
        c = v_mul(c, 1.0 / len(pts))
        ref = (1, 0, 0) if abs(normal[0]) < 0.9 else (0, 1, 0)
        u = v_norm(v_cross(ref, normal)); w = v_cross(normal, u)
        order = sorted(range(len(pts)), key=lambda i: math.atan2(v_dot(v_sub(pts[i], c), w), v_dot(v_sub(pts[i], c), u)))
        pts = [pts[i] for i in order]
        if uvs is not None:
            uvs = [uvs[i] for i in order]
    idx = [part.vert(q, normal, uvs[i] if uvs else _planar_uv(q, normal)) for i, q in enumerate(pts)]
    for i in range(1, len(idx) - 1):
        part.tri(idx[0], idx[i], idx[i + 1])
    return part


def boxr(x0, y0, z0, x1, y1, z1, bevel=0.0):
    """Axis-aligned box from min/max corners, optionally with chamfered edges (flat shaded)."""
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0
    if z0 > z1: z0, z1 = z1, z0
    cx, cy, cz = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    a, b, c = (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2
    e = min(bevel, a * 0.45, b * 0.45, c * 0.45)
    part = Part()
    if e <= 1e-6:
        for ax in range(3):
            for sg in (-1, 1):
                pts = []
                for s1 in (-1, 1):
                    for s2 in (-1, 1):
                        q = [0, 0, 0]; q[ax] = sg
                        o = [i for i in range(3) if i != ax]
                        q[o[0]] = s1; q[o[1]] = s2
                        pts.append((cx + q[0] * a, cy + q[1] * b, cz + q[2] * c))
                nrm = [0, 0, 0]; nrm[ax] = sg
                add_poly(part, pts, tuple(nrm), sort=True)
        return part
    h = (a, b, c)
    hi = (a - e, b - e, c - e)

    def pt(signs, full_axes):
        return tuple((cx, cy, cz)[i] + signs[i] * (h[i] if i in full_axes else hi[i]) for i in range(3))

    for ax in range(3):  # main faces
        for sg in (-1, 1):
            o = [i for i in range(3) if i != ax]
            pts = []
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    sgn = [0, 0, 0]; sgn[ax] = sg; sgn[o[0]] = s1; sgn[o[1]] = s2
                    pts.append(pt(sgn, (ax,)))
            nrm = [0, 0, 0]; nrm[ax] = sg
            add_poly(part, pts, tuple(nrm), sort=True)
    for a1 in range(3):  # edge chamfers
        for a2 in range(a1 + 1, 3):
            a3 = 3 - a1 - a2
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    pts = []
                    for s3 in (-1, 1):
                        sgn = [0, 0, 0]; sgn[a1] = s1; sgn[a2] = s2; sgn[a3] = s3
                        pts.append(pt(sgn, (a1,)))
                        pts.append(pt(sgn, (a2,)))
                    nrm = [0, 0, 0]; nrm[a1] = s1; nrm[a2] = s2
                    add_poly(part, pts, tuple(nrm), sort=True)
    for sx in (-1, 1):  # corner triangles
        for sy in (-1, 1):
            for sz in (-1, 1):
                sgn = (sx, sy, sz)
                add_poly(part, [pt(sgn, (0,)), pt(sgn, (1,)), pt(sgn, (2,))], (sx, sy, sz), sort=True)
    return part


def box(w, h, d, c=(0, 0, 0), bevel=0.0):
    """Box of size w x h x d centred at c."""
    return boxr(c[0] - w / 2, c[1] - h / 2, c[2] - d / 2, c[0] + w / 2, c[1] + h / 2, c[2] + d / 2, bevel)


def grid(P, N, T, part=None):
    """Quad grid: P[i][j]; rows i run 'up' (or along a path), columns j run 'right' seen from outside."""
    part = part or Part()
    idx = [[part.vert(P[i][j], N[i][j], T[i][j]) for j in range(len(P[i]))] for i in range(len(P))]
    for i in range(len(P) - 1):
        for j in range(len(P[i]) - 1):
            a, b, c, d = idx[i][j], idx[i][j + 1], idx[i + 1][j + 1], idx[i + 1][j]
            part.tri(a, b, c)
            part.tri(a, c, d)
    return part


def lathe(profile, seg=24, smooth=35.0, uv01=False, a0=0.0, a1=360.0, seam=0.0):
    """Surface of revolution around +Y. profile: [(r, y), ...] listed bottom->top
    (going round the outside), so normals face outward. Corners sharper than `smooth`
    degrees get hard edges. A closed profile (first == last point) makes a torus-like ring.
    `seam` rotates the start angle (degrees)."""
    prof = [(float(r), float(y)) for r, y in profile]
    closed = len(prof) > 2 and abs(prof[0][0] - prof[-1][0]) < 1e-9 and abs(prof[0][1] - prof[-1][1]) < 1e-9
    nseg = len(prof) - 1
    seg_n = []
    for k in range(nseg):
        dr = prof[k + 1][0] - prof[k][0]; dy = prof[k + 1][1] - prof[k][1]
        l = math.hypot(dr, dy) or 1.0
        seg_n.append((dy / l, -dr / l))
    cos_lim = math.cos(math.radians(smooth))

    def end_normal(k, at_start):
        n0 = seg_n[k]
        nb = None
        if at_start:
            if k > 0: nb = seg_n[k - 1]
            elif closed: nb = seg_n[-1]
        else:
            if k < nseg - 1: nb = seg_n[k + 1]
            elif closed: nb = seg_n[0]
        if nb is not None and n0[0] * nb[0] + n0[1] * nb[1] > cos_lim:
            sx, sy = n0[0] + nb[0], n0[1] + nb[1]
            l = math.hypot(sx, sy) or 1.0
            return (sx / l, sy / l)
        return n0

    lens = [0.0]
    for k in range(nseg):
        lens.append(lens[-1] + math.hypot(prof[k + 1][0] - prof[k][0], prof[k + 1][1] - prof[k][1]))
    total = lens[-1] or 1.0
    rmax = max(r for r, _ in prof) or 0.01
    part = Part(uv01)
    full = abs((a1 - a0) - 360.0) < 1e-6
    for k in range(nseg):
        if prof[k][0] < 1e-9 and prof[k + 1][0] < 1e-9:
            continue
        nA, nB = end_normal(k, True), end_normal(k, False)
        P, N, T = [[], []], [[], []], [[], []]
        for j in range(seg + 1):
            f = j / seg
            th = math.radians(a0 + seam + (a1 - a0) * f)
            st, ct = math.sin(th), math.cos(th)
            for row, (r, y), nn, L in ((0, prof[k], nA, lens[k]), (1, prof[k + 1], nB, lens[k + 1])):
                P[row].append((r * st, y, r * ct))
                N[row].append((nn[0] * st, nn[1], nn[0] * ct))
                if uv01:
                    T[row].append((f, L / total))
                else:
                    T[row].append((math.radians((a1 - a0) * f) * rmax, L))
        grid(P, N, T, part)
    return part


def cyl(r, h, seg=20, y0=0.0, r2=None):
    """Capped cylinder (or frustum with top radius r2) standing on y0."""
    r2 = r if r2 is None else r2
    return lathe([(0, y0), (r, y0), (r2, y0 + h), (0, y0 + h)], seg)


def sphere(r, seg=20, rings=12, uv01=False):
    prof = [(r * math.sin(math.pi * i / rings), -r * math.cos(math.pi * i / rings)) for i in range(rings + 1)]
    return lathe(prof, seg, smooth=89, uv01=uv01)


def torus(R, r, seg=24, sides=10, a0=0.0, a1=360.0):
    """Ring in the XZ plane (around +Y)."""
    prof = [(R + r * math.cos(2 * math.pi * i / sides), r * math.sin(2 * math.pi * i / sides)) for i in range(sides + 1)]
    return lathe(prof, seg, smooth=89, a0=a0, a1=a1)


def superellipsoid(rx, ry, rz, e1=0.3, e2=0.3, seg=24, rings=14):
    """Rounded-box like blob (pillows, cushions). e -> 0 squarer, e = 1 ellipsoid."""
    def sp(v, e):
        return math.copysign(abs(v) ** e, v)
    P, N, T = [], [], []
    rm = (rx + rz) / 2
    for i in range(rings + 1):
        eta = -math.pi / 2 + math.pi * i / rings
        ce, se = math.cos(eta), math.sin(eta)
        if i == 0 or i == rings:
            ce = 0.0
        rowp, rown, rowt = [], [], []
        for j in range(seg + 1):
            th = 2 * math.pi * j / seg
            st, ct = math.sin(th), math.cos(th)
            rowp.append((rx * sp(ce, e1) * sp(st, e2), ry * sp(se, e1), rz * sp(ce, e1) * sp(ct, e2)))
            rown.append(v_norm((sp(ce, 2 - e1) * sp(st, 2 - e2) / rx, sp(se, 2 - e1) / ry, sp(ce, 2 - e1) * sp(ct, 2 - e2) / rz)))
            rowt.append((th * rm, (eta + math.pi / 2) * ry))
        P.append(rowp); N.append(rown); T.append(rowt)
    return grid(P, N, T)


def _tri2d(poly):
    """Ear-clipping triangulation of a simple CCW 2D polygon. Returns index triples."""
    idx = list(range(len(poly)))
    out = []

    def area2(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        found = False
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area2(a, b, c) <= 1e-12:
                continue
            inside = False
            for m in idx:
                if m in (i0, i1, i2):
                    continue
                p = poly[m]
                if area2(a, b, p) >= -1e-12 and area2(b, c, p) >= -1e-12 and area2(c, a, p) >= -1e-12:
                    inside = True
                    break
            if not inside:
                out.append((i0, i1, i2))
                idx.pop(k)
                found = True
                break
        if not found:  # numerical trouble: fall back to a fan
            for k in range(1, len(idx) - 1):
                out.append((idx[0], idx[k], idx[k + 1]))
            return out
    if len(idx) == 3:
        out.append(tuple(idx))
    return out


def _poly_area(poly):
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def extrude(poly, depth, smooth=30.0, uv01=False, z0=None):
    """Extrudes a 2D polygon (XY plane) along Z from z0 (default -depth/2) to z0 + depth."""
    poly = [(float(x), float(y)) for x, y in poly]
    if _poly_area(poly) < 0:
        poly.reverse()
    z0 = -depth / 2 if z0 is None else z0
    z1 = z0 + depth
    part = Part(uv01)
    xs = [q[0] for q in poly]; ys = [q[1] for q in poly]
    mnx, mny = min(xs), min(ys)
    W, H = (max(xs) - mnx) or 1, (max(ys) - mny) or 1

    def cap_uv(x, y, front):
        if uv01:
            return ((x - mnx) / W, (y - mny) / H) if front else (1 - (x - mnx) / W, (y - mny) / H)
        return (x, y) if front else (-x, y)

    tris = _tri2d(poly)
    fi = [part.vert((x, y, z1), (0, 0, 1), cap_uv(x, y, True)) for x, y in poly]
    bi = [part.vert((x, y, z0), (0, 0, -1), cap_uv(x, y, False)) for x, y in poly]
    for a, b, c in tris:
        part.tri(fi[a], fi[b], fi[c])
        part.tri(bi[a], bi[c], bi[b])
    n = len(poly)
    en = []
    for k in range(n):
        x0_, y0_ = poly[k]; x1_, y1_ = poly[(k + 1) % n]
        l = math.hypot(x1_ - x0_, y1_ - y0_) or 1
        en.append(((y1_ - y0_) / l, -(x1_ - x0_) / l))
    cl = math.cos(math.radians(smooth))
    s = 0.0
    for k in range(n):
        pa, pb = poly[k], poly[(k + 1) % n]
        nk = en[k]

        def vn(nb):
            if nk[0] * nb[0] + nk[1] * nb[1] > cl:
                sx, sy = nk[0] + nb[0], nk[1] + nb[1]
                l = math.hypot(sx, sy) or 1
                return (sx / l, sy / l, 0)
            return (nk[0], nk[1], 0)

        na, nb_ = vn(en[k - 1]), vn(en[(k + 1) % n])
        L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        A = part.vert((pa[0], pa[1], z0), na, (s, z0)); B = part.vert((pb[0], pb[1], z0), nb_, (s + L, z0))
        C = part.vert((pb[0], pb[1], z1), nb_, (s + L, z1)); D = part.vert((pa[0], pa[1], z1), na, (s, z1))
        part.tri(A, B, C); part.tri(A, C, D)
        s += L
    return part


def sweep(path, section, smooth=True, caps=True, scales=None, hint=None, closed=False):
    """Sweeps a closed 2D section (CCW, in the local N/B plane) along a 3D polyline.
    smooth=True gives averaged normals around the section (tubes); False gives flat sides.
    `hint` picks the initial N direction; `scales` tapers the section per path point."""
    npth = len(path)
    if closed:
        path = list(path) + [path[0]]
        npth = len(path)
    tans = []
    for i in range(npth):
        if closed and (i == 0 or i == npth - 1):
            t = v_add(v_norm(v_sub(path[1], path[0])), v_norm(v_sub(path[-1], path[-2])))
        elif i == 0:
            t = v_sub(path[1], path[0])
        elif i == npth - 1:
            t = v_sub(path[-1], path[-2])
        else:
            t = v_add(v_norm(v_sub(path[i], path[i - 1])), v_norm(v_sub(path[i + 1], path[i])))
        tans.append(v_norm(t))
    h = hint or ((0, 1, 0) if abs(tans[0][1]) < 0.9 else (1, 0, 0))
    Nv = v_norm(v_sub(h, v_mul(tans[0], v_dot(h, tans[0]))))
    frames = []
    for i in range(npth):
        if i > 0:  # parallel transport
            t0, t1 = tans[i - 1], tans[i]
            ax = v_cross(t0, t1)
            if v_len(ax) > 1e-9:
                ang = math.degrees(math.atan2(v_len(ax), v_dot(t0, t1)))
                Nv = v_norm(m_mul_v(rot_axis(ax, ang), Nv))
            Nv = v_norm(v_sub(Nv, v_mul(tans[i], v_dot(Nv, tans[i]))))
        Bv = v_cross(tans[i], Nv)
        frames.append((Nv, Bv))
    sec = [(float(x), float(y)) for x, y in section]
    if _poly_area(sec) < 0:
        sec.reverse()
    ns = len(sec)
    en = []
    for k in range(ns):
        a, b = sec[k], sec[(k + 1) % ns]
        l = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
        en.append(((b[1] - a[1]) / l, -(b[0] - a[0]) / l))
    plen = [0.0]
    for i in range(1, npth):
        plen.append(plen[-1] + v_len(v_sub(path[i], path[i - 1])))
    part = Part()

    def place(i, x, y):
        sc = (scales[i] if scales else 1.0)
        Nv, Bv = frames[i]
        return v_add(path[i], v_add(v_mul(Nv, x * sc), v_mul(Bv, y * sc)))

    def nrm3(i, nx, ny):
        Nv, Bv = frames[i]
        return v_norm(v_add(v_mul(Nv, nx), v_mul(Bv, ny)))

    if smooth:
        vnorm = []
        for k in range(ns):
            a, b = en[k - 1], en[k]
            vnorm.append(v_norm((a[0] + b[0], a[1] + b[1], 0)))
        cols = list(range(ns)) + [0]
        P, N, T = [], [], []
        for i in range(npth):
            s = 0.0
            rp, rn, rt = [], [], []
            for ci, k in enumerate(cols):
                if ci > 0:
                    pa, pb = sec[cols[ci - 1]], sec[k]
                    s += math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                x, y = sec[k]
                rp.append(place(i, x, y)); rn.append(nrm3(i, vnorm[k][0], vnorm[k][1])); rt.append((s, plen[i]))
            P.append(rp); N.append(rn); T.append(rt)
        grid(P, N, T, part)
    else:
        s = 0.0
        for k in range(ns):
            a, b = sec[k], sec[(k + 1) % ns]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            P, N, T = [], [], []
            for i in range(npth):
                P.append([place(i, *a), place(i, *b)])
                N.append([nrm3(i, *en[k])] * 2)
                T.append([(s, plen[i]), (s + L, plen[i])])
            grid(P, N, T, part)
            s += L
    if caps and not closed:
        tris = _tri2d(sec)
        for i, sign in ((0, -1), (npth - 1, 1)):
            tn = v_mul(tans[i], sign)
            ids = [part.vert(place(i, x, y), tn, _planar_uv(place(i, x, y), tn)) for x, y in sec]
            for a, b, c in tris:
                if sign > 0:
                    part.tri(ids[a], ids[b], ids[c])
                else:
                    part.tri(ids[a], ids[c], ids[b])
    return part


def circle2d(r, n=10, rx=None):
    rx = r if rx is None else rx
    return [(rx * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def rect2d(w, h):
    return [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]


def tube(path, r, sides=10, caps=True, scales=None, hint=None):
    return sweep(path, circle2d(r, sides), True, caps, scales, hint)


def arc_pts(c, r, a0, a1, n, plane="xy"):
    """Points on a circular arc (degrees) in the given plane around centre c."""
    out = []
    for i in range(n + 1):
        a = math.radians(a0 + (a1 - a0) * i / n)
        u, w = r * math.cos(a), r * math.sin(a)
        if plane == "xy":
            out.append((c[0] + u, c[1] + w, c[2]))
        elif plane == "zy":
            out.append((c[0], c[1] + w, c[2] + u))
        else:  # xz
            out.append((c[0] + u, c[1], c[2] + w))
    return out


def bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        out.append(tuple(u ** 3 * p0[k] + 3 * u * u * t * p1[k] + 3 * u * t * t * p2[k] + t ** 3 * p3[k] for k in range(3)))
    return out


def sheet(fn, nu, nv, thick=0.01, uv=None):
    """Two-sided thin sheet from fn(u, v) -> point, u,v in [0,1]; front faces the
    side of dP/du x dP/dv. `uv(u, v, P)` gives texture coordinates (default: metres)."""
    P = [[fn(j / nu, i / nv) for j in range(nu + 1)] for i in range(nv + 1)]
    N = []
    for i in range(nv + 1):
        row = []
        for j in range(nu + 1):
            du = v_sub(P[i][min(j + 1, nu)], P[i][max(j - 1, 0)])
            dv = v_sub(P[min(i + 1, nv)][j], P[max(i - 1, 0)][j])
            row.append(v_norm(v_cross(du, dv)))
        N.append(row)
    T = []
    for i in range(nv + 1):
        row = []
        acc = 0.0
        for j in range(nu + 1):
            if j > 0:
                acc += v_len(v_sub(P[i][j], P[i][j - 1]))
            row.append(uv(j / nu, i / nv, P[i][j]) if uv else (acc, P[i][j][1]))
        T.append(row)
    front = grid(P, N, T)
    back = grid([[v_sub(P[i][j], v_mul(N[i][j], thick)) for j in range(nu + 1)] for i in range(nv + 1)],
                N, T).flip()
    return front.merge(back)


# --------------------------------------------------------------------------------------------
# Materials and models
# --------------------------------------------------------------------------------------------
TEX_SCALE = {  # texture repeats per metre for tiling materials
    "wood_dark": 1.2, "wood_light": 1.2, "wood_painted_white": 1.0, "stone": 0.9, "brick_soot": 1.4,
    "velvet_red": 1.6, "linen": 1.2, "quilt": 0.9, "brass": 3.0, "iron": 2.5, "wax": 6.0, "flame": 6.0,
    "embers": 3.0, "skin": 3.0, "coat": 2.0, "trousers": 2.5, "leather": 3.0, "hair": 4.0,
    "doll_dress": 6.0, "curtain": 0.8, "marble": 1.2, "glass": 3.0, "paper": 1.0, "mirror": 1.0,
    "book_spines": 1.0, "world_map": 1.0, "portrait": 1.0, "clock_face": 1.0, "night_sky": 1.0,
    "carpet_persian": 1.0, "carpet_round": 1.0, "map_parchment": 1.0, "porcelain_cracked": 1.0,
}

KE_EMBERS = (1.0, 0.35, 0.08)
KE_FLAME = (1.0, 0.6, 0.2)


class Model:
    def __init__(self, mid, size, mode=""):
        self.id, self.size, self.mode = mid, size, mode
        self.mats = {}    # name -> (texture, Ke)
        self.parts = []   # (material name, Part)
        self.flips = 0    # triangles whose winding had to be corrected (should stay 0)

    def mat(self, tex, ke=None, name=None):
        name = name or (tex if ke is None else tex + "_glow")
        self.mats[name] = (tex, ke)
        return name

    def add(self, part, tex, ke=None, name=None):
        mname = self.mat(tex, ke, name)
        if not part.uv01:
            sc = TEX_SCALE.get(tex, 1.0)
            part = part.copy().map_uv(lambda u, v: (u * sc, v * sc))
        orient_to_normals(part)
        self.flips += part.flips
        self.parts.append((mname, part))
        return part

    def bounds(self):
        mn = [1e9] * 3; mx = [-1e9] * 3
        for _, q in self.parts:
            for p in q.p:
                for k in range(3):
                    mn[k] = min(mn[k], p[k]); mx[k] = max(mx[k], p[k])
        return mn, mx

    def finalize(self):
        """Scales to the exact contract size and moves the origin to the contract convention."""
        mn, mx = self.bounds()
        raw = [mx[k] - mn[k] for k in range(3)]
        sc = [self.size[k] / raw[k] for k in range(3)]
        for k in range(3):
            if abs(sc[k] - 1) > 0.12:
                raise SystemExit(f"{self.id}: raw size {raw} too far from contract {self.size}")
        cx, cz = (mn[0] + mx[0]) / 2, (mn[2] + mx[2]) / 2
        tx, tz = -cx * sc[0], -cz * sc[2]
        ty = -mn[1] * sc[1]
        if self.mode == "wall":
            tz = -mn[2] * sc[2]
        elif self.mode == "hang":
            ty = -mx[1] * sc[1]
        for _, q in self.parts:
            q.xform(((sc[0], 0, 0), (0, sc[1], 0), (0, 0, sc[2])), (tx, ty, tz))
        self.raw = raw
        self.scale = sc

    def tri_count(self):
        return sum(len(q.f) for _, q in self.parts)

    def write(self, out_dir=OUT_DIR):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, self.id + ".mtl"), "w", newline="\n") as fm:
            fm.write(f"# {self.id} materials (generated by tools/gen_models.py)\n")
            for name, (tex, ke) in self.mats.items():
                fm.write(f"\nnewmtl {name}\nKa 0 0 0\nKd 1 1 1\nKs 0.1 0.1 0.1\nNs 24\nd 1\nillum 2\n")
                if ke is not None:
                    fm.write("Ke %.3f %.3f %.3f\n" % ke)
                fm.write(f"map_Kd ../textures/{tex}.png\n")
        vmap, tmap, nmap = {}, {}, {}
        vs, ts, ns = [], [], []

        def key_of(store, lst, val, nd):
            k = tuple(round(x, nd) + 0.0 for x in val)
            i = store.get(k)
            if i is None:
                lst.append(k); i = len(lst); store[k] = i
            return i

        groups = {}
        for mname, q in self.parts:
            groups.setdefault(mname, []).append(q)
        flines = []
        for mname, plist in groups.items():
            flines.append(f"g {self.id}_{mname}\nusemtl {mname}\n")
            for q in plist:
                for a, b, c in q.f:
                    corner = []
                    for i in (a, b, c):
                        vi = key_of(vmap, vs, q.p[i], 5)
                        ti = key_of(tmap, ts, q.t[i], 4)
                        ni = key_of(nmap, ns, v_norm(q.n[i]), 4)
                        corner.append(f"{vi}/{ti}/{ni}")
                    flines.append("f " + " ".join(corner) + "\n")
        with open(os.path.join(out_dir, self.id + ".obj"), "w", newline="\n") as fo:
            fo.write(f"# {self.id} - Haunted Manor (generated by tools/gen_models.py)\n")
            fo.write(f"# size {self.size[0]} x {self.size[1]} x {self.size[2]} m, {self.tri_count()} triangles\n")
            fo.write(f"mtllib {self.id}.mtl\no {self.id}\n")
            fo.writelines("v %g %g %g\n" % v for v in vs)
            fo.writelines("vt %g %g\n" % t for t in ts)
            fo.writelines("vn %g %g %g\n" % n for n in ns)
            fo.writelines(flines)


MODELS = {}


def model(mid, size, mode=""):
    def deco(fn):
        MODELS[mid] = (fn, size, mode)
        return fn
    return deco


# ===================================== shared props =====================================
def flame(h, r, seg=12):
    """Tear-drop candle flame standing on y = 0 (one-shot UVs: image wraps once around)."""
    prof = [(0, 0), (r * 0.55, h * 0.03), (r * 0.95, h * 0.13), (r, h * 0.27), (r * 0.82, h * 0.47),
            (r * 0.48, h * 0.7), (r * 0.16, h * 0.9), (0, h)]
    return lathe(prof, seg, smooth=70, uv01=True)


def candle(r, h, seg=12):
    """Wax candle with a slightly melted, dished top."""
    return lathe([(0, 0), (r, 0), (r, h * 0.96), (r * 0.9, h), (r * 0.55, h * 0.975), (0, h * 0.975)], seg, smooth=40)


def knob(r, depth, seg=12):
    """Round drawer knob pointing +Z, base at z = 0."""
    return lathe([(0, 0), (r * 0.7, 0), (r * 0.45, depth * 0.35), (r * 0.9, depth * 0.6), (r, depth * 0.8),
                  (r * 0.6, depth), (0, depth)], seg, smooth=50).rx(90)


def bail_pull(w=0.07):
    """Drawer bail handle facing +Z: backplate + hanging half ring, centred at origin."""
    plate = boxr(-w / 2, -0.016, 0, w / 2, 0.016, 0.004, 0.002)
    ring = torus(w * 0.36, 0.0035, 14, 6, 90, 270).rx(-90).rx(-20).t_(0, 0.004, 0.008)
    bosses = [sphere(0.006, 8, 5).t_(sx * w * 0.36, 0.004, 0.008) for sx in (-1, 1)]
    return merged(plate, ring, *bosses)


def bun_foot(r, h, seg=14):
    return lathe([(0, 0), (r * 0.6, 0), (r, h * 0.35), (r * 0.95, h * 0.7), (r * 0.6, h * 0.9), (r * 0.55, h), (0, h)], seg, smooth=50)


def mitred_frame(W, H, prof, smooth=40.0):
    """Picture-frame ring with mitred corners from a closed cross-section profile [(d, z), ...]:
    d = inset from the outer edge, z = height above the back (z = 0). The profile runs from the
    outer back corner up the outside, over the face, down the inside and back along the bottom."""
    prof = list(prof)
    if prof[0] != prof[-1]:
        prof.append(prof[0])
    ns = len(prof) - 1
    sn = []
    for k in range(ns):
        dd = prof[k + 1][0] - prof[k][0]; dz = prof[k + 1][1] - prof[k][1]
        l = math.hypot(dd, dz) or 1
        sn.append((-dz / l, dd / l))
    cl = math.cos(math.radians(smooth))

    def end_n(k, other):
        a, b = sn[k], sn[other % ns]
        if a[0] * b[0] + a[1] * b[1] > cl:
            x, y = a[0] + b[0], a[1] + b[1]; l = math.hypot(x, y) or 1
            return (x / l, y / l)
        return a

    out = Part()
    for L, E, ang in ((W / 2, H / 2, 0), (H / 2, W / 2, 90), (W / 2, H / 2, 180), (H / 2, W / 2, 270)):
        bar = Part()
        acc = 0.0
        for k in range(ns):
            (d0, z0), (d1, z1) = prof[k], prof[k + 1]
            n0, n1 = end_n(k, k - 1), end_n(k, k + 1)
            seglen = math.hypot(d1 - d0, z1 - z0)
            P = [[(-(L - d0), -E + d0, z0), (L - d0, -E + d0, z0)], [(-(L - d1), -E + d1, z1), (L - d1, -E + d1, z1)]]
            N = [[(0, n0[0], n0[1])] * 2, [(0, n1[0], n1[1])] * 2]
            T = [[(-(L - d0), acc), (L - d0, acc)], [(-(L - d1), acc + seglen), (L - d1, acc + seglen)]]
            grid(P, N, T, bar)
            acc += seglen
        out.merge(bar.rz(ang))
    return out


def orient_to_normals(part):
    """Makes every triangle's winding agree with its vertex normals (CCW seen from outside)."""
    fixed = []
    part.flips = 0
    for a, b, c in part.f:
        g = v_cross(v_sub(part.p[b], part.p[a]), v_sub(part.p[c], part.p[a]))
        n = v_add(v_add(part.n[a], part.n[b]), part.n[c])
        if v_dot(g, n) < 0:
            fixed.append((a, c, b))
            part.flips += 1
        else:
            fixed.append((a, b, c))
    part.f = fixed
    return part


# Book-spine atlas layout (see assets/textures/book_spines.png).
SPINE_ROWS = 4          # rows of books stacked vertically in the texture
SPINE_TEX_W = 0.45      # metres of shelf represented by the full texture width
SPINE_MARGIN = 0.02


def book_part(t, h, d, rng):
    """Upright book, spine on +Z at z = 0, occupying x 0..t, y 0..h, z -d..0.
    The spine samples a random strip of the book_spines atlas; covers/pages take its colour."""
    p = boxr(0, 0, -d, t, h, 0)
    du = min(0.95, t / SPINE_TEX_W)
    u0 = rng.uniform(0.0, 1.0 - du)
    row = rng.randrange(SPINE_ROWS)
    v0 = (row + SPINE_MARGIN) / SPINE_ROWS
    dv = (1 - 2 * SPINE_MARGIN) / SPINE_ROWS
    T = []
    for (x, y, z), n in zip(p.p, p.n):
        if n[2] > 0.5:
            T.append((u0 + x / t * du, v0 + y / h * dv))
        else:
            T.append((u0 + du * 0.5, v0 + dv * 0.08))
    p.t = T
    p.uv01 = True
    return p


# ===================================== MODELS =====================================
@model("player", (0.55, 1.80, 0.35))
def m_player(m):
    SK, CO, TR, LE, HA = "skin", "coat", "trousers", "leather", "hair"
    for sx in (-1, 1):
        x = sx * 0.1
        # boots: rounded foot + shaft + sole + heel
        m.add(superellipsoid(0.052, 0.045, 0.15, 0.45, 0.6, 16, 10).t_(x, 0.05, 0.035), LE)
        m.add(lathe([(0, 0.05), (0.056, 0.05), (0.057, 0.25), (0.062, 0.3), (0, 0.3)], 14), LE).t_(x, 0, -0.01)
        m.add(boxr(x - 0.05, 0.0, -0.11, x + 0.05, 0.018, 0.17, 0.006), LE)
        m.add(boxr(x - 0.042, 0.0, -0.115, x + 0.042, 0.04, -0.05, 0.006), LE)
        # trouser legs
        m.add(lathe([(0, 0.27), (0.058, 0.27), (0.062, 0.45), (0.072, 0.62), (0.085, 0.84), (0.0, 0.9)], 14), TR).t_(x, 0, 0)
    # long coat (elliptical cross-section)
    coat = lathe([(0, 0.6), (0.215, 0.6), (0.21, 0.7), (0.195, 0.86), (0.178, 0.98), (0.172, 1.03), (0.182, 1.13),
                  (0.198, 1.26), (0.205, 1.36), (0.196, 1.42), (0.165, 1.465), (0.11, 1.5), (0.06, 1.52), (0, 1.52)], 24, smooth=50)
    m.add(coat.s(1, 1, 0.74), CO)
    # stand-up collar and lapels
    m.add(lathe([(0.055, 1.46), (0.085, 1.465), (0.09, 1.52), (0.075, 1.575), (0.06, 1.57), (0.055, 1.46)], 16, smooth=50).s(1, 1, 0.95).t_(0, 0, -0.005), CO)
    # front placket running from the collar to the hem, following the coat's front profile
    plk = [(0.0, y, 0.74 * r + 0.002) for y, r in ((0.62, 0.213), (0.75, 0.205), (0.86, 0.195), (0.98, 0.178), (1.03, 0.172),
                                                    (1.13, 0.182), (1.26, 0.198), (1.36, 0.205), (1.42, 0.196), (1.46, 0.17))]
    m.add(sweep(plk, rect2d(0.035, 0.006), smooth=False, hint=(1, 0, 0)), CO)
    for sx in (-1, 1):
        # flap pockets
        m.add(boxr(-0.05, -0.018, -0.008, 0.05, 0.018, 0.008, 0.005).ry(sx * 35).t_(sx * 0.109, 0.875, 0.118), CO)
    # buttons and belt
    for i in range(5):
        y = 0.92 + i * 0.1
        z = 0.74 * (0.178 if y < 1.05 else 0.19) + 0.005
        m.add(sphere(0.011, 8, 5).s(1, 1, 0.6).t_(0, y, z), LE)
    m.add(lathe([(0.176, 1.0), (0.181, 1.0), (0.181, 1.045), (0.176, 1.045), (0.176, 1.0)], 24).s(1, 1, 0.76), LE)
    m.add(boxr(-0.025, 0.995, 0.128, 0.025, 1.05, 0.142, 0.004), LE)
    # arms: shoulder ball, sleeves (tapered tubes), cuffs, hands
    for sx in (-1, 1):
        sh = (sx * 0.2, 1.4, 0.0)
        el = (sx * 0.22, 1.12, 0.012)
        wr = (sx * 0.226, 0.87, 0.035)
        m.add(sphere(0.054, 14, 8).t_(sh[0], sh[1] - 0.012, sh[2]), CO)
        up = [v_lerp(sh, el, i / 4) for i in range(5)]
        lo = [v_lerp(el, wr, i / 4) for i in range(5)]
        m.add(tube(up, 0.054, 12, True, [1.0, 0.98, 0.95, 0.92, 0.9]), CO)
        m.add(sphere(0.049, 12, 8).t_(*el), CO)
        m.add(tube(lo, 0.049, 12, True, [1.0, 0.97, 0.94, 0.92, 0.9]), CO)
        cuff_dir = v_norm(v_sub(wr, el))
        cuff = lathe([(0.044, -0.02), (0.048, -0.02), (0.048, 0.015), (0.044, 0.015), (0.044, -0.02)], 14)
        ang = math.degrees(math.acos(max(-1, min(1, v_dot((0, -1, 0), cuff_dir)))))
        axis = v_cross((0, 1, 0), v_mul(cuff_dir, -1))
        if v_len(axis) > 1e-6:
            cuff.rot(axis, ang)
        m.add(cuff.t_(*v_add(wr, v_mul(cuff_dir, -0.005))), CO)
        hand = superellipsoid(0.026, 0.058, 0.045, 0.7, 0.7, 12, 10).rz(sx * 4).t_(sx * 0.23, 0.8, 0.04)
        m.add(hand, SK)
        m.add(superellipsoid(0.013, 0.03, 0.014, 0.8, 0.8, 8, 6).rx(25).t_(sx * 0.222, 0.81, 0.083), SK)
    # neck and head
    m.add(cyl(0.047, 0.13, 14, 1.47), SK)
    m.add(superellipsoid(0.083, 0.108, 0.097, 0.9, 0.9, 20, 14).t_(0, 1.668, 0.012), SK)
    m.add(superellipsoid(0.06, 0.05, 0.06, 0.9, 0.9, 14, 8).t_(0, 1.598, 0.04), SK)  # jaw / chin
    m.add(superellipsoid(0.013, 0.027, 0.016, 0.7, 0.7, 10, 8).rx(-18).t_(0, 1.655, 0.1), SK)  # nose
    for sx in (-1, 1):
        m.add(superellipsoid(0.012, 0.028, 0.02, 0.9, 0.9, 10, 6).t_(sx * 0.084, 1.664, 0.0), SK)          # ears
        m.add(sphere(0.0095, 8, 6).t_(sx * 0.031, 1.69, 0.097), HA)                                       # eyes
        m.add(boxr(-0.019, -0.004, -0.005, 0.019, 0.004, 0.005, 0.002).rz(-sx * 8).t_(sx * 0.032, 1.712, 0.101), HA)  # brows
        m.add(boxr(sx * 0.074, 1.6, -0.04, sx * 0.087, 1.69, 0.03, 0.006), HA)                           # sideburns
    m.add(boxr(-0.022, 1.588, 0.095, 0.022, 1.596, 0.104, 0.002), SK)  # lips
    # hair: tilted cap of an ellipsoid covering top and back of the head
    cap = [(0.095 * math.cos(math.radians(a)), 0.095 * math.sin(math.radians(a))) for a in range(-25, 91, 8)] + [(0.0, 0.095)]
    m.add(lathe(cap, 22, smooth=60).s(0.97, 1.18, 1.1).rx(-10).t_(0, 1.672, -0.006), HA)
    m.add(superellipsoid(0.07, 0.03, 0.05, 0.7, 0.7, 14, 8).rx(-20).t_(0.012, 1.762, 0.05), HA)   # swept-back front lock


@model("lib_bookshelf", (2.0, 2.6, 0.45))
def m_bookshelf(m):
    W = "wood_dark"
    rng = random.Random(11)
    zb, zf = -0.225, 0.165
    m.add(boxr(-0.975, 0.0, zb, 0.975, 0.12, 0.195, 0.012), W)          # plinth
    m.add(boxr(-0.96, 0.095, zb, 0.96, 0.135, 0.2, 0.008), W)            # plinth top moulding
    for sx in (-1, 1):
        m.add(boxr(sx * 0.95, 0.12, zb, sx * 0.9, 2.42, zf), W)            # carcass sides
        m.add(boxr(sx * 0.965, 0.135, zf - 0.01, sx * 0.885, 2.39, zf + 0.022, 0.007), W)  # pilasters
        for y in (0.2, 2.3):                                                  # pilaster capitals / bases
            m.add(boxr(sx * 0.972, y, zf - 0.01, sx * 0.878, y + 0.05, zf + 0.03, 0.008), W)
    m.add(boxr(-0.9, 0.12, zb, 0.9, 2.42, zb + 0.02), W)                     # back
    m.add(boxr(-0.95, 2.39, zb, 0.95, 2.42, zf), W)                          # top
    m.add(boxr(-0.9, 0.12, zb, 0.9, 0.16, zf), W)                            # bottom
    m.add(boxr(-0.02, 0.16, zb, 0.02, 2.39, zf), W)                          # centre divider
    m.add(boxr(-0.036, 0.135, zf - 0.01, 0.036, 2.39, zf + 0.022, 0.007), W)
    # crown moulding: stepped cornice with a row of dentils
    m.add(boxr(-0.97, 2.42, zb, 0.97, 2.46, 0.19, 0.008), W)
    m.add(boxr(-0.985, 2.46, zb, 0.985, 2.5, 0.205, 0.016), W)
    m.add(boxr(-0.965, 2.5, zb, 0.965, 2.555, 0.19), W)
    m.add(boxr(-1.0, 2.555, zb, 1.0, 2.6, 0.225, 0.012), W)
    n_dent = 31
    for i in range(n_dent):
        x = -0.93 + 1.86 * i / (n_dent - 1)
        m.add(boxr(x - 0.016, 2.515, 0.19, x + 0.016, 2.555, 0.208, 0.003), W)
    for sx in (-1, 1):  # dentils on the sides
        for k in range(6):
            z = -0.19 + k * 0.07
            m.add(boxr(sx * 0.965, 2.515, z - 0.016, sx * 0.983, 2.555, z + 0.016, 0.003), W)
    # shelves
    comp_h = (2.39 - 0.16 - 4 * 0.03) / 5
    bottoms = [0.16 + c * (comp_h + 0.03) for c in range(5)]
    for c in range(1, 5):
        y = bottoms[c] - 0.03
        m.add(boxr(-0.9, y, zb + 0.02, 0.9, y + 0.03, zf - 0.005), W)
        for x0, x1 in ((-0.9, -0.02), (0.02, 0.9)):
            m.add(boxr(x0, y - 0.006, zf - 0.02, x1, y + 0.03, zf + 0.004, 0.005), W)  # shelf lip
    # books
    for c, yb in enumerate(bottoms):
        for x0, x1 in ((-0.9, -0.02), (0.02, 0.9)):
            fill_shelf(m, x0 + 0.006, x1 - 0.006, yb, comp_h, rng)


def fill_shelf(m, x0, x1, yb, comp_h, rng):
    x = x0
    max_h = comp_h - 0.035
    prev_h = max_h * 0.8
    while x < x1 - 0.02:
        r = rng.random()
        if r < 0.07 and x1 - x > 0.3:  # horizontal stack
            w = rng.uniform(0.2, 0.26)
            y = yb
            for _ in range(rng.randint(2, 5)):
                th = rng.uniform(0.025, 0.045)
                d = rng.uniform(0.15, 0.2)
                ww = w - rng.uniform(0, 0.04)
                b = book_part(th, ww, d, rng).rz(-90).t_(0, th, 0)
                zfront = 0.15 - rng.uniform(0, 0.02)
                m.add(b.t_(x + rng.uniform(0, w - ww), y, zfront), "book_spines")
                y += th
            x += w + rng.uniform(0.005, 0.02)
            prev_h = 0.1
        elif r < 0.13 and x1 - x > 0.1 and prev_h > 0.18:  # gap then a book leaning on the previous one
            t = rng.uniform(0.025, 0.045)
            h = rng.uniform(0.6, 0.9) * max_h
            a = rng.uniform(12, 24)
            px = x + h * math.sin(math.radians(a))
            d = rng.uniform(0.16, 0.24)
            b = book_part(t, h, d, rng).rz(a)
            m.add(b.t_(px, yb, 0.15 - rng.uniform(0, 0.02)), "book_spines")
            x = px + t * math.cos(math.radians(a)) + rng.uniform(0.005, 0.03)
            prev_h = h * 0.6
        else:
            t = rng.uniform(0.022, 0.06)
            if x + t > x1:
                break
            h = rng.uniform(0.62, 1.0) * max_h
            d = rng.uniform(0.17, 0.27)
            b = book_part(t, h, d, rng)
            m.add(b.t_(x, yb, 0.152 - rng.uniform(0, 0.025)), "book_spines")
            x += t + (0.0015 if rng.random() < 0.8 else rng.uniform(0.003, 0.01))
            prev_h = h


@model("lib_desk", (1.6, 0.78, 0.8))
def m_desk(m):
    W, P, BR = "wood_dark", "paper", "brass"
    m.add(boxr(-0.8, 0.69, -0.4, 0.8, 0.72, 0.4, 0.008), W)               # top
    m.add(boxr(-0.785, 0.672, -0.385, 0.785, 0.692, 0.385, 0.006), W)     # top moulding
    for sx in (-1, 1):
        xi, xo = sx * 0.36, sx * 0.765
        m.add(boxr(sx * 0.345, 0.0, -0.385, sx * 0.78, 0.06, 0.37, 0.01), W)     # plinth
        m.add(boxr(xi, 0.06, -0.37, xo, 0.672, 0.35), W)                         # pedestal carcass
        m.add(boxr(sx * 0.763, 0.13, -0.3, sx * 0.772, 0.6, 0.29, 0.006), W)      # side panel
        # three drawers per pedestal
        y = 0.08
        for hgt in (0.235, 0.17, 0.135):
            xa, xb = sorted((xi + sx * 0.025, xo - sx * 0.025))
            m.add(boxr(xa, y, 0.35, xb, y + hgt, 0.366, 0.006), W)
            m.add(boxr(xa + 0.03, y + 0.03, 0.366, xb - 0.03, y + hgt - 0.03, 0.372, 0.008), W)  # raised panel
            cx = (xa + xb) / 2
            m.add(bail_pull(0.075).t_(cx, y + hgt * 0.55, 0.372), BR)
            y += hgt + 0.015
        m.add(boxr(-0.006, -0.012, 0, 0.006, 0.012, 0.003, 0.001).t_((xi + xo) / 2, 0.615, 0.366), BR)  # escutcheon
    # centre: frieze drawer, modesty panel
    m.add(boxr(-0.36, 0.585, -0.37, 0.36, 0.672, 0.35), W)
    m.add(boxr(-0.335, 0.592, 0.35, 0.335, 0.666, 0.366, 0.006), W)
    for sx in (-1, 1):
        m.add(knob(0.013, 0.022).t_(sx * 0.16, 0.629, 0.366), BR)
    m.add(boxr(-0.36, 0.06, -0.37, 0.36, 0.585, -0.35), W)
    # papers (one-shot paper texture), a closed letter and a quill
    rng = random.Random(3)
    for i, (x, z, a, w, d) in enumerate(((-0.12, 0.08, 8, 0.21, 0.297), (-0.05, 0.12, -14, 0.21, 0.297), (0.2, -0.02, 25, 0.16, 0.22))):
        sh = extrude(rect2d(w, d), 0.0015, uv01=True).rx(-90).ry(a).t_(x, 0.72 + 0.00075 + i * 0.0016, z)
        m.add(sh, P)
    # inkwell
    ink = lathe([(0, 0), (0.045, 0), (0.046, 0.006), (0.04, 0.01), (0.034, 0.012), (0.036, 0.03), (0.03, 0.038),
                 (0.014, 0.041), (0.012, 0.047), (0.016, 0.049), (0.012, 0.052), (0.0, 0.052)], 18, smooth=40)
    m.add(ink.t_(0.47, 0.72, 0.14), BR)
    m.add(sphere(0.008, 10, 6).t_(0.47, 0.775 - 0.008, 0.14), BR)
    # quill lying beside the inkwell: feather vane + brass nib
    vane = superellipsoid(0.018, 0.003, 0.11, 0.9, 0.9, 10, 6).t_(0, 0, 0.06)
    shaft = tube([(0, 0, -0.06), (0, 0, 0.17)], 0.0025, 6)
    nib = lathe([(0, 0), (0.003, 0), (0, 0.025)], 6).rx(-90).t_(0, 0, -0.06)
    q = merged(vane, shaft).ry(-55).t_(0.34, 0.726, 0.2)
    m.add(q, P)
    m.add(nib.ry(-55).t_(0.34, 0.726, 0.2), BR)
    # letter opener
    m.add(boxr(-0.004, 0, -0.09, 0.004, 0.004, 0.09, 0.001).ry(70).t_(-0.45, 0.72, 0.18), BR)


@model("lib_armchair", (0.9, 1.15, 0.9))
def m_armchair(m):
    V, Wd = "velvet_red", "wood_dark"
    for sx in (-1, 1):
        front = lathe([(0, 0), (0.02, 0), (0.024, 0.012), (0.017, 0.03), (0.028, 0.07), (0.031, 0.1), (0.024, 0.125),
                       (0.03, 0.135), (0.03, 0.16), (0, 0.16)], 14, smooth=40)
        m.add(front.t_(sx * 0.36, 0, 0.33), Wd)
        back = lathe([(0, 0), (0.018, 0), (0.026, 0.16), (0, 0.16)], 10).rx(-8)
        m.add(back.t_(sx * 0.36, 0, -0.33), Wd)
    m.add(boxr(-0.4, 0.12, 0.365, 0.4, 0.17, 0.425, 0.012), Wd)                 # front rail
    m.add(boxr(-0.345, 0.14, -0.38, 0.345, 0.36, 0.405, 0.03), V)               # seat base
    m.add(superellipsoid(0.335, 0.065, 0.36, 0.22, 0.25, 24, 10).t_(0, 0.415, 0.035), V)  # cushion
    # piping on cushion front edge
    m.add(tube([(-0.3, 0.415, 0.395), (0.3, 0.415, 0.395)], 0.008, 6), V)

    def tilt(p):
        return p.t_(0, -0.36, 0.33).rx(-6).t_(0, 0.36, -0.33)

    arch = [(0.355 * math.cos(math.radians(a)), 0.98 + 0.17 * math.sin(math.radians(a))) for a in range(0, 181, 10)]
    back_shape = [(-0.355, 0.33), (0.355, 0.33)] + arch
    m.add(tilt(extrude(back_shape, 0.12, smooth=40).t_(0, 0, -0.32)), V)
    m.add(tilt(superellipsoid(0.29, 0.27, 0.05, 0.25, 0.3, 22, 12).t_(0, 0.73, -0.255)), V)
    for row in range(3):
        for col in range(3 if row % 2 == 0 else 2):
            x = (col - (1 if row % 2 == 0 else 0.5)) * 0.17
            y = 0.6 + row * 0.14
            m.add(tilt(sphere(0.012, 8, 6).s(1, 1, 0.6).t_(x, y, -0.205)), V)
    # wings
    wing = [(-0.43, 0.58), (0.0, 0.58), (0.06, 0.64), (0.095, 0.74), (0.1, 0.85), (0.075, 0.95), (0.01, 1.01),
            (-0.1, 1.045), (-0.28, 1.065), (-0.43, 1.07)]
    for sx in (-1, 1):
        w = extrude(wing, 0.07, smooth=40).ry(-90)
        w.t_(0, 0, 0.43).ry(sx * 4).t_(sx * 0.372, 0, -0.43)   # splay outward about the back edge
        m.add(w, V)
        # arm panel, rolled arm, scroll front with wooden boss
        m.add(boxr(sx * 0.335, 0.14, -0.37, sx * 0.425, 0.6, 0.405, 0.015), V)
        m.add(cyl(0.064, 0.8, 18).rx(90).t_(sx * 0.385, 0.6, -0.39), V)
        scroll = [(0.05, 0.14)] + [(0.068 * math.cos(math.radians(a)), 0.6 + 0.068 * math.sin(math.radians(a)))
                                   for a in range(-45, 226, 15)] + [(-0.05, 0.14)]
        m.add(extrude(scroll, 0.03, smooth=40, z0=0.405).t_(sx * 0.385, 0, 0), V)
        m.add(lathe([(0, 0), (0.03, 0), (0.026, 0.012), (0.012, 0.02), (0, 0.02)], 14, smooth=50).rx(90).t_(sx * 0.385, 0.6, 0.435), Wd)
        m.add(tube([(sx * 0.385, 0.16, 0.42), (sx * 0.385, 0.53, 0.42)], 0.007, 6), V)


@model("lib_fireplace", (2.2, 1.6, 0.6), "wall")
def m_fireplace(m):
    ST, WD, BK = "stone", "wood_dark", "brick_soot"
    rng = random.Random(5)
    m.add(boxr(-1.02, 0.0, 0.0, 1.02, 0.06, 0.6, 0.012), ST)                  # hearth slab
    m.add(boxr(-0.62, 0.06, 0.42, 0.62, 0.075, 0.58, 0.006), ST)              # hearth front stone
    # jambs of stacked stone blocks
    for sx in (-1, 1):
        x_in, x_out = sx * 0.62, sx * 1.02
        m.add(boxr(sx * 0.6, 0.06, 0.0, sx * 1.05, 0.19, 0.46, 0.012), ST)     # base block
        y = 0.19
        row = 0
        while y < 0.93:
            hgt = min(rng.uniform(0.15, 0.19), 0.95 - y)
            jitter = rng.uniform(-0.006, 0.006)
            if row % 2 == 0:
                m.add(boxr(x_in, y + 0.004, 0.0, x_out, y + hgt - 0.004, 0.42 + jitter, 0.014), ST)
            else:
                split = sx * rng.uniform(0.78, 0.86)
                m.add(boxr(x_in, y + 0.004, 0.0, split - sx * 0.004, y + hgt - 0.004, 0.42 + jitter, 0.014), ST)
                m.add(boxr(split + sx * 0.004, y + 0.004, 0.0, x_out, y + hgt - 0.004, 0.42 - jitter, 0.014), ST)
            # mortar core behind the joints
            m.add(boxr(x_in + sx * 0.01, y - 0.004, 0.0, x_out - sx * 0.01, y + hgt + 0.004, 0.405), ST)
            y += hgt
            row += 1
    # flat arch: voussoirs with a raised keystone
    edges = [-0.62, -0.38, -0.14, 0.14, 0.38, 0.62]
    for i in range(5):
        xa, xb = edges[i] + 0.004, edges[i + 1] - 0.004
        if i == 2:
            m.add(boxr(xa, 0.9, 0.0, xb, 1.15, 0.455, 0.016), ST)
        else:
            m.add(boxr(xa, 0.95, 0.0, xb, 1.12, 0.425, 0.014), ST)
    m.add(boxr(-1.02, 0.945, 0.0, 1.02, 1.125, 0.405), ST)                    # backing behind the arch
    m.add(boxr(-1.02, 0.93, 0.0, -0.62, 1.12, 0.42, 0.014), ST)
    m.add(boxr(0.62, 0.93, 0.0, 1.02, 1.12, 0.42, 0.014), ST)
    # wooden mantel: frieze, corbels, shelf
    m.add(boxr(-1.06, 1.12, 0.0, 1.06, 1.2, 0.46, 0.01), WD)
    m.add(boxr(-1.08, 1.185, 0.0, 1.08, 1.205, 0.5, 0.006), WD)
    m.add(boxr(-1.1, 1.2, 0.0, 1.1, 1.27, 0.55, 0.012), WD)
    corbel = [(0.0, 0.0), (0.03, 0.0), (0.04, 0.04), (0.07, 0.08), (0.1, 0.12), (0.12, 0.16), (0.12, 0.2), (0.0, 0.2)]
    for sx in (-1, 1):
        for xc in (0.86, 0.68):
            c = extrude(corbel, 0.06, smooth=35).ry(-90).t_(sx * xc, 1.0, 0.42)
            m.add(c, WD)
    # overmantel stone panel with recessed centre and cap
    m.add(boxr(-0.96, 1.27, 0.0, 0.96, 1.55, 0.1, 0.01), ST)
    m.add(boxr(-0.7, 1.3, 0.1, 0.7, 1.52, 0.115, 0.01), ST)
    m.add(boxr(-0.6, 1.32, 0.115, 0.6, 1.5, 0.122, 0.006), ST)
    m.add(boxr(-1.0, 1.54, 0.0, 1.0, 1.6, 0.15, 0.012), ST)
    # firebox: sooty back, splayed cheeks, ceiling, floor
    m.add(boxr(-0.5, 0.06, 0.0, 0.5, 0.95, 0.05), BK)
    for sx in (-1, 1):
        a = math.degrees(math.atan2(0.12, 0.37))
        cheek = boxr(-0.015, 0.06, 0.0, 0.015, 0.95, math.hypot(0.12, 0.37)).ry(sx * a).t_(sx * 0.5, 0, 0.05)
        m.add(cheek, BK)
    m.add(boxr(-0.62, 0.93, 0.0, 0.62, 0.955, 0.42), BK)
    m.add(boxr(-0.6, 0.06, 0.05, 0.6, 0.07, 0.42), BK)
    # log pile (glowing embers) and ember bed
    EM = ("embers", KE_EMBERS)
    for x, z, y, a, r, L in ((-0.02, 0.17, 0.12, 8, 0.055, 0.66), (0.03, 0.29, 0.12, -6, 0.05, 0.62), (0.0, 0.23, 0.2, 20, 0.045, 0.5)):
        log = cyl(r, L, 12, -L / 2).rz(90).ry(a).t_(x, y - 0.01, z)
        m.add(log, *EM)
    for _ in range(26):
        x = rng.uniform(-0.38, 0.38); z = rng.uniform(0.1, 0.38); s = rng.uniform(0.02, 0.045)
        ch = boxr(-s, 0, -s * 0.8, s, s * 0.7, s * 0.8, s * 0.25).ry(rng.uniform(0, 90)).t_(x, 0.07, z)
        m.add(ch, *EM)
    # flames (emissive tear drops, flattened, leaning)
    for x, h, lean in ((-0.25, 0.26, 10), (-0.12, 0.38, 4), (0.0, 0.46, -2), (0.12, 0.36, -6), (0.25, 0.25, -10), (-0.05, 0.3, 14), (0.07, 0.28, -12)):
        f = flame(h, h * 0.24, 12).s(1, 1, 0.6).rz(lean).t_(x, 0.14, 0.23 + rng.uniform(-0.03, 0.03))
        m.add(f, "flame", KE_FLAME)


@model("lib_clock", (0.6, 2.3, 0.4))
def m_clock(m):
    W, BR = "wood_dark", "brass"
    m.add(boxr(-0.3, 0.0, -0.2, 0.3, 0.07, 0.2, 0.012), W)                    # skirt
    m.add(boxr(-0.27, 0.07, -0.175, 0.27, 0.38, 0.175, 0.006), W)             # plinth
    m.add(boxr(-0.19, 0.12, 0.175, 0.19, 0.33, 0.19, 0.01), W)                # plinth panel
    m.add(boxr(-0.13, 0.17, 0.19, 0.13, 0.28, 0.198, 0.008), W)
    m.add(boxr(-0.285, 0.38, -0.188, 0.285, 0.42, 0.188, 0.01), W)            # plinth cap
    # waist: back, sides, front door frame with an open window to the pendulum
    m.add(boxr(-0.2, 0.42, -0.14, 0.2, 1.5, -0.12), W)
    for sx in (-1, 1):
        m.add(boxr(sx * 0.18, 0.42, -0.14, sx * 0.2, 1.5, 0.13), W)
        m.add(boxr(sx * 0.18, 0.44, 0.12, sx * 0.12, 1.48, 0.15, 0.005), W)   # door stiles
        m.add(lathe([(0, 0), (0.017, 0), (0.017, 0.02), (0.012, 0.03), (0.013, 1.0), (0.017, 1.01), (0.017, 1.04), (0, 1.04)],
                    12, smooth=40).t_(sx * 0.2, 0.44, 0.135), W)              # quarter columns
    m.add(boxr(-0.18, 0.42, -0.12, 0.18, 0.44, 0.13), W)                      # floor of waist
    m.add(boxr(-0.18, 0.44, 0.12, 0.18, 0.62, 0.15, 0.005), W)                # door bottom rail
    # door top rail with an arched underside (listed CCW)
    rail = [(-0.18, 1.28), (-0.12, 1.28)] + [(0.12 * math.cos(math.radians(180 - a)), 1.3 + 0.06 * math.sin(math.radians(180 - a))) for a in range(0, 181, 15)] + [(0.12, 1.28), (0.18, 1.28), (0.18, 1.48), (-0.18, 1.48)]
    m.add(extrude(rail, 0.03, z0=0.12), W)
    m.add(boxr(-0.007, -0.016, 0, 0.007, 0.016, 0.004, 0.001).t_(0.15, 1.0, 0.15), BR)  # keyhole
    # pendulum and weights inside the waist
    m.add(boxr(-0.005, 0.84, -0.03, 0.005, 1.48, -0.024), BR)
    m.add(lathe([(0, 0), (0.065, 0), (0.065, 0.004), (0.05, 0.01), (0, 0.012)], 24, smooth=50).rx(90).t_(0, 0.8, -0.034), BR)
    for sx in (-1, 1):
        m.add(boxr(-0.002, 1.25, -0.002, 0.002, 1.48, 0.002).t_(sx * 0.085, 0, -0.06), BR)
        m.add(lathe([(0, 0), (0.026, 0), (0.026, 0.16), (0.02, 0.17), (0.004, 0.172), (0, 0.172)], 14, smooth=40).t_(sx * 0.085, 1.08, -0.06), BR)
    # hood
    m.add(boxr(-0.27, 1.5, -0.17, 0.27, 1.56, 0.18, 0.01), W)
    m.add(boxr(-0.25, 1.56, -0.16, 0.25, 2.04, 0.15), W)
    m.add(boxr(-0.25, 1.56, 0.15, -0.19, 2.04, 0.168, 0.004), W)
    m.add(boxr(0.19, 1.56, 0.15, 0.25, 2.04, 0.168, 0.004), W)
    m.add(boxr(-0.19, 1.56, 0.15, 0.19, 1.61, 0.168, 0.004), W)
    m.add(boxr(-0.19, 1.97, 0.15, 0.19, 2.04, 0.168, 0.004), W)
    face = Part(uv01=True)
    add_poly(face, [(-0.185, 1.615, 0.156), (0.185, 1.615, 0.156), (0.185, 1.965, 0.156), (-0.185, 1.965, 0.156)], (0, 0, 1),
             uvs=[(0.03, 0.03), (0.97, 0.03), (0.97, 0.97), (0.03, 0.97)])
    m.add(face, "clock_face")
    for sx in (-1, 1):
        col = lathe([(0, 0), (0.024, 0), (0.024, 0.02), (0.017, 0.03), (0.019, 0.44), (0.025, 0.45), (0.025, 0.48), (0, 0.48)], 12, smooth=40)
        m.add(col.t_(sx * 0.23, 1.56, 0.17), W)
        for y in (1.585, 2.02):
            m.add(cyl(0.021, 0.014, 12, y - 0.007).t_(sx * 0.23, 0, 0.17), BR)
    m.add(boxr(-0.29, 2.04, -0.19, 0.29, 2.09, 0.2, 0.012), W)               # cornice
    m.add(boxr(-0.275, 2.03, -0.18, 0.275, 2.045, 0.19, 0.005), W)
    ped = [(-0.25, 0.0), (0.25, 0.0)] + [(0.25 * math.cos(math.radians(a)), 0.13 * math.sin(math.radians(a))) for a in range(10, 171, 10)]
    m.add(extrude(ped, 0.3, smooth=40).t_(0, 2.09, -0.005), W)
    m.add(extrude(ped, 0.03, smooth=40).s(1.06, 1.1, 1).t_(0, 2.085, 0.15), W)   # pediment moulding
    fin = [(0, 0), (0.02, 0), (0.024, 0.01), (0.012, 0.02), (0.02, 0.035), (0.022, 0.045), (0.015, 0.058), (0.006, 0.07), (0.0, 0.085)]
    for sx in (-1, 1):
        m.add(lathe(fin, 12, smooth=50).s(0.9).t_(sx * 0.255, 2.09, 0.16), BR)
    m.add(lathe(fin, 12, smooth=50).t_(0, 2.215, 0.0), BR)


@model("lib_globe", (0.7, 1.15, 0.7))
def m_globe(m):
    WL, BR = "wood_light", "brass"
    cy = 0.82
    tilt = 23.5
    globe = sphere(0.275, 36, 22, uv01=True).ry(180).rz(tilt).t_(0, cy, 0)
    m.add(globe, "world_map")
    # meridian ring (in the XY plane, tilted with the axis) + axis pins
    mer = lathe([(0.29, -0.01), (0.302, -0.01), (0.302, 0.01), (0.29, 0.01), (0.29, -0.01)], 48).rx(90).rz(tilt).t_(0, cy, 0)
    m.add(mer, BR)
    ax = (-math.sin(math.radians(tilt)), math.cos(math.radians(tilt)), 0)
    for sg in (-1, 1):
        p = v_add((0, cy, 0), v_mul(ax, sg * 0.288))
        m.add(sphere(0.012, 10, 6).t_(*p), BR)
    # horizon ring held by four curved legs
    m.add(lathe([(0.3, cy - 0.012), (0.345, cy - 0.012), (0.345, cy + 0.006), (0.3, cy + 0.006), (0.3, cy - 0.012)], 48), WL)
    m.add(torus(0.3225, 0.004, 48, 6).t_(0, cy + 0.007, 0), BR)
    for k in range(4):
        a = 45 + 90 * k
        leg = bezier((0.05, 0.16, 0), (0.25, 0.2, 0), (0.33, 0.45, 0), (0.322, cy - 0.012, 0), 14)
        m.add(tube(leg, 0.016, 8, True, [1.2 - 0.4 * i / 14 for i in range(15)]).ry(a + 90), WL)
    # base, column and cradle
    m.add(lathe([(0, 0), (0.29, 0), (0.3, 0.012), (0.29, 0.03), (0.25, 0.04), (0.24, 0.055), (0.1, 0.06), (0.0, 0.06)], 40, smooth=40), WL)
    col = lathe([(0, 0.06), (0.08, 0.06), (0.08, 0.08), (0.06, 0.1), (0.045, 0.14), (0.06, 0.2), (0.065, 0.24), (0.05, 0.29),
                 (0.03, 0.33), (0.028, 0.42), (0.04, 0.44), (0.04, 0.46), (0.02, 0.48), (0.0, 0.48)], 20, smooth=40)
    m.add(col, WL)
    m.add(lathe([(0, 0.46), (0.02, 0.46), (0.03, 0.5), (0.028, 0.53), (0.0, 0.53)], 12, smooth=40), BR)
    # compass rose stretchers between the legs
    m.add(torus(0.16, 0.009, 32, 6).t_(0, 0.22, 0), WL)
    # finial on top of the meridian
    m.add(sphere(0.011, 10, 6).t_(0, cy + 0.302 + 0.011, 0), BR)


@model("lib_candelabra", (0.45, 0.6, 0.15))
def m_candelabra(m):
    BR, WX = "brass", "wax"
    base = [(0, 0), (0.072, 0), (0.074, 0.006), (0.066, 0.014), (0.046, 0.024), (0.03, 0.04), (0.02, 0.062), (0.016, 0.09),
            (0.023, 0.1), (0.016, 0.11), (0.013, 0.19), (0.02, 0.2), (0.026, 0.225), (0.018, 0.245), (0.012, 0.27),
            (0.012, 0.36), (0.0, 0.36)]
    m.add(lathe(base, 20, smooth=40), BR)
    cup = [(0, 0), (0.025, 0), (0.026, 0.004), (0.014, 0.007), (0.016, 0.03), (0.013, 0.032), (0.0, 0.032)]
    cups = [(0.0, 0.36)]
    for sx in (-1, 1):
        for xe, ye, ys, c1, c2 in ((0.1, 0.33, 0.27, (0.06, 0.2), (0.1, 0.24)), (0.2, 0.3, 0.25, (0.1, 0.16), (0.2, 0.2))):
            path = bezier((sx * 0.01, ys, 0), (sx * c1[0], c1[1], 0), (sx * c2[0], c2[1], 0), (sx * xe, ye, 0), 14)
            m.add(tube(path, 0.0065, 8), BR)
            cups.append((sx * xe, ye))
        # decorative scroll curl under the outer arm
        curl = [(sx * (0.12 + 0.025 * math.cos(math.radians(a))), 0.2 + 0.025 * math.sin(math.radians(a)), 0) for a in range(-90, 181, 20)]
        m.add(tube(curl, 0.0035, 6), BR)
    m.add(sphere(0.018, 12, 8).t_(0, 0.26, 0), BR)
    for x, y in cups:
        m.add(lathe(cup, 16, smooth=40).t_(x, y, 0), BR)
        h = 0.175 if x == 0 else (0.13 if abs(x) < 0.15 else 0.12)
        m.add(candle(0.0115, h, 12).t_(x, y + 0.025, 0), WX)
        m.add(flame(0.042, 0.0085, 10).t_(x, y + 0.025 + h * 0.975, 0), "flame", KE_FLAME)


@model("lib_side_table", (0.6, 0.7, 0.6))
def m_side_table(m):
    W = "wood_light"
    m.add(lathe([(0, 0.665), (0.28, 0.665), (0.296, 0.672), (0.3, 0.682), (0.296, 0.693), (0.286, 0.7), (0, 0.7)], 40, smooth=40), W)
    m.add(lathe([(0.2, 0.615), (0.24, 0.615), (0.25, 0.64), (0.25, 0.665), (0.2, 0.665), (0.2, 0.615)], 40, smooth=40), W)
    col = [(0, 0.12), (0.025, 0.12), (0.04, 0.15), (0.06, 0.2), (0.065, 0.24), (0.055, 0.27), (0.065, 0.29), (0.05, 0.31),
           (0.035, 0.38), (0.03, 0.47), (0.04, 0.5), (0.045, 0.53), (0.035, 0.56), (0.045, 0.6), (0.11, 0.615), (0.11, 0.625), (0.0, 0.625)]
    m.add(lathe(col, 20, smooth=35), W)
    for k in range(3):
        path = bezier((0.04, 0.27, 0), (0.12, 0.25, 0), (0.18, 0.06, 0), (0.265, 0.03, 0), 14)
        leg = sweep(path, rect2d(0.04, 0.032), smooth=False, hint=(0, 0, 1),
                    scales=[1.0 - 0.35 * i / 14 for i in range(15)])
        m.add(leg.ry(k * 120 + 90), W)
        foot = superellipsoid(0.03, 0.017, 0.022, 0.6, 0.6, 12, 8).t_(0.265, 0.017, 0).ry(k * 120 + 90)
        m.add(foot, W)


@model("lib_rug", (3.0, 0.02, 2.0))
def m_lib_rug(m):
    C = "carpet_persian"
    body = extrude(rect2d(2.9, 2.0), 0.02, uv01=True, z0=0.0).rx(-90)
    for i, n in enumerate(body.n):
        if abs(n[1]) < 0.5:
            body.t[i] = (0.5, 0.012)
    m.add(body, C)
    rng = random.Random(9)
    fr = Part(uv01=True)
    for sx in (-1, 1):
        for k in range(64):
            z = -0.98 + 1.96 * k / 63
            L = rng.uniform(0.04, 0.05)
            t = boxr(0, 0.0, -0.004, L, 0.004, 0.004).ry(rng.uniform(-8, 8))
            if sx < 0:
                t.ry(180)
            fr.merge(t.t_(sx * 1.448, 0, z))
    fr.t = [(0.004, 0.5) for _ in fr.t]
    m.add(fr, C)


@model("lib_portrait", (1.0, 1.3, 0.08), "wall")
def m_portrait(m):
    BR = "brass"
    prof = [(0, 0), (0, 0.045), (0.008, 0.058), (0.018, 0.064), (0.028, 0.058), (0.034, 0.048), (0.045, 0.05),
            (0.056, 0.072), (0.07, 0.08), (0.084, 0.073), (0.094, 0.058), (0.104, 0.05), (0.11, 0.036), (0.12, 0.03), (0.12, 0.0)]
    m.add(mitred_frame(1.0, 1.3, prof), BR)
    canvas = Part(uv01=True)
    cw, ch = 0.38, 0.53
    add_poly(canvas, [(-cw, -ch, 0.028), (cw, -ch, 0.028), (cw, ch, 0.028), (-cw, ch, 0.028)], (0, 0, 1),
             uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    add_poly(canvas, [(-cw, -ch, 0.004), (-cw, ch, 0.004), (cw, ch, 0.004), (cw, -ch, 0.004)], (0, 0, -1),
             uvs=[(0, 0), (0, 1), (1, 1), (1, 0)])
    m.add(canvas, "portrait")
    for sx in (-1, 1):
        for sy in (-1, 1):
            m.add(sphere(0.018, 12, 8).s(1, 1, 0.5).t_(sx * (0.5 - 0.07), sy * (0.65 - 0.07), 0.08 - 0.009), BR)
    m.add(superellipsoid(0.07, 0.035, 0.012, 0.6, 0.6, 14, 8).t_(0, 0.65 - 0.06, 0.068), BR)  # cartouche
    m.add(superellipsoid(0.05, 0.025, 0.01, 0.6, 0.6, 14, 8).t_(0, -0.65 + 0.06, 0.07), BR)
    for k in range(2):  # small rosettes mid-sides
        sx = -1 if k == 0 else 1
        m.add(sphere(0.013, 10, 6).s(1, 1, 0.5).t_(sx * (0.5 - 0.07), 0, 0.074), BR)
    # m.add y offset: centre is at origin; finalize moves bottom to y = 0, back to z = 0


@model("lib_bust", (0.45, 1.65, 0.45))
def m_bust(m):
    MB = "marble"
    m.add(boxr(-0.225, 0.0, -0.225, 0.225, 0.06, 0.225, 0.01), MB)
    m.add(boxr(-0.2, 0.06, -0.2, 0.2, 0.11, 0.2, 0.012), MB)
    col = [(0, 0.11), (0.15, 0.11), (0.15, 0.13), (0.135, 0.145), (0.14, 0.16), (0.12, 0.18), (0.112, 0.2), (0.1, 0.9),
           (0.11, 0.91), (0.11, 0.93), (0.1, 0.94), (0.13, 0.97), (0.0, 0.97)]
    m.add(lathe(col, 32, smooth=35), MB)
    m.add(boxr(-0.18, 0.97, -0.18, 0.18, 1.02, 0.18, 0.01), MB)
    m.add(boxr(-0.19, 1.02, -0.19, 0.19, 1.05, 0.19, 0.008), MB)
    # bust: socle, chest, shoulders, neck, head
    m.add(lathe([(0, 1.05), (0.075, 1.05), (0.075, 1.065), (0.06, 1.075), (0.045, 1.1), (0.05, 1.13), (0.07, 1.14),
                 (0.07, 1.155), (0.0, 1.155)], 20, smooth=40), MB)
    m.add(boxr(-0.05, 1.08, 0.04, 0.05, 1.12, 0.068, 0.005), MB)  # name plate
    m.add(superellipsoid(0.19, 0.13, 0.105, 0.55, 0.75, 24, 12).t_(0, 1.29, 0.0), MB)
    for sx in (-1, 1):
        m.add(superellipsoid(0.08, 0.06, 0.085, 0.7, 0.8, 14, 8).t_(sx * 0.125, 1.375, -0.01), MB)
    sash = bezier((-0.17, 1.4, 0.03), (-0.05, 1.33, 0.13), (0.05, 1.25, 0.12), (0.16, 1.2, 0.05), 14)
    m.add(sweep(sash, rect2d(0.05, 0.018), smooth=True, hint=(0, 0, 1)), MB)
    m.add(cyl(0.045, 0.12, 16, 1.38), MB)
    m.add(superellipsoid(0.074, 0.1, 0.087, 0.9, 0.9, 20, 14).t_(0, 1.545, 0.01), MB)
    m.add(superellipsoid(0.055, 0.045, 0.055, 0.9, 0.9, 14, 8).t_(0, 1.485, 0.035), MB)
    m.add(superellipsoid(0.012, 0.025, 0.015, 0.7, 0.7, 10, 8).rx(-18).t_(0, 1.535, 0.092), MB)
    for sx in (-1, 1):
        m.add(superellipsoid(0.011, 0.025, 0.018, 0.9, 0.9, 10, 6).t_(sx * 0.074, 1.54, 0.0), MB)
        m.add(boxr(-0.017, -0.003, -0.004, 0.017, 0.003, 0.004, 0.002).rz(-sx * 8).t_(sx * 0.028, 1.583, 0.088), MB)
    cap = [(0.082 * math.cos(math.radians(a)), 0.082 * math.sin(math.radians(a))) for a in range(-30, 91, 8)] + [(0.0, 0.082)]
    m.add(lathe(cap, 20, smooth=60).s(0.98, 1.05, 1.12).rx(-25).t_(0, 1.565, 0.0), MB)
    rng = random.Random(4)
    for k in range(16):  # curls
        a = math.radians(-100 + k * 200 / 15)
        x, z = 0.082 * math.sin(a), 0.09 * math.cos(a) - 0.005
        y = 1.6 + 0.012 * math.sin(k * 1.7)
        if z > 0.03:
            y += 0.01
        m.add(sphere(0.02 + rng.uniform(0, 0.006), 10, 6).t_(x, y, z), MB)
    for k in range(5):
        m.add(sphere(0.022, 10, 6).t_(-0.04 + 0.02 * k, 1.635, -0.02 + 0.01 * (k % 2)), MB)


@model("lib_chandelier", (1.0, 0.9, 1.0), "hang")
def m_chandelier(m):
    IR, BR, WX = "iron", "brass", "wax"
    m.add(lathe([(0, -0.055), (0.03, -0.055), (0.06, -0.04), (0.08, -0.018), (0.085, 0.0), (0.0, 0.0)], 24, smooth=40), IR)
    m.add(torus(0.012, 0.0035, 12, 6).rx(90).t_(0, -0.066, 0), IR)
    y = -0.075
    k = 0
    while y > -0.33:
        link = torus(0.012, 0.003, 12, 6).s(1, 1, 1.7).rx(90)
        if k % 2:
            link.ry(90)
        m.add(link.t_(0, y - 0.018, 0), IR)
        y -= 0.03
        k += 1
    stem = [(0, -0.9), (0.006, -0.895), (0.018, -0.87), (0.03, -0.84), (0.032, -0.82), (0.02, -0.8), (0.03, -0.78),
            (0.08, -0.74), (0.095, -0.71), (0.09, -0.68), (0.06, -0.66), (0.03, -0.64), (0.025, -0.6), (0.035, -0.58),
            (0.022, -0.56), (0.018, -0.46), (0.03, -0.44), (0.045, -0.41), (0.03, -0.38), (0.014, -0.36), (0.014, -0.33), (0.0, -0.33)]
    m.add(lathe(stem, 24, smooth=35), BR)
    narms = 8
    cup = [(0, 0), (0.045, 0), (0.047, 0.005), (0.03, 0.01), (0.02, 0.012), (0.022, 0.04), (0.017, 0.042), (0.0, 0.042)]
    for i in range(narms):
        a = 360.0 * i / narms
        arm = bezier((0.08, -0.72, 0), (0.22, -0.8, 0), (0.42, -0.76, 0), (0.45, -0.64, 0), 16)
        m.add(tube(arm, 0.009, 8).ry(a + 90), IR)
        curl = [(0.2 + 0.035 * math.cos(math.radians(t)), -0.7 + 0.035 * math.sin(math.radians(t)), 0) for t in range(-120, 200, 20)]
        m.add(tube(curl, 0.004, 6).ry(a + 90), IR)
        m.add(lathe(cup, 16, smooth=40).t_(0.45, -0.64, 0).ry(a + 90), BR)
        m.add(candle(0.014, 0.12, 12).t_(0.45, -0.64 + 0.04, 0).ry(a + 90), WX)
        m.add(flame(0.05, 0.01, 10).t_(0.45, -0.64 + 0.04 + 0.117, 0).ry(a + 90), "flame", KE_FLAME)
    m.add(torus(0.28, 0.006, 48, 6).t_(0, -0.765, 0), IR)
    m.add(torus(0.12, 0.005, 32, 6).t_(0, -0.47, 0), IR)
    for i in range(narms):  # upper spokes from the stem collar to the upper ring
        a = 360.0 * i / narms + 22.5
        m.add(tube(bezier((0.03, -0.42, 0), (0.07, -0.4, 0), (0.1, -0.44, 0), (0.12, -0.47, 0), 8), 0.004, 6).ry(a), IR)


@model("bed_bed", (1.8, 2.1, 2.2))
def m_bed(m):
    W, LN, QT = "wood_dark", "linen", "quilt"
    post = [(0, 0), (0.04, 0), (0.042, 0.06), (0.03, 0.09), (0.035, 0.15), (0.032, 0.25), (0.04, 0.27), (0.04, 0.5),
            (0.03, 0.52), (0.026, 0.62), (0.034, 0.72), (0.04, 0.78), (0.03, 0.86), (0.022, 0.95), (0.022, 1.7),
            (0.03, 1.74), (0.024, 1.78), (0.022, 1.93), (0.03, 1.95), (0.03, 2.0), (0.02, 2.01), (0.03, 2.035),
            (0.033, 2.06), (0.02, 2.085), (0.006, 2.098), (0.0, 2.1)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            x, z = sx * 0.86, sz * 1.06
            m.add(lathe(post, 16, smooth=40).t_(x, 0, z), W)
            m.add(boxr(x - 0.04, 0.27, z - 0.04, x + 0.04, 0.5, z + 0.04, 0.006), W)
    # rails
    for sx in (-1, 1):
        m.add(boxr(sx * 0.82, 0.3, -1.02, sx * 0.86, 0.48, 1.02, 0.006), W)
    for sz in (-1, 1):
        m.add(boxr(-0.82, 0.3, sz * 1.04, 0.82, 0.48, sz * 1.08, 0.006), W)
    # canopy frame
    for sx in (-1, 1):
        m.add(boxr(sx * 0.84, 1.95, -1.02, sx * 0.88, 2.0, 1.02, 0.005), W)
    for sz in (-1, 1):
        m.add(boxr(-0.82, 1.95, sz * 1.04, 0.82, 2.0, sz * 1.08, 0.005), W)
    # headboard with arched top, crest moulding and raised panels
    arch = [(0.82 * math.cos(math.radians(a)), 1.25 + 0.22 * math.sin(math.radians(a))) for a in range(0, 181, 9)]
    head = [(-0.82, 0.48), (0.82, 0.48)] + arch
    m.add(extrude(head, 0.05, smooth=30).t_(0, 0, -1.06), W)
    crest = [(0.81 * math.cos(math.radians(a)), 1.25 + 0.215 * math.sin(math.radians(a)), -1.06) for a in range(0, 181, 6)]
    m.add(tube(crest, 0.025, 8), W)
    for cx in (-0.5, 0.0, 0.5):
        pw = 0.2
        ptop = [(cx + pw * math.cos(math.radians(a)), 1.1 + 0.1 * math.sin(math.radians(a))) for a in range(0, 181, 15)]
        m.add(extrude([(cx - pw, 0.62), (cx + pw, 0.62)] + ptop, 0.012, smooth=30, z0=-1.035), W)
    # footboard
    m.add(boxr(-0.82, 0.48, 1.04, 0.82, 0.92, 1.08), W)
    m.add(boxr(-0.84, 0.92, 1.025, 0.84, 0.97, 1.095, 0.01), W)
    for cx in (-0.42, 0.42):
        m.add(boxr(cx - 0.32, 0.56, 1.08, cx + 0.32, 0.86, 1.09, 0.008), W)
        m.add(boxr(cx - 0.32, 0.56, 1.03, cx + 0.32, 0.86, 1.04, 0.008), W)
    # mattress + sheet fold + pillows
    m.add(boxr(-0.84, 0.46, -1.03, 0.84, 0.68, 1.03, 0.05), LN)
    for sx in (-1, 1):
        p = superellipsoid(0.33, 0.075, 0.2, 0.3, 0.35, 20, 10).rx(-18).t_(sx * 0.37, 0.76, -0.8)
        m.add(p, LN)
    # quilt draped over the mattress (length preserving drape around a rounded rectangle)
    top_y, hx, z0_, z1_ = 0.71, 0.83, -0.42, 1.02
    bend = 0.05

    def drape(u, v):
        X = -1.12 + 2.24 * u
        Z = z0_ + (z1_ - z0_) * v
        qx = max(-hx, min(hx, X)); qz = min(z1_, Z)
        dx, dz = X - qx, Z - qz
        d = math.hypot(dx, dz)
        if d < 1e-9:
            return (X, top_y + 0.004 * math.sin(X * 25) * math.sin(Z * 25), Z)
        ux, uz = dx / d, dz / d
        quarter = bend * math.pi / 2
        if d <= quarter:
            a = d / bend
            off, dy = bend * math.sin(a), bend * (math.cos(a) - 1)
        else:
            hang = d - quarter
            wav = 0.012 * math.sin((X + Z) * 22) * min(1.0, hang / 0.1)
            off, dy = bend + wav, -bend - hang
        return (qx + ux * off, top_y + dy, qz + uz * off)

    quilt = sheet(drape, 60, 40, 0.012, uv=lambda u, v, P: (-1.12 + 2.24 * u, -(z0_ + (z1_ - z0_) * v)))
    m.add(orient_to_normals(quilt), QT)
    m.add(superellipsoid(0.86, 0.025, 0.07, 0.4, 0.3, 24, 8).t_(0, 0.715, -0.44), LN)  # turned-down sheet
    # valance hanging from the canopy rails
    def valance(x0, z0, x1, z1):
        L = math.hypot(x1 - x0, z1 - z0)
        nx, nz = (z1 - z0) / L, -(x1 - x0) / L
        def fn(u, v):
            w = 0.012 * math.sin(u * L * 40)
            return (x0 + (x1 - x0) * u + nx * w, 1.77 + 0.22 * v, z0 + (z1 - z0) * u + nz * w)
        return sheet(fn, int(L * 40), 3, 0.006)
    m.add(valance(-0.89, 1.09, 0.89, 1.09), LN)
    m.add(valance(0.89, -1.09, -0.89, -1.09), LN)
    m.add(valance(0.89, 1.09, 0.89, -1.09), LN)
    m.add(valance(-0.89, -1.09, -0.89, 1.09), LN)


@model("bed_wardrobe", (1.4, 2.2, 0.6))
def m_wardrobe(m):
    W, BR = "wood_light", "brass"
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add(bun_foot(0.045, 0.08).t_(sx * 0.6, 0, sz * 0.22), W)
    m.add(boxr(-0.68, 0.08, -0.28, 0.68, 0.2, 0.28, 0.012), W)
    m.add(boxr(-0.66, 0.2, -0.27, 0.66, 1.98, 0.25), W)
    m.add(boxr(-0.62, 0.235, 0.25, 0.62, 0.46, 0.268, 0.008), W)       # drawer
    m.add(boxr(-0.58, 0.27, 0.268, 0.58, 0.425, 0.274, 0.01), W)
    for sx in (-1, 1):
        m.add(bail_pull(0.09).t_(sx * 0.32, 0.35, 0.274), BR)
    # doors with raised panels (upper panel arched)
    for sx in (-1, 1):
        xa, xb = sorted((sx * 0.006, sx * 0.61))
        m.add(boxr(xa, 0.49, 0.25, xb, 1.95, 0.272, 0.006), W)
        cx = (xa + xb) / 2
        pw = (xb - xa) / 2 - 0.07
        m.add(boxr(cx - pw, 0.58, 0.272, cx + pw, 0.98, 0.286, 0.014), W)
        arch = [(cx + pw * math.cos(math.radians(a)), 1.7 + 0.13 * math.sin(math.radians(a))) for a in range(0, 181, 12)]
        m.add(extrude([(cx - pw, 1.06), (cx + pw, 1.06)] + arch, 0.014, smooth=30, z0=0.272), W)
        m.add(extrude([(cx - pw + 0.03, 1.09), (cx + pw - 0.03, 1.09)] +
                      [(cx + (pw - 0.03) * math.cos(math.radians(a)), 1.7 + 0.1 * math.sin(math.radians(a))) for a in range(0, 181, 12)],
                      0.008, smooth=30, z0=0.286), W)
        m.add(knob(0.017, 0.03).t_(sx * 0.05, 1.2, 0.272), BR)
        # side panels
        for y0, y1 in ((0.3, 0.95), (1.05, 1.88)):
            m.add(boxr(sx * 0.66, y0, -0.2, sx * 0.672, y1, 0.18, 0.006), W)
    m.add(extrude([(-0.012, -0.02), (0.012, -0.02), (0.012, 0.01), (0.0, 0.025), (-0.012, 0.01)], 0.004, z0=0.272).t_(0.04, 1.14, 0), BR)
    for sx in (-1, 1):
        m.add(boxr(sx * 0.612, 0.2, 0.245, sx * 0.668, 1.98, 0.282, 0.006), W)       # pilasters
        for y in (0.2, 1.93):
            m.add(boxr(sx * 0.606, y, 0.245, sx * 0.674, y + 0.05, 0.29, 0.008), W)
    # cornice
    m.add(boxr(-0.672, 1.98, -0.275, 0.672, 2.06, 0.278), W)
    m.add(boxr(-0.69, 2.06, -0.29, 0.69, 2.12, 0.29, 0.02), W)
    m.add(boxr(-0.7, 2.12, -0.3, 0.7, 2.165, 0.3, 0.01), W)
    ped = [(-0.32, 0.0), (0.32, 0.0), (0.0, 0.035)]
    m.add(extrude(ped, 0.04, z0=0.24).t_(0, 2.165, 0), W)
    m.add(boxr(-0.66, 2.0, 0.278, 0.66, 2.04, 0.286, 0.006), W)


@model("bed_rocking_chair", (0.65, 1.1, 1.0))
def m_rocking_chair(m):
    W = "wood_light"
    R = 1.3

    def rocker_y(z):
        return R - math.sqrt(R * R - z * z)

    for sx in (-1, 1):
        path = [(sx * 0.27, rocker_y(z), z) for z in [-0.5 + i / 24 for i in range(25)]]
        rk = sweep(path, [(-0.014, 0.0), (0.014, 0.0), (0.014, 0.045), (-0.014, 0.045)], smooth=False, hint=(1, 0, 0))
        m.add(rk, W)
        # front legs run up to the arm rest; back legs to the seat
        fl = lathe([(0, 0), (0.02, 0), (0.02, 0.05), (0.016, 0.07), (0.022, 0.12), (0.016, 0.2), (0.016, 0.33), (0.021, 0.36),
                    (0.021, 0.4), (0.014, 0.44), (0.017, 0.5), (0.013, 0.56), (0.015, 0.6), (0.0, 0.6)], 12, smooth=40)
        m.add(fl.t_(sx * 0.27, rocker_y(0.2) + 0.04, 0.2), W)
        bl = lathe([(0, 0), (0.02, 0), (0.02, 0.05), (0.016, 0.07), (0.022, 0.12), (0.016, 0.2), (0.02, 0.36), (0.0, 0.37)], 12, smooth=40)
        m.add(bl.t_(sx * 0.27, rocker_y(-0.2) + 0.04, -0.2), W)
        # side stretcher
        m.add(lathe([(0, 0), (0.011, 0), (0.013, 0.2), (0.011, 0.4), (0, 0.4)], 8, smooth=50).rx(90).t_(sx * 0.27, 0.2, -0.2), W)
        # arm rest with rounded front
        arm = boxr(-0.032, 0, -0.27, 0.032, 0.024, 0.22, 0.008)
        m.add(arm.t_(sx * 0.288, 0.64, 0), W)
        m.add(cyl(0.032, 0.024, 14).t_(sx * 0.288, 0.64, 0.22), W)
    m.add(lathe([(0, 0), (0.011, 0), (0.013, 0.27), (0.011, 0.54), (0, 0.54)], 8, smooth=50).rz(-90).t_(-0.27, 0.22, 0.2), W)
    m.add(boxr(-0.26, 0.39, -0.23, 0.26, 0.43, 0.25, 0.012), W)       # seat
    m.add(boxr(-0.26, 0.36, 0.2, 0.26, 0.39, 0.24), W)                 # front seat rail
    m.add(boxr(-0.26, 0.36, -0.22, 0.26, 0.39, -0.18), W)
    # back assembly built upright at the seat's rear edge then tilted back
    back = []
    for sx in (-1, 1):
        back.append(lathe([(0, 0), (0.021, 0), (0.02, 0.06), (0.016, 0.1), (0.019, 0.5), (0.016, 0.6), (0.018, 0.64),
                           (0.024, 0.67), (0.016, 0.7), (0.0, 0.72)], 12, smooth=40).t_(sx * 0.23, 0, 0))
        back.append(lathe([(0, 0), (0.014, 0), (0.018, 0.012), (0.012, 0.03), (0.0, 0.035)], 10, smooth=50).t_(sx * 0.23, 0.715, 0))
    back.append(boxr(-0.23, 0.05, -0.014, 0.23, 0.1, 0.014, 0.006))
    rail = [(-0.23, 0.52), (0.23, 0.52)] + [(0.23 * math.cos(math.radians(a)), 0.6 + 0.04 * math.sin(math.radians(a))) for a in range(0, 181, 12)]
    back.append(extrude(rail, 0.03, smooth=30))
    for i in range(7):
        x = -0.18 + 0.06 * i
        back.append(lathe([(0, 0.09), (0.008, 0.09), (0.012, 0.16), (0.01, 0.26), (0.007, 0.4), (0.009, 0.48),
                           (0.008, 0.53), (0, 0.53)], 8, smooth=40).t_(x, 0, 0))
    for p in back:
        m.add(p.s(1, 0.95, 1).rx(-12).t_(0, 0.42, -0.215), W)


@model("bed_nightstand", (0.5, 0.62, 0.42))
def m_nightstand(m):
    W, BR = "wood_painted_white", "brass"
    leg = [(0, 0), (0.016, 0), (0.018, 0.02), (0.014, 0.05), (0.02, 0.12), (0.024, 0.2), (0.02, 0.26), (0.026, 0.29),
           (0.026, 0.585), (0.0, 0.585)]
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add(lathe(leg, 12, smooth=40).t_(sx * 0.205, 0, sz * 0.16), W)
    m.add(boxr(-0.25, 0.585, -0.21, 0.25, 0.62, 0.21, 0.01), W)        # top
    m.add(boxr(-0.21, 0.3, -0.17, 0.21, 0.32, 0.17), W)                # case bottom
    m.add(boxr(-0.21, 0.43, -0.17, 0.21, 0.45, 0.17), W)                # divider
    for sx in (-1, 1):
        m.add(boxr(sx * 0.19, 0.3, -0.17, sx * 0.21, 0.585, 0.17), W)     # sides
    m.add(boxr(-0.21, 0.3, -0.17, 0.21, 0.585, -0.15), W)               # back
    m.add(boxr(-0.205, 0.455, 0.16, 0.205, 0.575, 0.182, 0.006), W)     # drawer front
    m.add(boxr(-0.17, 0.475, 0.182, 0.17, 0.555, 0.188, 0.008), W)
    m.add(knob(0.014, 0.026).t_(0, 0.515, 0.188), BR)
    m.add(boxr(-0.18, 0.08, -0.15, 0.18, 0.1, 0.15, 0.004), W)          # lower shelf
    m.add(boxr(-0.21, 0.28, 0.15, 0.21, 0.3, 0.17, 0.004), W)            # apron


@model("bed_lamp", (0.25, 0.5, 0.25))
def m_lamp(m):
    BR = "brass"
    body = [(0, 0), (0.125, 0), (0.125, 0.008), (0.11, 0.016), (0.08, 0.026), (0.05, 0.04), (0.03, 0.06), (0.024, 0.1),
            (0.034, 0.11), (0.03, 0.12), (0.06, 0.13), (0.088, 0.155), (0.1, 0.185), (0.092, 0.215), (0.065, 0.238),
            (0.04, 0.248), (0.038, 0.262), (0.042, 0.27), (0.04, 0.295), (0.0, 0.295)]
    m.add(lathe(body, 28, smooth=40), BR)
    m.add(cyl(0.008, 0.025, 10).rz(-90).t_(0.04, 0.282, 0), BR)       # wick wheel shaft
    m.add(cyl(0.012, 0.005, 12).rz(-90).t_(0.065, 0.282, 0), BR)
    m.add(torus(0.024, 0.006, 14, 6).rx(90).t_(0.1, 0.13, 0), BR)       # finger loop
    for k in range(8):  # gallery prongs
        a = 360 * k / 8
        m.add(boxr(-0.004, 0, -0.0015, 0.004, 0.035, 0.0015).rz(-15).t_(0.036, 0.288, 0).ry(a), BR)
    chim = [(0.0, 0.29), (0.034, 0.29), (0.05, 0.31), (0.06, 0.345), (0.055, 0.38), (0.035, 0.41), (0.029, 0.44),
            (0.029, 0.5), (0.025, 0.5), (0.025, 0.46), (0.0, 0.46)]
    m.add(lathe(chim, 24, smooth=40), "glass", (0.25, 0.18, 0.08))
    m.add(flame(0.06, 0.012, 10).t_(0, 0.305, 0), "flame", KE_FLAME)


@model("bed_dresser", (1.2, 1.95, 0.5))
def m_dresser(m):
    W, MR, BR = "wood_painted_white", "mirror", "brass"
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add(bun_foot(0.04, 0.06).t_(sx * 0.52, 0, sz * 0.18), W)
    m.add(boxr(-0.58, 0.06, -0.23, 0.58, 0.12, 0.225, 0.01), W)
    m.add(boxr(-0.56, 0.12, -0.22, 0.56, 0.86, 0.2), W)
    m.add(boxr(-0.6, 0.86, -0.25, 0.6, 0.9, 0.25, 0.01), W)
    for sx in (-1, 1):
        m.add(lathe([(0, 0), (0.024, 0), (0.024, 0.02), (0.018, 0.03), (0.02, 0.71), (0.024, 0.72), (0.024, 0.74), (0, 0.74)],
                    12, smooth=40).t_(sx * 0.555, 0.12, 0.2), W)
    y = 0.15
    for hgt, split in ((0.2, False), (0.18, False), (0.16, False), (0.11, True)):
        spans = [(-0.52, -0.006), (0.006, 0.52)] if split else [(-0.52, 0.52)]
        for xa, xb in spans:
            m.add(boxr(xa, y, 0.2, xb, y + hgt, 0.216, 0.006), W)
            m.add(boxr(xa + 0.025, y + 0.022, 0.216, xb - 0.025, y + hgt - 0.022, 0.221, 0.008), W)
            cx = (xa + xb) / 2
            if split:
                m.add(bail_pull(0.07).t_(cx, y + hgt / 2, 0.221), BR)
            else:
                for px in (cx - 0.25, cx + 0.25):
                    m.add(bail_pull(0.07).t_(px, y + hgt / 2, 0.221), BR)
                m.add(boxr(-0.006, -0.012, 0, 0.006, 0.012, 0.003, 0.001).t_(cx, y + hgt / 2, 0.221), BR)
        y += hgt + 0.012
    # mirror: turned uprights, arched frame (swept moulding), glass, crest
    for sx in (-1, 1):
        m.add(boxr(sx * 0.37, 0.9, -0.12, sx * 0.47, 0.96, 0.02, 0.008), W)
        m.add(lathe([(0, 0), (0.022, 0), (0.022, 0.03), (0.016, 0.05), (0.02, 0.2), (0.015, 0.5), (0.018, 0.6), (0.014, 0.66),
                     (0.02, 0.68), (0.012, 0.7), (0, 0.72)], 12, smooth=40).t_(sx * 0.42, 0.96, -0.05), W)
        m.add(lathe([(0, 0), (0.015, 0), (0.018, 0.012), (0.01, 0.024), (0, 0.03)], 10, smooth=50).rz(-sx * 90).t_(sx * 0.4, 1.4, -0.05), BR)
    outline = [(0.33, 0.98)] + [(0.33 * math.cos(math.radians(a)), 1.6 + 0.26 * math.sin(math.radians(a))) for a in range(0, 181, 9)] + [(-0.33, 0.98)]
    path = [(x, y, -0.05) for x, y in outline]
    m.add(sweep(path, rect2d(0.04, 0.05), smooth=False, hint=(0, 0, 1)), W)
    m.add(boxr(-0.355, 0.955, -0.07, 0.355, 1.005, -0.03, 0.006), W)
    glass = extrude([(x * 0.94, 0.98 + (y - 0.98) * 0.975) for x, y in outline[::-1]], 0.006, uv01=True, z0=-0.055)
    m.add(glass, MR)
    m.add(extrude([(x, y) for x, y in outline[::-1]], 0.012, z0=-0.075), W)     # backboard
    crest = [(-0.14, 0.0), (0.14, 0.0), (0.1, 0.03), (0.04, 0.045), (0.0, 0.085), (-0.04, 0.045), (-0.1, 0.03)]
    m.add(extrude(crest, 0.03, z0=-0.065).t_(0, 1.865, 0), W)


@model("bed_music_box", (0.26, 0.16, 0.18))
def m_music_box(m):
    W, BR, DR = "wood_dark", "brass", "doll_dress"
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add(sphere(0.008, 10, 6).t_(sx * 0.112, 0.008, sz * 0.062), BR)
    m.add(boxr(-0.13, 0.008, -0.08, 0.13, 0.018, 0.08, 0.003), W)
    m.add(boxr(-0.124, 0.018, -0.074, 0.124, 0.07, 0.074, 0.002), W)
    m.add(boxr(-0.13, 0.068, -0.08, 0.13, 0.076, 0.08, 0.002), W)
    m.add(boxr(-0.126, 0.076, -0.076, 0.126, 0.092, 0.076, 0.005), W)          # lid
    m.add(boxr(-0.1, 0.0915, -0.055, 0.1, 0.0935, 0.055), BR)                  # inlay plate
    m.add(boxr(-0.094, 0.0925, -0.049, 0.094, 0.0945, 0.049), W)
    for sx in (-1, 1):                                                          # corner caps
        for sz in (-1, 1):
            m.add(boxr(sx * 0.131, 0.075, sz * 0.081, sx * 0.112, 0.093, sz * 0.062, 0.002), BR)
    m.add(extrude([(-0.008, -0.012), (0.008, -0.012), (0.008, 0.004), (0.0, 0.012), (-0.008, 0.004)], 0.002, z0=0.074).t_(0, 0.045, 0), BR)
    # winding key at the back
    m.add(cyl(0.004, 0.018, 8).rx(-90).t_(0, 0.04, -0.074), BR)
    for sx in (-1, 1):
        m.add(superellipsoid(0.011, 0.009, 0.0025, 0.8, 0.8, 10, 6).t_(sx * 0.01, 0.04, -0.095), BR)
    # ballerina on a brass platform
    m.add(cyl(0.028, 0.006, 20, 0.0935), BR)
    yb = 0.0995
    m.add(tube([(0.003, yb, 0), (0.002, yb + 0.025, 0)], 0.0025, 6), BR)                              # standing leg
    m.add(tube([(-0.002, yb + 0.024, 0), (-0.01, yb + 0.012, 0.006), (-0.004, yb + 0.006, 0.01)], 0.0022, 6), BR)  # bent leg
    m.add(lathe([(0, 0), (0.024, 0.0), (0.026, 0.003), (0.01, 0.007), (0.0, 0.007)], 18, smooth=50).t_(0, yb + 0.022, 0), DR)
    m.add(superellipsoid(0.0065, 0.013, 0.005, 0.8, 0.8, 10, 8).t_(0, yb + 0.04, 0), DR)
    m.add(sphere(0.0065, 10, 8).t_(0, yb + 0.054, 0), BR)
    m.add(sphere(0.0035, 8, 5).t_(0, yb + 0.059, -0.004), BR)                                           # bun
    arms = [(0.011 * math.cos(math.radians(a)), yb + 0.047 + 0.009 * math.sin(math.radians(a)), 0.0) for a in range(-20, 201, 20)]
    m.add(tube(arms, 0.0018, 6), BR)


@model("bed_toy_chest", (0.9, 0.55, 0.5))
def m_toy_chest(m):
    W, IR = "wood_light", "iron"
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.add(boxr(sx * 0.42, 0.0, sz * 0.22, sx * 0.36, 0.03, sz * 0.16, 0.006), W)
    m.add(boxr(-0.41, 0.03, -0.22, 0.41, 0.4, 0.22, 0.004), W)
    rng = random.Random(2)
    for k in range(3):   # planks front/back/sides
        y0 = 0.035 + k * 0.122; y1 = y0 + 0.118
        for sz in (-1, 1):
            m.add(boxr(-0.41, y0, sz * 0.22, 0.41, y1, sz * (0.232 + rng.uniform(0, 0.003)), 0.004), W)
        for sx in (-1, 1):
            m.add(boxr(sx * 0.41, y0, -0.232, sx * (0.422 + rng.uniform(0, 0.003)), y1, 0.232, 0.004), W)
    # barrel lid
    lid = [(-0.24, 0.0), (0.24, 0.0), (0.24, 0.05)] + [(0.24 * math.cos(math.radians(a)), 0.05 + 0.095 * math.sin(math.radians(a))) for a in range(10, 171, 10)] + [(-0.24, 0.05)]
    m.add(extrude(lid, 0.86, smooth=30).ry(-90).t_(0, 0.405, 0), W)
    m.add(boxr(-0.43, 0.395, -0.245, 0.43, 0.415, 0.245, 0.004), W)
    # iron bands over lid and down the body
    for bx in (-0.3, 0.3):
        # follows the lid curve slightly proud of its surface
        arc = [(bx, 0.405 + 0.05 + 0.101 * math.sin(math.radians(a)), 0.246 * math.cos(math.radians(a))) for a in range(0, 181, 10)]
        band = sweep(arc, rect2d(0.045, 0.006), smooth=False, hint=(1, 0, 0), caps=True)
        m.add(band, IR)
        for sz in (-1, 1):
            m.add(boxr(bx - 0.0225, 0.03, sz * 0.232, bx + 0.0225, 0.455, sz * 0.24, 0.002), IR)
            for y in (0.08, 0.2, 0.32, 0.43):
                m.add(sphere(0.0055, 8, 5).t_(bx, y, sz * 0.241), IR)
    for sx in (-1, 1):   # corner brackets
        for sz in (-1, 1):
            for y0 in (0.03, 0.32):
                m.add(boxr(sx * 0.425, y0, sz * 0.2, sx * 0.4, y0 + 0.08, sz * 0.241, 0.002), IR)
                m.add(boxr(sx * 0.425, y0, sz * 0.24, sx * 0.383, y0 + 0.08, sz * 0.233, 0.002), IR)
        # side handles
        m.add(boxr(sx * 0.422, 0.25, -0.05, sx * 0.43, 0.29, 0.05, 0.002), IR)
        h = torus(0.04, 0.0055, 14, 6, 0, 180).ry(90).rz(-90 * sx)
        m.add(h.t_(sx * 0.432, 0.27, 0), IR)
    # hasp and padlock
    m.add(boxr(-0.025, 0.33, 0.24, 0.025, 0.44, 0.248, 0.003), IR)
    m.add(boxr(-0.028, 0.27, 0.245, 0.028, 0.325, 0.262, 0.006), IR)
    m.add(torus(0.018, 0.0045, 12, 6, 0, 180).rx(90).t_(0, 0.322, 0.253), IR)


@model("bed_doll", (0.22, 0.42, 0.25))
def m_doll(m):
    SK, DR, HA = "skin", "doll_dress", "hair"
    skirt = lathe([(0, 0.0), (0.1, 0.0), (0.104, 0.01), (0.096, 0.03), (0.078, 0.06), (0.058, 0.09), (0.046, 0.12),
                   (0.04, 0.138), (0.0, 0.138)], 24, smooth=40).s(1, 1, 1.08).t_(0, 0, -0.015)
    m.add(skirt, DR)
    m.add(torus(0.1, 0.006, 32, 6).s(1, 1, 1.08).t_(0, 0.006, -0.015), DR)  # hem ruffle
    for sx in (-1, 1):  # legs stretched forward, with shoes
        m.add(tube([(sx * 0.035, 0.024, 0.02), (sx * 0.038, 0.022, 0.1)], 0.017, 10), SK)
        m.add(superellipsoid(0.019, 0.016, 0.028, 0.6, 0.6, 10, 6).t_(sx * 0.038, 0.02, 0.108), HA)
    m.add(superellipsoid(0.045, 0.07, 0.035, 0.8, 0.8, 16, 10).t_(0, 0.195, -0.008), DR)       # bodice
    m.add(torus(0.03, 0.008, 18, 6).s(1, 0.7, 1).t_(0, 0.268, -0.005), DR)                     # collar
    m.add(cyl(0.015, 0.03, 10, 0.262), SK)
    m.add(sphere(0.056, 20, 14).t_(0, 0.335, 0.0), SK)
    for sx in (-1, 1):
        m.add(sphere(0.008, 8, 6).t_(sx * 0.02, 0.343, 0.05), HA)                              # eyes
        m.add(sphere(0.012, 8, 6).s(1, 0.7, 0.4).t_(sx * 0.03, 0.32, 0.047), DR)              # cheeks
        # puffed sleeves, forearms, hands resting on the lap
        m.add(superellipsoid(0.022, 0.025, 0.022, 0.8, 0.8, 10, 8).t_(sx * 0.052, 0.245, -0.005), DR)
        m.add(tube([(sx * 0.056, 0.232, 0.0), (sx * 0.064, 0.18, 0.02), (sx * 0.052, 0.14, 0.055)], 0.011, 8), SK)
        m.add(sphere(0.013, 10, 6).t_(sx * 0.047, 0.133, 0.066), SK)
    m.add(sphere(0.006, 8, 5).s(1.4, 0.6, 0.6).t_(0, 0.312, 0.052), DR)                        # mouth
    cap = [(0.06 * math.cos(math.radians(a)), 0.06 * math.sin(math.radians(a))) for a in range(-35, 91, 10)] + [(0.0, 0.06)]
    m.add(lathe(cap, 20, smooth=60).s(1.03, 1.05, 1.1).rx(-30).t_(0, 0.345, -0.004), HA)
    for k in range(8):  # ringlets
        a = math.radians(-110 + k * 220 / 7)
        x, z = 0.052 * math.sin(a), 0.052 * math.cos(a) - 0.012
        if z > 0.03:
            continue
        pts = [(x * (1 + 0.1 * t / 6) + 0.006 * math.cos(t * 1.6), 0.33 - 0.012 * t, z + 0.006 * math.sin(t * 1.6) - 0.004 * t) for t in range(7)]
        m.add(tube(pts, 0.008, 6, True, [1.0 - 0.06 * t for t in range(7)]), HA)
    for sx in (-1, 1):  # bow
        m.add(superellipsoid(0.022, 0.014, 0.008, 0.7, 0.7, 10, 6).rz(sx * 20).t_(sx * 0.02, 0.402, -0.01), DR)
    m.add(sphere(0.008, 8, 6).t_(0, 0.402, -0.01), DR)


@model("bed_rug", (2.4, 0.02, 2.4))
def m_bed_rug(m):
    C = "carpet_round"
    body = extrude(circle2d(1.2, 72), 0.02, uv01=True, z0=0.0).rx(-90)
    for i, n in enumerate(body.n):
        if abs(n[1]) < 0.5:
            body.t[i] = (0.5, 0.004)
    m.add(body, C)
    rim = torus(1.19, 0.01, 72, 6)
    rim.uv01 = True
    rim.t = [(0.5, 0.008)] * len(rim.t)
    m.add(rim.t_(0, 0.01, 0), C)


@model("bed_window", (1.6, 2.2, 0.2), "wall")
def m_window(m):
    W, CU = "wood_painted_white", "curtain"
    # casing and sill
    for sx in (-1, 1):
        m.add(boxr(sx * 0.5, 0.74, 0.0, sx * 0.6, 2.06, 0.03, 0.006), W)
    m.add(boxr(-0.64, 2.04, 0.0, 0.64, 2.11, 0.035, 0.006), W)
    m.add(boxr(-0.67, 2.1, 0.0, 0.67, 2.125, 0.05, 0.008), W)
    m.add(boxr(-0.62, 0.74, 0.0, 0.62, 0.79, 0.12, 0.008), W)
    m.add(boxr(-0.56, 0.66, 0.0, 0.56, 0.74, 0.025, 0.005), W)
    # sash frame and muntins
    m.add(boxr(-0.5, 0.79, 0.0, -0.45, 2.04, 0.045, 0.004), W)
    m.add(boxr(0.45, 0.79, 0.0, 0.5, 2.04, 0.045, 0.004), W)
    m.add(boxr(-0.5, 0.79, 0.0, 0.5, 0.85, 0.045, 0.004), W)
    m.add(boxr(-0.5, 1.98, 0.0, 0.5, 2.04, 0.045, 0.004), W)
    m.add(boxr(-0.5, 1.39, 0.0, 0.5, 1.44, 0.05, 0.004), W)          # meeting rail
    m.add(boxr(-0.0125, 0.85, 0.0, 0.0125, 1.98, 0.04, 0.003), W)
    for y in (1.12, 1.71):
        m.add(boxr(-0.45, y - 0.0125, 0.0, 0.45, y + 0.0125, 0.04, 0.003), W)
    sky = Part(uv01=True)
    add_poly(sky, [(-0.45, 0.85, 0.015), (0.45, 0.85, 0.015), (0.45, 1.98, 0.015), (-0.45, 1.98, 0.015)], (0, 0, 1),
             uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])
    m.add(sky, "night_sky", (0.08, 0.1, 0.18))
    # curtain rod with finials, brackets and rings
    m.add(cyl(0.016, 1.5, 14, -0.75).rz(-90).t_(0, 2.16, 0.165), W)
    for sx in (-1, 1):
        fin = lathe([(0, 0), (0.02, 0), (0.024, 0.01), (0.012, 0.02), (0.035, 0.045), (0.03, 0.065), (0.012, 0.08), (0.0, 0.085)], 14, smooth=40)
        m.add(fin.rz(-90 * sx).t_(sx * 0.715, 2.16, 0.165), W)
        m.add(boxr(sx * 0.64, 2.12, 0.0, sx * 0.67, 2.16, 0.165, 0.006), W)
    # curtains: pleated sheets gathered by tie-backs
    for sx in (-1, 1):
        def width(y):
            t_tie = 1.05
            if y >= t_tie:
                k = (y - t_tie) / (2.14 - t_tie)
                return 0.17 + (0.42 - 0.17) * (0.5 - 0.5 * math.cos(math.pi * min(1, k) ** 0.8))
            k = (t_tie - y) / t_tie
            return 0.17 + 0.15 * math.sin(math.pi * 0.5 * k)

        def curtain(u, v, sx=sx):
            y = 0.0 + 2.14 * v
            w = width(y)
            xo = sx * 0.78
            xi = xo - sx * w
            x = xo + (xi - xo) * u if sx < 0 else xi + (xo - xi) * u
            pleats = 6
            amp = 0.012 + 0.018 * (0.42 - w) / 0.25
            z = 0.168 + amp * math.sin(u * pleats * 2 * math.pi)
            return (x, y, z)

        cur = sheet(curtain, 60, 40, 0.008, uv=lambda u, v, P: (u * 0.6, P[1]))
        m.add(orient_to_normals(cur), CU)
        tie = torus(0.1, 0.012, 20, 6).s(1, 0.6, 0.3).t_(sx * (0.78 - 0.085), 1.05, 0.168)
        m.add(tie, CU)
        m.add(lathe([(0, 0), (0.015, 0.0), (0.022, 0.05), (0.01, 0.08), (0.006, 0.1), (0.0, 0.105)], 10, smooth=50).t_(sx * 0.7, 0.88, 0.19), CU)
        for k in range(6):  # rings
            x = sx * (0.78 - 0.07 * k - 0.02)
            m.add(torus(0.024, 0.004, 12, 6).rz(90).t_(x, 2.16, 0.165), W)


# ===================================== quest extension (v2) =====================================
def compact(part):
    """Drops vertices no triangle references."""
    used = sorted({i for f in part.f for i in f})
    remap = {o: k for k, o in enumerate(used)}
    q = Part(part.uv01)
    q.p = [part.p[i] for i in used]; q.n = [part.n[i] for i in used]; q.t = [part.t[i] for i in used]
    q.f = [tuple(remap[i] for i in f) for f in part.f]
    return q


def split_tris(part, pred):
    """Splits a part into (selected, rest) by a predicate on the triangle's index triple."""
    a, b = part.copy(), part.copy()
    a.f = [f for f in part.f if pred(f)]
    b.f = [f for f in part.f if not pred(f)]
    return compact(a), compact(b)


def smooth_normals(part, nd=6):
    """Recomputes smooth vertex normals (area weighted, shared across coincident vertices)."""
    keys = [tuple(round(c, nd) for c in q) for q in part.p]
    acc = {}
    for a, b, c in part.f:
        g = v_cross(v_sub(part.p[b], part.p[a]), v_sub(part.p[c], part.p[a]))
        for i in (a, b, c):
            acc[keys[i]] = v_add(acc.get(keys[i], (0.0, 0.0, 0.0)), g)
    part.n = [v_norm(acc[k]) if k in acc else n for k, n in zip(keys, part.n)]
    return part


def _in_poly(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x0, y0), (x1, y1) = poly[i - 1], poly[i]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def frame_matrix(f, up=(0, 1, 0)):
    """Rotation whose local +Z is `f` and local +Y is as close to `up` as possible (columns s, u, f)."""
    f = v_norm(f)
    s_ = v_norm(v_cross(up, f))
    u = v_cross(f, s_)
    return ((s_[0], u[0], f[0]), (s_[1], u[1], f[1]), (s_[2], u[2], f[2]))


@model("lib_map", (0.45, 0.06, 0.32))
def m_map(m):
    R0, gap, turns, band = 0.022, 0.0035, 1.6, 0.0036
    a = 0.225 - R0 - band
    D = 0.32
    nflat, nroll = 30, 60
    right = [(a * i / nflat, 0.0) for i in range(nflat + 1)]
    for k in range(1, nroll + 1):  # the end curls up and back over itself in a tightening spiral
        phi = 2 * math.pi * turns * k / nroll
        r = R0 - gap * phi / (2 * math.pi)
        right.append((a + r * math.sin(phi), R0 - r * math.cos(phi)))
    roll_len = [0.0]
    for k in range(nflat + 1, len(right)):
        roll_len.append(roll_len[-1] + math.hypot(right[k][0] - right[k - 1][0], right[k][1] - right[k - 1][1]))
    # the flat part shows u 0.08..0.92 of the image; the rolled-up margins take the rest
    side_u = [0.5 + 0.42 * k / nflat for k in range(nflat + 1)] + [0.92 + 0.08 * e / roll_len[-1] for e in roll_len[1:]]
    prof = [(-x, y) for x, y in reversed(right[1:])] + right
    U = [1 - u for u in reversed(side_u[1:])] + side_u
    nu = len(prof) - 1

    def surf(u, v):
        j = int(round(u * nu))
        x, y = prof[j]
        z = D / 2 - D * v
        if abs(x) < a - 0.03 and y == 0.0:  # gentle cockling of old parchment, fading towards the rolls
            fade = min(1.0, (a - 0.03 - abs(x)) / 0.05)
            y = 0.0012 * fade * (1 + math.sin(x * 23 + 1.3) * math.sin(z * 17 + 0.4))
        return (x, y, z)

    paper = sheet(surf, nu, 10, 0.0012, uv=lambda u, v, P: (U[int(round(u * nu))], v))
    paper.uv01 = True
    m.add(paper, "map_parchment")
    # velvet ribbon tied round the right-hand roll: band, bow loops, knot, two tails resting on the map
    zr = 0.06
    ring = lathe([(R0 + 0.0006, -0.009), (R0 + band, -0.009), (R0 + band, 0.009), (R0 + 0.0006, 0.009), (R0 + 0.0006, -0.009)], 36)
    m.add(ring.rx(90).t_(a, R0, zr), "velvet_red")
    top = 2 * R0 + band
    m.add(sphere(0.0048, 12, 8).s(1.2, 0.8, 1).t_(a, top + 0.0022, zr), "velvet_red")
    for sz in (-1, 1):
        loop = torus(0.0105, 0.0022, 18, 6).s(0.55, 1, 1).rx(sz * 22).t_(a, top + 0.0035, zr + sz * 0.0115)
        m.add(loop, "velvet_red")
        tail = bezier((a - 0.004, top + 0.001, zr + sz * 0.003), (a - 0.03, top + 0.004, zr + sz * 0.01),
                      (a - 0.048, 0.01, zr + sz * 0.02), (a - 0.085 - (0.012 if sz > 0 else 0.0), 0.0028, zr + sz * 0.032), 14)
        m.add(sweep(tail, rect2d(0.011, 0.0012), smooth=False, hint=(0, 0, 1)), "velvet_red")


@model("bed_key", (0.16, 0.02, 0.05))
def m_key(m):
    BR, KE = "brass", (0.25, 0.18, 0.05)
    yc = 0.01
    bx = -0.0555
    parts = [torus(0.0185, 0.0052, 36, 8).t_(bx, yc, 0)]                       # looped bow
    for k in range(4):                                                          # quatrefoil inside the bow
        ang = math.radians(45 + 90 * k)
        parts.append(torus(0.0062, 0.0016, 18, 6).t_(bx + 0.0074 * math.cos(ang), yc, 0.0074 * math.sin(ang)))
    parts.append(sphere(0.0042, 12, 8).s(1, 0.85, 1).t_(bx, yc, 0))
    for k in range(1, 8):                                                       # beads round the bow
        ang = math.radians(k * 45)
        parts.append(sphere(0.0027, 10, 6).t_(bx - 0.0228 * math.cos(ang), yc, 0.0228 * math.sin(ang)))
    parts.append(cyl(0.0045, 0.116, 14).rz(-90).t_(-0.038, yc, 0))             # shaft
    collar = [(0, 0), (0.006, 0), (0.0075, 0.003), (0.01, 0.0065), (0.0075, 0.01), (0.0058, 0.0125), (0.0068, 0.016),
              (0.0048, 0.019), (0, 0.019)]
    parts.append(lathe(collar, 16, smooth=50).rz(-90).t_(-0.04, yc, 0))
    parts.append(torus(0.0047, 0.0014, 14, 6).rz(90).t_(0.012, yc, 0))         # rings on the shaft
    parts.append(torus(0.0047, 0.0014, 14, 6).rz(90).t_(0.044, yc, 0))
    parts.append(sphere(0.0048, 12, 8).t_(0.078, yc, 0))                       # rounded tip
    bit = [(0.05, 0.0), (0.0775, 0.0), (0.0775, 0.021), (0.0718, 0.021), (0.0718, 0.0145), (0.0672, 0.0145),
           (0.0672, 0.021), (0.0615, 0.021), (0.0615, 0.011), (0.0568, 0.011), (0.0568, 0.018), (0.05, 0.018)]
    parts.append(extrude(bit, 0.0055, smooth=20).rx(90).t_(0, yc, 0))          # toothed bit, lying flat
    for q in parts:
        m.add(q, BR, KE)


# Eye / hole layout on the porcelain_cracked face image (u, v in 0..1, v = 0 at the chin).
SCARE_EYES = ((0.35, 0.535), (0.65, 0.535))
SCARE_EYE_R = (0.085, 0.072)
SCARE_HOLE = (0.2, 0.4)


@model("scare_doll", (0.55, 0.9, 0.45))
def m_scare_doll(m):
    SK, DR, HA, FACE = "skin", "doll_dress", "hair", "porcelain_cracked"
    SOCK = ("hair", (0.6, 0.0, 0.0), "eye_socket")
    GLINT = ("skin", (1.0, 0.25, 0.15), "eye_glint")
    rng = random.Random(13)
    HR = (0.1, 0.12, 0.1)
    HEAD_T = (0.012, 0.745, 0.05)

    def hp(q):  # head-local point -> world (chin dipped, head cocked hard to one side)
        return v_add(m_mul_v(rot_z(-24), m_mul_v(rot_x(10), q)), HEAD_T)

    def head(part):
        return part.rx(10).rz(-24).t_(*HEAD_T)

    def socket_depth(u, v):
        d = 0.0
        for eu, ev in SCARE_EYES:
            k = ((u - eu) / SCARE_EYE_R[0]) ** 2 + ((v - ev) / SCARE_EYE_R[1]) ** 2
            if k < 1:
                d = max(d, 0.03 * (1 - k) ** 1.2)
        return d

    # ---- porcelain face: front half of the head, face image mapped once, real hollow eye sockets
    prof = [(math.sin(math.pi * i / 40), -math.cos(math.pi * i / 40)) for i in range(41)]
    face = lathe(prof, 40, smooth=89, a0=-90, a1=90).s(*HR)
    back = lathe(prof, 24, smooth=89, a0=90, a1=270).s(*HR)

    def uv_of(q):
        return (0.5 + q[0] / (2 * HR[0]), 0.5 + q[1] / (2 * HR[1]))

    def ell_n(q):
        return v_norm((q[0] / HR[0] ** 2, q[1] / HR[1] ** 2, q[2] / HR[2] ** 2))

    def surf_pt(u, v, lift=0.0):
        x, y = (u - 0.5) * 2 * HR[0], (v - 0.5) * 2 * HR[1]
        z = HR[2] * math.sqrt(max(0.0, 1 - (x / HR[0]) ** 2 - (y / HR[1]) ** 2))
        q = (x, y, z)
        return v_add(q, v_mul(ell_n(q), lift))

    uvs, newp = [], []
    for q in face.p:
        u, v = uv_of(q)
        k = ((u - 0.5) / 0.06) ** 2 + ((v - 0.44) / 0.07) ** 2
        bump = 0.008 * (1 - k) ** 2 if k < 1 else 0.0                   # small porcelain nose
        newp.append(v_add(q, v_mul(ell_n(q), bump - socket_depth(u, v))))
        uvs.append((u, v))
    face.p, face.t, face.uv01 = newp, uvs, True                         # UVs from the undisplaced shape
    smooth_normals(face)
    hole = []
    for k in range(11):  # jagged broken-off shard on the left cheek
        ang = 2 * math.pi * k / 11
        r = 0.045 + rng.uniform(0.0, 0.028)
        hole.append((SCARE_HOLE[0] + r * math.cos(ang), SCARE_HOLE[1] + 0.8 * r * math.sin(ang)))

    def centroid_uv(part, f):
        return (sum(part.t[i][0] for i in f) / 3, sum(part.t[i][1] for i in f) / 3)

    _, face = split_tris(face, lambda f: _in_poly(centroid_uv(face, f), hole))
    sockets, face = split_tris(face, lambda f: min(socket_depth(*face.t[i]) for i in f) > 0.005)
    m.add(head(face), FACE)
    m.add(head(sockets), *SOCK)
    m.add(head(back), SK)
    m.add(head(sphere(1.0, 20, 12).s(HR[0] * 0.68, HR[1] * 0.68, HR[2] * 0.68)), *SOCK)  # glowing hollow inside the head
    for eu, ev in SCARE_EYES:  # pin-point lights deep in the sockets
        base = surf_pt(eu, ev)
        m.add(head(sphere(0.0065, 10, 6).t_(*v_add(base, v_mul(ell_n(base), -0.026)))), *GLINT)
    cracks = [[(0.2, 0.455), (0.17, 0.55), (0.2, 0.63), (0.17, 0.72)],      # radiating from the broken cheek
              [(0.255, 0.42), (0.3, 0.44), (0.33, 0.41), (0.39, 0.43)],
              [(0.23, 0.345), (0.27, 0.26), (0.25, 0.18), (0.3, 0.1)],
              [(0.42, 0.8), (0.47, 0.74), (0.52, 0.7), (0.55, 0.64)],      # continues the painted crack between the eyes
              [(0.74, 0.47), (0.77, 0.4), (0.74, 0.33), (0.79, 0.26)]]
    for c in cracks:
        pts = []
        for i in range(len(c) - 1):
            for t in (0.0, 0.33, 0.66):
                pts.append(surf_pt(c[i][0] + (c[i + 1][0] - c[i][0]) * t, c[i][1] + (c[i + 1][1] - c[i][1]) * t, 0.001))
        pts.append(surf_pt(c[-1][0], c[-1][1], 0.001))
        m.add(head(tube(pts, 0.0013, 5, True, [1.0 - 0.6 * i / (len(pts) - 1) for i in range(len(pts))])), HA)
    # ---- hair: tilted cap over crown and back, matted wild strands, a torn ribbon bow
    cap = [(math.cos(math.radians(t)), math.sin(math.radians(t))) for t in range(-35, 91, 7)] + [(0.0, 1.0)]
    m.add(head(lathe(cap, 26, smooth=60).rx(-65).s(HR[0] * 1.08, HR[1] * 1.07, HR[2] * 1.08)), HA)  # tilt first so it hugs the head
    for k in range(46):
        th = rng.uniform(0, 2 * math.pi)
        lat = rng.uniform(-0.15, 1.2)
        local = (1.06 * HR[0] * math.cos(lat) * math.sin(th), 1.05 * HR[1] * math.sin(lat), 1.06 * HR[2] * math.cos(lat) * math.cos(th))
        if local[2] > 0.02 and (local[1] < 0.06 or abs(local[0]) < 0.065):  # keep the face clear except for the strands below
            continue
        p0 = hp(local)
        dirv = v_mul(v_norm(v_sub(p0, HEAD_T)), 0.8 if rng.random() < 0.25 else 0.35)   # a few wild fly-aways
        L = rng.uniform(0.16, 0.4)
        pts = [p0]
        cur = p0
        for i in range(7):
            dirv = v_norm(v_add(v_add(dirv, (0, -0.75, -0.18)), (rng.uniform(-0.4, 0.4), rng.uniform(-0.15, 0.2), rng.uniform(-0.25, 0.25))))
            cur = v_add(cur, v_mul(dirv, L / 7))
            cur = (max(-0.255, min(0.255, cur[0])), max(0.3, min(0.895, cur[1])), max(-0.14, cur[2]))
            pts.append(cur)
        m.add(tube(pts, 0.0055, 5, True, [1.0 - 0.65 * i / 7 for i in range(8)]), HA)
    # matted mass of long hair hanging down the back, ragged at the ends
    ends = [rng.uniform(0.0, 0.35) for _ in range(41)]

    def mass(u, v):
        j = int(round(u * 40))
        th = math.radians(92 + 176 * u)
        yb = -0.27 + 0.1 * ends[j]
        y = yb + (0.03 - yb) * v
        r = 0.112 + 0.045 * (1 - v) + 0.008 * math.sin(th * 13 + y * 40)
        return (r * math.sin(th) * HR[0] / 0.1, y, r * math.cos(th) - 0.01)

    m.add(head(sheet(mass, 40, 8, 0.004)), HA)
    for x0, sway in ((-0.07, -0.02), (0.014, 0.025)):  # strands falling across the face
        loc = [(x0, 0.075, 0.098), (x0 + sway * 0.4, 0.02, 0.112), (x0 + sway, -0.05, 0.112), (x0 + sway * 1.4, -0.12, 0.1),
               (x0 + sway * 1.6, -0.19, 0.075), (x0 + sway * 2.0, -0.25, 0.05)]
        m.add(tube([hp(q) for q in loc], 0.0042, 5, True, [1.0, 0.95, 0.85, 0.7, 0.55, 0.4]), HA)
    for sz in (-1, 1):
        m.add(head(superellipsoid(0.028, 0.016, 0.009, 0.7, 0.7, 10, 6).rz(sz * 25).ry(-50).t_(0.08 + sz * 0.012, 0.085, -0.03)), DR)
    m.add(head(sphere(0.009, 8, 6).t_(0.082, 0.085, -0.03)), DR)

    # ---- body: leaning bodice, torn lace collar, neck
    def lean(part):
        return part.t_(0, -0.42, 0).rx(12).t_(0, 0.42, 0)

    m.add(lean(superellipsoid(0.085, 0.105, 0.065, 0.6, 0.7, 20, 12).t_(0, 0.53, 0)), DR)
    m.add(lean(cyl(0.02, 0.075, 12, 0.6)), SK)
    m.add(lean(torus(0.04, 0.008, 20, 6).s(1, 0.5, 0.85).t_(0, 0.627, 0)), DR)
    for k in range(14):
        ang = 360.0 * k / 14
        spike = lathe([(0, 0), (0.008, 0), (0, rng.uniform(0.018, 0.032))], 6).rx(115).t_(0, 0, 0.04).ry(ang).s(1, 1, 0.85)
        m.add(lean(spike.t_(0, 0.625, 0)), DR)
    # ---- torn skirt: jagged hem, rag strips, a few slits, swept back by the lunge
    nu = 72
    hem = []
    for j in range(nu):
        h = 0.15 + rng.uniform(-0.02, 0.03) + (0.02 if j % 2 else -0.015)
        if rng.random() < 0.12:
            h = rng.uniform(0.06, 0.09)          # long hanging rag
        hem.append(h)
    hem.append(hem[0])
    waist = 0.45

    def skirt(u, v):
        j = int(round(u * nu))
        h = hem[j]
        y = h + (waist - h) * v
        t = max(0.0, (waist - y) / (waist - 0.12))
        r = 0.078 + 0.055 * t ** 0.8 + 0.007 * math.sin(u * 2 * math.pi * 9) * t
        th = 2 * math.pi * u
        return (r * math.sin(th), y, r * math.cos(th) - 0.012 * t ** 1.5)

    sk = sheet(skirt, nu, 10, 0.004)
    slits = [rng.uniform(0, 2 * math.pi) for _ in range(4)]

    def in_slit(f):
        cx = sum(sk.p[i][0] for i in f) / 3; cy = sum(sk.p[i][1] for i in f) / 3; cz = sum(sk.p[i][2] for i in f) / 3
        ang = math.atan2(cx, cz) % (2 * math.pi)
        return cy < 0.3 and any(abs((ang - s0 + math.pi) % (2 * math.pi) - math.pi) < 0.06 for s0 in slits)

    _, sk = split_tris(sk, in_slit)
    m.add(sk, DR)
    # ---- legs: one braced forward, one pushing off on its toes
    for hip, knee, ankle, shoe_c, shoe_rx in (((-0.045, 0.42, 0.0), (-0.06, 0.22, 0.07), (-0.06, 0.06, 0.04), (-0.06, 0.024, 0.065), 0),
                                              ((0.045, 0.42, 0.0), (0.066, 0.22, -0.035), (0.075, 0.075, -0.1), (0.076, 0.035, -0.112), -55)):
        m.add(tube([hip, knee], 0.022, 10, True, [1.0, 0.85]), SK)
        m.add(sphere(0.019, 10, 6).t_(*knee), SK)
        m.add(tube([knee, ankle], 0.018, 10, True, [1.0, 0.8]), SK)
        m.add(superellipsoid(0.024, 0.021, 0.042, 0.6, 0.6, 12, 8).rx(shoe_rx).t_(*shoe_c), HA)
    # ---- arms reaching for the viewer, with long clawed fingers
    for sx, sh, el, wr in ((1, (0.1, 0.6, 0.035), (0.18, 0.575, 0.11), (0.212, 0.6, 0.18)),
                           (-1, (-0.1, 0.6, 0.035), (-0.17, 0.655, 0.1), (-0.198, 0.725, 0.168))):
        m.add(superellipsoid(0.043, 0.04, 0.043, 0.7, 0.7, 14, 8).t_(*sh), DR)              # puffed sleeve
        for k in range(3):                                                                  # torn sleeve rags
            ang = math.radians(rng.uniform(-70, 70))
            ox, oz = 0.03 * math.sin(ang) * sx, 0.03 * math.cos(ang)
            L = rng.uniform(0.04, 0.07)

            def rag(u, v, ox=ox, oz=oz, L=L, ang=ang, sh=sh, sx=sx):
                w = 0.016 * (1 - v) + 0.002
                return (sh[0] + ox + (u - 0.5) * w * math.cos(ang), sh[1] - 0.025 - L * v,
                        sh[2] + oz - (u - 0.5) * w * math.sin(ang) * sx)
            m.add(sheet(rag, 2, 4, 0.002), DR)
        m.add(tube([sh, el], 0.021, 10, True, [1.0, 0.9]), SK)
        m.add(sphere(0.019, 10, 6).t_(*el), SK)
        m.add(tube([el, wr], 0.018, 10, True, [1.0, 0.8]), SK)
        f = v_sub(wr, el)
        M = frame_matrix(f, (0, 1, 0))
        palm_c = v_add(wr, v_mul(v_norm(f), 0.02))
        hand = [superellipsoid(0.025, 0.009, 0.026, 0.7, 0.7, 12, 8)]
        claws = []
        for xb, Ls, yaw in ((-0.018, 0.075, -16), (-0.006, 0.088, -5), (0.006, 0.09, 5), (0.018, 0.078, 14)):
            pts = [(xb, 0.0, 0.02)]
            d = (math.sin(math.radians(yaw)), 0.0, math.cos(math.radians(yaw)))
            for pitch in (8 + rng.uniform(-4, 4), 30 + rng.uniform(-8, 8), 55 + rng.uniform(-10, 10)):
                pr = math.radians(pitch)
                dd = v_norm((d[0] * math.cos(pr), -math.sin(pr), d[2] * math.cos(pr)))
                pts.append(v_add(pts[-1], v_mul(dd, Ls / 3)))
            hand.append(tube(pts, 0.0058, 6, True, [1.0, 0.9, 0.8, 0.7]))
            hand.append(sphere(0.0052, 8, 5).t_(*pts[1]))                                   # knuckles
            hand.append(sphere(0.0047, 8, 5).t_(*pts[2]))
            tipdir = v_norm(v_sub(pts[3], pts[2]))
            claw = lathe([(0, 0), (0.0042, 0), (0, 0.014)], 6)
            axis = v_cross((0, 1, 0), tipdir)
            if v_len(axis) > 1e-6:
                claw.rot(axis, math.degrees(math.acos(max(-1.0, min(1.0, tipdir[1])))))
            claws.append(claw.t_(*pts[3]))
        tx = -sx * 0.026
        thumb = [(tx, 0.0, 0.0), (tx - sx * 0.012, -0.006, 0.02), (tx - sx * 0.014, -0.016, 0.04), (tx - sx * 0.01, -0.028, 0.052)]
        hand.append(tube(thumb, 0.0062, 6, True, [1.0, 0.9, 0.8, 0.7]))
        for q in hand:
            m.add(q.xform(M, palm_c), SK)
        for q in claws:
            m.add(q.xform(M, palm_c), HA)

# ===================================== build + verify =====================================
CONTRACT_TEXTURES = set("""wood_floor wallpaper_damask wallpaper_faded plaster wood_dark wood_light wood_painted_white stone
brick_soot velvet_red linen quilt carpet_persian carpet_round brass iron book_spines world_map portrait clock_face mirror
glass wax flame embers skin coat trousers leather hair doll_dress curtain marble night_sky paper
map_parchment porcelain_cracked""".split())


def build(mid):
    fn, size, mode = MODELS[mid]
    m = Model(mid, size, mode)
    fn(m)
    m.finalize()
    m.write()
    return m


def verify(mid, out_dir=OUT_DIR):
    """Re-reads the written OBJ/MTL and checks size, origin convention, indices and materials."""
    fn, size, mode = MODELS[mid]
    errors = []
    mats = {}
    cur = None
    for line in open(os.path.join(out_dir, mid + ".mtl"), encoding="utf-8"):
        t = line.split()
        if not t:
            continue
        if t[0] == "newmtl":
            cur = t[1]; mats[cur] = {}
        elif cur and t[0] in ("map_Kd", "Ke", "Kd"):
            mats[cur][t[0]] = t[1:]
    for name, d in mats.items():
        if "map_Kd" not in d:
            errors.append(f"material {name} has no map_Kd")
            continue
        tex = os.path.basename(d["map_Kd"][0])[:-4]
        if tex not in CONTRACT_TEXTURES:
            errors.append(f"material {name} uses unknown texture {tex}")
    nv = nt = nn = ntri = 0
    mn = [1e9] * 3; mx = [-1e9] * 3
    used = set()
    faces = []
    for line in open(os.path.join(out_dir, mid + ".obj"), encoding="utf-8"):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            nv += 1
            for k in range(3):
                x = float(t[k + 1]); mn[k] = min(mn[k], x); mx[k] = max(mx[k], x)
        elif t[0] == "vt":
            nt += 1
        elif t[0] == "vn":
            nn += 1
        elif t[0] == "usemtl":
            used.add(t[1])
        elif t[0] == "f":
            ntri += len(t) - 3
            faces.append(t[1:])
    for f in faces:
        for c in f:
            vi, ti, ni = (int(x) for x in c.split("/"))
            if not (1 <= vi <= nv and 1 <= ti <= nt and 1 <= ni <= nn):
                errors.append(f"bad index {c}")
                break
    for u in used - set(mats):
        errors.append(f"usemtl {u} not defined in mtl")
    dims = [mx[k] - mn[k] for k in range(3)]
    for k, ax in enumerate("WHD"):
        if abs(dims[k] - size[k]) > 0.05 * size[k]:
            errors.append(f"{ax} {dims[k]:.3f} vs contract {size[k]}")
    tol = 0.002
    if abs((mn[0] + mx[0]) / 2) > tol:
        errors.append("not centred on X")
    if mode == "wall":
        if abs(mn[2]) > tol:
            errors.append(f"[wall] min z {mn[2]:.4f} != 0")
    elif abs((mn[2] + mx[2]) / 2) > tol:
        errors.append("not centred on Z")
    if mode == "hang":
        if abs(mx[1]) > tol:
            errors.append(f"[hang] max y {mx[1]:.4f} != 0")
    elif abs(mn[1]) > tol:
        errors.append(f"min y {mn[1]:.4f} != 0")
    if ntri > 30000:
        errors.append(f"{ntri} triangles > 30000")
    return dims, ntri, sorted(mats), errors, (mn, mx)


def main(argv):
    ids = argv or list(MODELS)
    unknown = [i for i in ids if i not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model ids: {unknown}")
    failed = False
    print(f"{'id':20s} {'W x H x D (m)':>22s} {'tris':>6s} {'raw scale':>18s} flips  materials")
    for mid in ids:
        m = build(mid)
        dims, ntri, mats, errors, _ = verify(mid)
        sc = " ".join(f"{s:.3f}" for s in m.scale)
        print(f"{mid:20s} {dims[0]:6.3f} x {dims[1]:6.3f} x {dims[2]:6.3f} {ntri:6d} {sc:>18s} {m.flips:5d}  {' '.join(mats)}")
        for e in errors:
            print("   ERROR:", e)
        failed |= bool(errors)
    if len(ids) == len(MODELS):
        missing = [i for i in MODELS if not os.path.exists(os.path.join(OUT_DIR, i + ".obj"))]
        assert not missing, missing
    if failed:
        raise SystemExit("verification failed")
    print(f"OK: {len(ids)} model(s) written to {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
