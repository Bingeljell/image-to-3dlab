# build_bus_v2.py - the route (b) city bus (docs/route-b-models.md, Entwuerfe/25_fable_bus-blockout_brief.md).
#
# A generic modern 12 m European low-floor city bus, built by code: the final game mesh, kept as built.
#   blender -b -P build_bus_v2.py -- [--out <dir>] [--tex 4096] [--no-ao]
#
# Writes into --out (default: this script's folder):
#   bus_v2_parts.glb / .blend   body, wheel_FL/FR/RL/RR, door_1a..3b, interior: separate objects, origin at the pivot
#   bus_v2_joined.glb           the same faces as one mesh, transforms applied, same UVs and material
#   bus_v2_basecolor.png        the 4096 px atlas with the flat part colours (the livery guide)
#
# Frame (Blender): +Y is the front, +X the bus's right side (the kerb side with the doors), Z up, ground at z = 0,
# origin at the ground centre. That is the facing of Fable's ambulance.glb (front = -Z in glTF).
#
# How the body is made: a box whose edges are rounded by a mapping (deform), with grid lines where the features sit.
# The front and the rear are grids wrapped around the corners (wrap_grid), with tapered recesses and steps for the
# windscreen, the display, the lights, the grilles and the bumpers. The flat sides are triangulated around their
# holes (tri_fill): recessed windows, door openings, wheel arches. Every face carries tags: the object it belongs to,
# its colour, and the UV chart it is projected into; the charts are packed into one atlas and painted here.
import bpy, bmesh, math, sys, os, time
import numpy as np
from mathutils import Vector
from mathutils.geometry import tessellate_polygon, delaunay_2d_cdt

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


OUT = os.path.abspath(arg("--out", HERE))
NAME = arg("--name", "bus_v2")
DO_AO = "--no-ao" not in ARGS
TEX = int(arg("--tex", "4096"))     # atlas size
SS = 2                              # supersampling while painting
PAD = 4                             # gutter px on each side of every chart (8 px between two charts)
BASE_PPM = 300.0                    # px per metre before the auto-fit scale
WHEEL_SEGS = 32

# ---------------------------------------------------------------------------------------------------------------------
# colours (sRGB display values): the livery guide
# ---------------------------------------------------------------------------------------------------------------------
C = dict(
    skin=(0.88, 0.885, 0.87),        # white body
    yellow=(0.94, 0.75, 0.10),       # the band at the window sill
    black=(0.035, 0.035, 0.04),      # window frames, pillars, door frames
    bumper=(0.06, 0.06, 0.065),
    glass=(0.13, 0.18, 0.24),        # dark glass
    display=(0.018, 0.018, 0.02),    # destination display (Flow paints the route number)
    tyre=(0.07, 0.07, 0.075), rim=(0.68, 0.69, 0.70), hub=(0.45, 0.46, 0.47),
    under=(0.045, 0.045, 0.047),     # underside, wheel wells
    lamp=(0.97, 0.93, 0.80),         # warm white headlights
    amber=(0.95, 0.55, 0.08), tail=(0.72, 0.05, 0.05), revers=(0.90, 0.90, 0.88),
    grille=(0.075, 0.075, 0.08), ac=(0.74, 0.75, 0.75), hatch=(0.66, 0.67, 0.68),
    mirror=(0.45, 0.53, 0.60), trim=(0.09, 0.09, 0.095), plate=(0.90, 0.90, 0.88), jamb=(0.05, 0.05, 0.055),
    floor=(0.30, 0.31, 0.32), wall=(0.24, 0.245, 0.25), ceil=(0.29, 0.295, 0.30), housing=(0.22, 0.225, 0.23),
    seat=(0.12, 0.25, 0.52), pole=(0.92, 0.74, 0.12), dash=(0.11, 0.11, 0.115),
)

# ---------------------------------------------------------------------------------------------------------------------
# the vehicle's numbers (metres): a preset of the kit; the next steps derive more of the layout from them
# ---------------------------------------------------------------------------------------------------------------------
BUS_12M = dict(
    half_width=1.275,            # half width of the body (2.55 m)
    front=5.95, rear=-5.96,      # front / rear plane of the box; the bumpers add 0.05 / 0.04 -> 12.0 m
    skirt=0.27, roof=2.98,       # skirt bottom, roof
    radii=(0.30, 0.24, 0.36, 0.24),          # edge radii: across, up, along (front), along (rear)
    screen=(1.18, 8.5, 0.07),    # windscreen: rake starts at this height, rake in degrees, bow of the front
    rear_screen=(1.50, 3.0, 0.04),
    wheel_radius=0.50,           # tyres 1.0 m across
    axles=(3.25, -2.65),         # front, rear: front overhang 2.75 m, wheelbase 5.9 m
    tyre_width=dict(F=0.30, R=0.60),         # single front tyres, twin rear tyres
    door_opening=(0.34, 2.40),   # floor height at the doors .. top of the opening
    doors=[(3.92, 5.22), (-0.10, 1.25), (-4.62, -3.32)],      # y0, y1 of doors 1, 2, 3 on the right side
    glass=(1.30, 2.44),          # side glass, bottom .. top
    band_rear={-1: -4.62, 1: -5.52},         # rear end of the black window band, left / right side
    windows_left=[(3.92, 5.12), (2.52, 3.82), (1.12, 2.42), (-0.28, 1.02), (-1.68, -0.38), (-3.08, -1.78), (-4.48, -3.18)],
    windows_right=[(2.52, 3.82), (1.37, 2.42), (-1.68, -0.22), (-3.08, -1.78), (-5.40, -4.74)],
)
PRESETS = {
    "bus_12m": BUS_12M,
    # the same bus with its side windows spread by the kit's rule instead of placed by hand
    "bus_12m_rule": dict(BUS_12M, window_pitch=(1.40, 0.10), window_span={-1: (-4.48, 5.12), 1: (-5.40, 5.12)}),
    # the first bus from numbers alone: 9 m, two doors, windows and interior by rule
    "midibus_9m": dict(BUS_12M, front=4.45, rear=-4.46, axles=(2.15, -2.15), doors=[(2.82, 3.72), (-0.55, 0.65)],
                       band_rear={-1: -3.12, 1: -4.02}, window_pitch=(1.40, 0.10),
                       window_span={-1: (-2.98, 3.62), 1: (-3.90, 3.62)}, hatches=(-1.40,), interior_rule=True),
}
SPEC = PRESETS[arg("--preset", "bus_12m")]
W = SPEC["half_width"]
YF, YR = SPEC["front"], SPEC["rear"]
ZB, ZT = SPEC["skirt"], SPEC["roof"]
RX, RZ, RYF, RYR = SPEC["radii"]
Z_WS0, RAKE_F, BOW_F = SPEC["screen"][0], math.tan(math.radians(SPEC["screen"][1])), SPEC["screen"][2]
Z_RK0, RAKE_R, BOW_R = SPEC["rear_screen"][0], math.tan(math.radians(SPEC["rear_screen"][1])), SPEC["rear_screen"][2]
F6 = (1.0, 0.7321, 0.4226)      # fillet lines: distance from the box edge in radii (0, 15, 30 degrees; the edge is 45)
XIN = (-0.60, 0.0, 0.60)        # grid lines across the front, the rear and the roof
YMID = (-2.0, 2.0)              # grid lines along the roof and the cantrail
ZLF = [ZB, 0.60, 0.66, 0.90, Z_WS0, 2.44, 2.50, ZT - RZ, ZT - RZ * F6[1], ZT - RZ * F6[2], ZT]            # front rows
ZLR = [ZB, 0.58, 0.66, 0.92, 1.10, 1.42, 1.62, 2.46, ZT - RZ, ZT - RZ * F6[1], ZT - RZ * F6[2], ZT]      # rear rows

WHEEL_R = SPEC["wheel_radius"]
YA_F, YA_R = SPEC["axles"]
XO = W - 0.035                  # outer face of the tyres
RA, NA = 0.57, 18               # wheel arch radius, segments
XWI = 0.60                      # inner end of the wheel wells
WHEEL_W = SPEC["tyre_width"]

ZD0, ZD1 = SPEC["door_opening"]
DOORS = SPEC["doors"]
XDO, XDI = W - 0.05, W - 0.09   # door leaves: outer face (recessed 5 cm), inner face
XDC = (XDO + XDI) / 2
XI = W - 0.12                   # inner wall of the body = side walls of the interior
ZF, ZC = ZD0, 2.50              # interior floor, ceiling
YI0, YI1 = YR + 0.66, YF - 0.57  # interior rear / front wall
HATCHES = SPEC.get("hatches", (0.20, -2.50))   # roof hatches between the air conditioner and the rear pod

WIN_Z0, WIN_Z1 = SPEC["glass"]
BAND_Y0 = SPEC["band_rear"]
WIN_L = SPEC["windows_left"]
WIN_R = SPEC["windows_right"]
if "window_pitch" in SPEC:      # the kit's rule (shape_guide.spread_windows) instead of hand-placed windows
    sys.path.insert(0, HERE)
    import shape_guide as sg
    _pitch, _pillar = SPEC["window_pitch"]
    WIN_L = sg.spread_windows(*SPEC["window_span"][-1], pitch=_pitch, pillar=_pillar)
    WIN_R = sg.spread_windows(*SPEC["window_span"][1], pitch=_pitch, pillar=_pillar, keep_out=DOORS)
Y_SIDE0, Y_SIDE1 = YR + RYR + BOW_R, YF - RYF - BOW_F      # where the flat sides end (the yellow band stops there)

PIVOTS = {"body": (0.0, 0.0, 0.0), "interior": (0.0, 0.0, 0.0)}
for _n, _ya, _sx, _k in (("wheel_FL", YA_F, -1, "F"), ("wheel_FR", YA_F, 1, "F"), ("wheel_RL", YA_R, -1, "R"), ("wheel_RR", YA_R, 1, "R")):
    PIVOTS[_n] = (_sx * (XO - WHEEL_W[_k] / 2), _ya, WHEEL_R)
for _k, (_y0, _y1) in enumerate(DOORS):
    PIVOTS["door_%da" % (_k + 1)] = (XDC, _y1 - 0.02, ZD0)      # front leaf, hinge at its front edge
    PIVOTS["door_%db" % (_k + 1)] = (XDC, _y0 + 0.02, ZD0)      # rear leaf, hinge at its rear edge
OBJECTS = (["body", "wheel_FL", "wheel_FR", "wheel_RL", "wheel_RR"]
           + ["door_%d%s" % (k + 1, leaf) for k in range(len(DOORS)) for leaf in "ab"]
           + (["interior"] if SPEC.get("interior", True) else []))


# ---------------------------------------------------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------------------------------------------------
class Builder:
    def __init__(self):
        self.verts = []
        self.faces = []

    def v(self, x, y, z):
        self.verts.append((x, y, z))
        return len(self.verts) - 1

    def f(self, idx, **tags):
        self.faces.append((list(idx), tags))

    def rings(self, rings, tagfn, cap0=None, cap1=None):
        """rings: closed loops of xyz. Quads between consecutive loops, n-gon caps."""
        ids = [[self.v(*p) for p in r] for r in rings]
        n = len(ids[0])
        for s in range(len(ids) - 1):
            for i in range(n):
                j = (i + 1) % n
                q = [ids[s][i], ids[s][j], ids[s + 1][j], ids[s + 1][i]]
                self.f(q, **tagfn(s, i, q))
        if cap0 is not None:
            self.f(list(reversed(ids[0])), **cap0)
        if cap1 is not None:
            self.f(list(ids[-1]), **cap1)
        return ids

    def box(self, x0, x1, y0, y1, z0, z1, obj, part, mat, dens=1.0, top_inset=0.0, zmid=None, bottom="under"):
        """axis box, optionally with a chamfered top section starting at zmid."""
        def rect(z, ins):
            return [(x0 + ins, y0 + ins, z), (x1 - ins, y0 + ins, z), (x1 - ins, y1 - ins, z), (x0 + ins, y1 - ins, z)]
        rr = [rect(z0, 0.0)]
        rr += [rect(zmid, 0.0), rect(z1, top_inset)] if top_inset > 0 else [rect(z1, 0.0)]
        t = dict(obj=obj, part=part, mat=mat, dens=dens)
        self.rings(rr, lambda s, i, q: dict(t), cap0=dict(obj=obj, part=part, mat=bottom, dens=dens * 0.3), cap1=dict(t))


def inside(poly, x, y):
    c = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            c = not c
    return c


def tri_fill(b, outer, holes, proj, **tags):
    """triangles (constrained Delaunay) for the flat region inside the loop `outer` and outside the loops `holes`;
    loops are vertex ids, proj maps xyz to the region's plane. Every loop vertex is used, none is added."""
    loops = [list(outer)] + [list(h) for h in holes]
    ids = [i for lp in loops for i in lp]
    co = [Vector(proj(b.verts[i])) for i in ids]
    edges, k = [], 0
    for lp in loops:
        n = len(lp)
        edges += [(k + a, k + (a + 1) % n) for a in range(n)]
        k += n
    vo, _eo, fo, ov = delaunay_2d_cdt(co, edges, [], 0, 1e-6)[:4]
    vmap = []
    for src in ov:
        if not src:
            raise RuntimeError("tri_fill: loops cross each other")
        vmap.append(ids[src[0]])
    if len(set(vmap)) != len(ids):
        raise RuntimeError("tri_fill: two loop vertices share a position")
    polys = [[proj(b.verts[i]) for i in lp] for lp in loops]
    n = 0
    for tri in fo:
        p = [vo[i] for i in tri]
        if abs((p[1][0] - p[0][0]) * (p[2][1] - p[0][1]) - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0])) < 1e-9:
            continue
        cx, cy = sum(q[0] for q in p) / 3.0, sum(q[1] for q in p) / 3.0
        if inside(polys[0], cx, cy) and not any(inside(h, cx, cy) for h in polys[1:]):
            b.f([vmap[i] for i in tri], **tags)
            n += 1
    return n


# ---------------------------------------------------------------------------------------------------------------------
# the body: a box with rounded edges
# ---------------------------------------------------------------------------------------------------------------------
def deform(x, y, z):
    """a point on the box surface -> its place on the rounded body, and the outward direction there."""
    cx = min(max(x, -(W - RX)), W - RX)
    cy = min(max(y, YR + RYR), YF - RYF)
    cz = min(z, ZT - RZ)                                  # the bottom edge stays sharp
    ry = RYF if y > cy else RYR
    dx, dy, dz = (x - cx) / RX, (y - cy) / ry, (z - cz) / RZ
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    if n < 1e-9:
        raise ValueError("deform: (%g, %g, %g) is not on the box surface" % (x, y, z))
    ux, uy, uz = dx / n, dy / n, dz / n
    px, py, pz = cx + RX * ux, cy + ry * uy, cz + RZ * uz
    if y >= YF - RYF - 1e-6:                              # the front end: raked windscreen, bowed in plan
        py -= max(0.0, pz - Z_WS0) * RAKE_F + BOW_F * (px / W) ** 2
    elif y <= YR + RYR + 1e-6:
        py += max(0.0, pz - Z_RK0) * RAKE_R + BOW_R * (px / W) ** 2
    return (px, py, pz), (ux, uy, uz)


class Body:
    """shared grid nodes on the box surface, so neighbouring patches are welded."""
    def __init__(self, b):
        self.b = b
        self.cache = {}

    def node(self, x, y, z):
        key = (round(x, 5), round(y, 5), round(z, 5))
        if key not in self.cache:
            self.cache[key] = self.b.v(*deform(x, y, z)[0])
        return self.cache[key]


T_BODY = dict(obj="body", part="body")
T_SIDE = {-1: dict(T_BODY, chart=("body", "-X"), view="-X"), 1: dict(T_BODY, chart=("body", "+X"), view="+X")}
T_FRONT = dict(T_BODY, chart=("body", "+Y"), view="+Y")
T_REAR = dict(T_BODY, chart=("body", "-Y"), view="-Y")
T_ROOF = dict(T_BODY, chart=("body", "+Z"), view="+Z")


def end_path(front):
    """box points around one end of the bus, from one flat side to the other (17 grid lines)."""
    if front:
        p = [(-W, YF - RYF * f) for f in F6] + [(-W, YF)] + [(-W + RX * f, YF) for f in reversed(F6)]
        p += [(x, YF) for x in XIN]
        p += [(W - RX * f, YF) for f in F6] + [(W, YF)] + [(W, YF - RYF * f) for f in reversed(F6)]
    else:
        p = [(W, YR + RYR * f) for f in F6] + [(W, YR)] + [(W - RX * f, YR) for f in reversed(F6)]
        p += [(x, YR) for x in reversed(XIN)]
        p += [(-W + RX * f, YR) for f in F6] + [(-W, YR)] + [(-W, YR + RYR * f) for f in reversed(F6)]
    return p


def wrap_grid(b, body, path, zl, col_tags, cell_mat, blocks):
    """the grid around one end. blocks: rectangles of cells set back (depth > 0) or standing proud (depth < 0),
    with tapered edges, so they stay inside their parent's UV chart."""
    nu, nv = len(path), len(zl)
    base = [[body.node(path[i][0], path[i][1], zl[j]) for j in range(nv)] for i in range(nu)]
    owner = set()
    for blk in blocks:
        for i in range(blk["i0"], blk["i1"]):
            for j in range(blk["j0"], blk["j1"]):
                assert (i, j) not in owner
                owner.add((i, j))
    for i in range(nu - 1):
        for j in range(nv - 1):
            if (i, j) not in owner:
                b.f([base[i][j], base[i + 1][j], base[i + 1][j + 1], base[i][j + 1]], mat=cell_mat(i, j), **col_tags(i))
    for blk in blocks:
        i0, i1, j0, j1, tp, dp = blk["i0"], blk["i1"], blk["j0"], blk["j1"], blk["taper"], blk["depth"]
        nb = {}
        for i in range(i0, i1 + 1):
            x, y = path[i]
            if i in (i0, i1):
                x2, y2 = path[i + 1] if i == i0 else path[i - 1]
                ln = math.hypot(x2 - x, y2 - y)
                x, y = x + (x2 - x) * tp / ln, y + (y2 - y) * tp / ln
            for j in range(j0, j1 + 1):
                z = zl[j] + (tp if j == j0 else (-tp if j == j1 else 0.0))
                p, u = deform(x, y, z)
                nb[i, j] = b.v(p[0] - dp * u[0], p[1] - dp * u[1], p[2] - dp * u[2])
        for i in range(i0, i1):
            for j in range(j0, j1):
                b.f([nb[i, j], nb[i + 1, j], nb[i + 1, j + 1], nb[i, j + 1]], mat=blk["mat"], **col_tags(i))
        rm = blk.get("rmat", "black")
        for i in range(i0, i1):
            b.f([base[i][j0], base[i + 1][j0], nb[i + 1, j0], nb[i, j0]], mat=rm, **col_tags(i))
            b.f([base[i][j1], base[i + 1][j1], nb[i + 1, j1], nb[i, j1]], mat=rm, **col_tags(i))
        for j in range(j0, j1):
            b.f([base[i0][j], base[i0][j + 1], nb[i0, j + 1], nb[i0, j]], mat=rm, **col_tags(i0))
            b.f([base[i1][j], base[i1][j + 1], nb[i1, j + 1], nb[i1, j]], mat=rm, **col_tags(i1 - 1))
    return base


def rrect(y0, z0, y1, z1, r, n=2):
    if r <= 0:
        return [(y1, z1), (y0, z1), (y0, z0), (y1, z0)]
    pts = []
    for (cy, cz, a0) in ((y1 - r, z1 - r, 0), (y0 + r, z1 - r, 90), (y0 + r, z0 + r, 180), (y1 - r, z0 + r, 270)):
        for i in range(n + 1):
            t = math.radians(a0 + 90.0 * i / n)
            pts.append((cy + r * math.cos(t), cz + r * math.sin(t)))
    return pts


def side_recess(b, sx, y0, y1, z0, z1, r, depth, taper, mat, rmat="black"):
    """a tapered recess in a flat side (a window, a grille); returns the loop that is the hole in the side."""
    x = sx * W
    lo = [b.v(x, y, z) for (y, z) in rrect(y0, z0, y1, z1, r)]
    li = [b.v(x - sx * depth, y, z) for (y, z) in rrect(y0 + taper, z0 + taper, y1 - taper, z1 - taper, max(0.0, r - taper))]
    n = len(lo)
    for k in range(n):
        b.f([lo[k], lo[(k + 1) % n], li[(k + 1) % n], li[k]], mat=rmat, **T_SIDE[sx])
    b.f(li, mat=mat, **T_SIDE[sx])
    return lo


def build_body(b):
    body = Body(b)
    # --- the front, wrapped around both corners --------------------------------------------------------------------
    fp = end_path(True)

    def col_f(i):
        x0, x1 = fp[i][0], fp[i + 1][0]
        return T_SIDE[-1] if (x0 == -W and x1 == -W) else (T_SIDE[1] if (x0 == W and x1 == W) else T_FRONT)

    def mat_f(i, j):
        return "black" if 4 <= j <= 6 else "skin"          # the mask around the windscreen and the display

    blocks_f = [
        dict(i0=0, i1=16, j0=0, j1=1, depth=-0.05, taper=0.025, mat="bumper", rmat="bumper"),
        dict(i0=4, i1=7, j0=2, j1=3, depth=0.025, taper=0.02, mat="lamp"),
        dict(i0=9, i1=12, j0=2, j1=3, depth=0.025, taper=0.02, mat="lamp"),
        dict(i0=4, i1=12, j0=4, j1=5, depth=0.02, taper=0.02, mat="glass"),
        dict(i0=4, i1=12, j0=6, j1=7, depth=0.025, taper=0.02, mat="display"),
    ]
    base_f = wrap_grid(b, body, fp, ZLF, col_f, mat_f, blocks_f)

    # --- the rear ---------------------------------------------------------------------------------------------------
    rp = end_path(False)

    def col_r(i):
        x0, x1 = rp[i][0], rp[i + 1][0]
        return T_SIDE[1] if (x0 == W and x1 == W) else (T_SIDE[-1] if (x0 == -W and x1 == -W) else T_REAR)

    blocks_r = [
        dict(i0=0, i1=16, j0=0, j1=1, depth=-0.04, taper=0.02, mat="bumper", rmat="bumper"),
        dict(i0=7, i1=9, j0=2, j1=3, depth=0.025, taper=0.02, mat="grille"),
        dict(i0=7, i1=9, j0=4, j1=5, depth=0.025, taper=0.02, mat="grille"),
        dict(i0=4, i1=6, j0=3, j1=5, depth=0.02, taper=0.015, mat="tail"),
        dict(i0=10, i1=12, j0=3, j1=5, depth=0.02, taper=0.015, mat="tail"),
        dict(i0=6, i1=10, j0=6, j1=7, depth=0.025, taper=0.02, mat="glass"),
    ]
    base_r = wrap_grid(b, body, rp, ZLR, col_r, lambda i, j: "skin", blocks_r)

    # --- the roof ---------------------------------------------------------------------------------------------------
    xs = [-W] + [-W + RX * f for f in reversed(F6)] + list(XIN) + [W - RX * f for f in F6] + [W]
    ys = [YR] + [YR + RYR * f for f in reversed(F6)] + list(YMID) + [YF - RYF * f for f in F6] + [YF]
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            b.f([body.node(xs[i], ys[j], ZT), body.node(xs[i + 1], ys[j], ZT), body.node(xs[i + 1], ys[j + 1], ZT),
                 body.node(xs[i], ys[j + 1], ZT)], mat="skin", **T_ROOF)

    # --- the flat sides ----------------------------------------------------------------------------------------------
    ysl = [YR + RYR] + list(YMID) + [YF - RYF]
    zc = [ZT - RZ, ZT - RZ * F6[1], ZT - RZ * F6[2], ZT]
    t0 = -math.asin((WHEEL_R - ZB) / RA)
    wells = {}
    for sx in (-1, 1):
        x = sx * W
        for i in range(len(ysl) - 1):                      # the cantrail above the flat side
            for j in range(3):
                b.f([body.node(x, ysl[i], zc[j]), body.node(x, ysl[i + 1], zc[j]), body.node(x, ysl[i + 1], zc[j + 1]),
                     body.node(x, ysl[i], zc[j + 1])], mat="skin", **T_SIDE[sx])
        for ax, ya in (("F", YA_F), ("R", YA_R)):          # wheel arches and wells
            name = "well_%s%s" % (ax, "L" if sx < 0 else "R")
            arc = [(ya + RA * math.cos(t0 + (math.pi - 2 * t0) * k / NA), WHEEL_R + RA * math.sin(t0 + (math.pi - 2 * t0) * k / NA))
                   for k in range(NA + 1)]                 # front end -> over the top -> rear end
            A = [b.v(x, y, z) for (y, z) in arc]
            B = [b.v(sx * XWI, y, z) for (y, z) in arc]
            ds = RA * (math.pi - 2 * t0) / NA
            for k in range(NA):
                luv = {A[k]: (k * ds, 0.0), A[k + 1]: ((k + 1) * ds, 0.0), B[k + 1]: ((k + 1) * ds, W - XWI), B[k]: (k * ds, W - XWI)}
                b.f([A[k], A[k + 1], B[k + 1], B[k]], obj="body", part=name, chart=name, luv=luv, mat="under", dens=0.3)
            b.f(B, obj="body", part=name + "b", chart=name + "b", view="+X" if sx > 0 else "-X", mat="under", dens=0.3)
            wells[sx, ax] = (A, B)
        # outline of the flat side: bottom (rear -> front) with the arches, front edge up, top back, rear edge down
        outer = [body.node(x, YR + RYR, ZB)] + list(reversed(wells[sx, "R"][0])) + list(reversed(wells[sx, "F"][0]))
        outer += [body.node(x, YF - RYF, z) for z in ZLF[:8]]
        outer += [body.node(x, y, ZT - RZ) for y in reversed(YMID)]
        outer += [body.node(x, YR + RYR, z) for z in reversed(ZLR[1:9])]
        holes = []
        for (y0, y1) in (WIN_L if sx < 0 else WIN_R):
            holes.append(side_recess(b, sx, y0, y1, WIN_Z0, WIN_Z1, 0.07, 0.03, 0.02, "glass"))
        if sx < 0:                                         # engine air intake at the left rear
            holes.append(side_recess(b, sx, YR + 0.51, YR + 1.18, 1.40, 2.30, 0.0, 0.025, 0.02, "grille"))
        else:
            for k, (y0, y1) in enumerate(DOORS):           # door openings with their jambs
                rect = ((y0, ZD0), (y1, ZD0), (y1, ZD1), (y0, ZD1))
                lo = [b.v(W, y, z) for (y, z) in rect]
                li = [b.v(XI, y, z) for (y, z) in rect]
                per = [0.0, y1 - y0, y1 - y0 + ZD1 - ZD0, 2 * (y1 - y0) + ZD1 - ZD0, 2 * (y1 - y0 + ZD1 - ZD0)]
                for e in range(4):
                    e2 = (e + 1) % 4
                    luv = {lo[e]: (per[e], 0.0), lo[e2]: (per[e + 1], 0.0), li[e2]: (per[e + 1], W - XI), li[e]: (per[e], W - XI)}
                    b.f([lo[e], lo[e2], li[e2], li[e]], obj="body", part="jamb%d" % (k + 1), chart="jamb%d" % (k + 1), luv=luv,
                        mat="jamb", dens=0.6)
                holes.append(lo)
        tri_fill(b, outer, holes, lambda p: (p[1], p[2]), mat="skin", **T_SIDE[sx])

    # --- the underside ----------------------------------------------------------------------------------------------
    def notch(sx, ax, forward):
        A, B = wells[sx, ax]
        seq = [A[NA], B[NA], B[0], A[0]]                   # rear end -> front end
        return seq if forward else list(reversed(seq))

    outline = [base_f[i][0] for i in range(17)]            # front, left -> right
    outline += notch(1, "F", False) + notch(1, "R", False)  # right side, front -> rear
    outline += [base_r[i][0] for i in range(17)]            # rear, right -> left
    outline += notch(-1, "R", True) + notch(-1, "F", True)  # left side, rear -> front
    tri_fill(b, outline, [], lambda p: (p[0], p[1]), obj="body", part="under", chart="under", view="-Z", mat="under", dens=0.12)


def loft_y(b, stations, z0, obj, part, mat, dens=0.9):
    """a roof unit: sections along y, each (y, height, half width), chamfered top edges, closed."""
    rings = []
    for (y, h, w) in stations:
        c = min(0.07, 0.4 * h)
        rings.append([(w, y, z0), (w, y, z0 + h - c), (w - c, y, z0 + h), (-(w - c), y, z0 + h), (-w, y, z0 + h - c), (-w, y, z0)])
    t = dict(obj=obj, part=part, mat=mat, dens=dens)

    def tag(s, i, q):
        return dict(obj=obj, part=part, mat="under", dens=dens * 0.2) if i == 5 else dict(t)

    b.rings(rings, tag, cap0=dict(t), cap1=dict(t))


def tube(b, pts, r, n, tags):
    """a bar of n sides along the points, closed at both ends."""
    rings = []
    for k, p in enumerate(pts):
        d = (Vector(pts[min(k + 1, len(pts) - 1)]) - Vector(pts[max(k - 1, 0)])).normalized()
        right = d.cross(Vector((0, 0, 1))).normalized()
        up = right.cross(d)
        rings.append([tuple(Vector(p) + r * (math.cos(2 * math.pi * (i + 0.5) / n) * right + math.sin(2 * math.pi * (i + 0.5) / n) * up))
                      for i in range(n)])
    b.rings(rings, lambda s, i, q: dict(tags), cap0=dict(tags), cap1=dict(tags))


def build_roof_and_mirrors(b):
    z0 = ZT - 0.02                                         # sunk 2 cm into the roof, so nothing is coplanar with it
    loft_y(b, [(YF - 4.75, 0.12, 0.90), (YF - 4.45, 0.31, 0.95), (YF - 1.75, 0.31, 0.95), (YF - 1.10, 0.14, 0.90)], z0, "body", "ac", "ac")
    loft_y(b, [(YR + 0.61, 0.07, 0.70), (YR + 0.81, 0.16, 0.75), (YR + 1.71, 0.16, 0.75), (YR + 1.96, 0.07, 0.70)], z0, "body", "pod", "ac")
    for k, yc in enumerate(HATCHES):
        b.box(-0.42, 0.42, yc - 0.42, yc + 0.42, z0, ZT + 0.07, "body", "hatch%d" % k, "hatch", dens=0.9, top_inset=0.03, zmid=ZT + 0.04)
    b.box(-0.17, 0.17, YF - 0.97, YF - 0.75, z0, ZT + 0.06, "body", "antenna", "hatch", dens=0.9, top_inset=0.03, zmid=ZT + 0.03)
    for sx, sn in ((-1, "L"), (1, "R")):
        # the arm leaves the body above the windscreen corner and reaches forward and outward
        tube(b, [(sx * 1.21, YF - 0.65, 2.66), (sx * 1.33, YF - 0.29, 2.63), (sx * 1.40, YF - 0.04, 2.47)], 0.028, 6,
             dict(obj="body", part="mirA" + sn, mat="black", dens=0.8))
        xa, xb = sorted((sx * 1.30, sx * 1.50))
        hs = [[(xa, y, 2.02), (xb, y, 2.02), (xb, y, 2.50), (xa, y, 2.50)] for y in (YF - 0.09, YF - 0.015)]
        hs.append([(xa + 0.03, YF + 0.015, 2.05), (xb - 0.03, YF + 0.015, 2.05), (xb - 0.03, YF + 0.015, 2.47), (xa + 0.03, YF + 0.015, 2.47)])
        t = dict(obj="body", part="mir" + sn, mat="black", dens=0.8)
        b.rings(hs, lambda s, i, q, t=t: dict(t), cap0=dict(t, mat="mirror"), cap1=dict(t))


def build_wheel(b, name, ya, sx, twin):
    n, R = WHEEL_SEGS, WHEEL_R
    wd = WHEEL_W["R" if twin else "F"]
    if twin:      # two tyres with a groove between them, a deep dish
        tread = [(R, wd - 0.03), (R, wd / 2 + 0.035), (R - 0.04, wd / 2 + 0.012), (R - 0.04, wd / 2 - 0.012), (R, wd / 2 - 0.035), (R, 0.03)]
        face = [(R - 0.03, 0.0), (0.315, 0.0), (0.295, 0.02), (0.255, 0.11), (0.16, 0.11), (0.13, 0.06)]
    else:         # one tyre, the hub standing out
        tread = [(R, wd - 0.03), (R, 0.03)]
        face = [(R - 0.03, 0.0), (0.315, 0.0), (0.295, 0.02), (0.25, 0.07), (0.16, 0.05), (0.13, 0.0)]
    prof = [(R - 0.03, wd)] + tread + face                 # (radius, depth behind the outer face), inside -> outside
    nt = len(tread)
    rings = [[(sx * (XO - d), ya + r * math.cos(2 * math.pi * i / n), R + r * math.sin(2 * math.pi * i / n)) for i in range(n)]
             for (r, d) in prof]
    vout, vin = ("+X", "-X") if sx > 0 else ("-X", "+X")
    du = 2 * math.pi * R / n
    acc = [0.0]
    for k in range(1, nt):
        acc.append(acc[-1] + math.hypot(tread[k][0] - tread[k - 1][0], tread[k][1] - tread[k - 1][1]))
    T = dict(obj=name, part=name)

    def tag(s, i, q):
        if s == 0:
            return dict(T, mat="tyre", chart=name + "_in", view=vin, dens=0.2)
        if s < nt:
            v0, v1 = acc[s - 1], acc[s]
            luv = {q[0]: (i * du, v0), q[1]: ((i + 1) * du, v0), q[2]: ((i + 1) * du, v1), q[3]: (i * du, v1)}
            return dict(T, mat="tyre", chart=name + "_tr", luv=luv, dens=0.5)
        k = s - nt                                         # 0 shoulder, 1 sidewall, 2 rim lip, 3 rim, 4 disc, ...
        return dict(T, mat="tyre" if k <= 1 else ("rim" if k <= 4 else "hub"), chart=name + "_out", view=vout, dens=1.0)

    b.rings(rings, tag, cap0=dict(T, mat="tyre", chart=name + "_in", view=vin, dens=0.2),
            cap1=dict(T, mat="hub", chart=name + "_out", view=vout, dens=1.0))


def build_door_leaf(b, name, ya, yb):
    """one door leaf between ya and yb: a black frame around glass, on both faces."""
    T = dict(obj=name, part=name)
    fr, frb, gd, gt = 0.065, 0.14, 0.012, 0.012            # frame width, bottom rail, glass depth, taper

    def face(x, sgn, view, dens):
        t = dict(T, view=view, dens=dens)
        o = [b.v(x, y, z) for (y, z) in ((ya, ZD0), (yb, ZD0), (yb, ZD1), (ya, ZD1))]
        g0 = ((ya + fr, ZD0 + frb), (yb - fr, ZD0 + frb), (yb - fr, ZD1 - fr), (ya + fr, ZD1 - fr))
        g1 = ((ya + fr + gt, ZD0 + frb + gt), (yb - fr - gt, ZD0 + frb + gt), (yb - fr - gt, ZD1 - fr - gt), (ya + fr + gt, ZD1 - fr - gt))
        h = [b.v(x, y, z) for (y, z) in g0]
        g = [b.v(x - sgn * gd, y, z) for (y, z) in g1]
        for k in range(4):
            k2 = (k + 1) % 4
            b.f([o[k], o[k2], h[k2], h[k]], mat="black", **t)
            b.f([h[k], h[k2], g[k2], g[k]], mat="black", **t)
        b.f(g, mat="glass", **t)
        return o

    o = face(XDO, 1, "+X", 1.0)
    i = face(XDI, -1, "-X", 0.5)
    for k in range(4):
        k2 = (k + 1) % 4
        b.f([o[k], o[k2], i[k2], i[k]], mat="black", dens=0.5, **T)


def seat(b, part, xa, xb, y0, z0, zs, zbk, depth=0.44, bt=0.09):
    """a seat block facing the front: base and backrest in one piece."""
    prof = [(y0, z0), (y0 + depth, z0), (y0 + depth, zs), (y0 + bt, zs), (y0 + bt, zbk), (y0, zbk)]
    t = dict(obj="interior", part=part, mat="seat", dens=0.3)

    def tag(s, i, q):
        return dict(t, mat="under", dens=0.1) if i == 0 else dict(t)

    b.rings([[(x, y, z) for (y, z) in prof] for x in (xa, xb)], tag, cap0=dict(t), cap1=dict(t))


def build_interior(b):
    R = dict(obj="interior", part="room", inward=True, dens=0.2)
    c = {}
    for sx in (-1, 1):
        for y in (YI0, YI1):
            for z in (ZF, ZC):
                c[sx, y, z] = b.v(sx * XI, y, z)
    # the right wall, with a notch for each door; the notch corners are also on the floor's edge
    wall = [c[1, YI0, ZF]]
    floor_edge = []
    for (y0, y1) in sorted(DOORS):
        n = [b.v(XI, y0, ZF), b.v(XI, y0, ZD1), b.v(XI, y1, ZD1), b.v(XI, y1, ZF)]
        wall += n
        floor_edge += [n[0], n[3]]
    wall += [c[1, YI1, ZF], c[1, YI1, ZC], c[1, YI0, ZC]]
    tri_fill(b, wall, [], lambda p: (p[1], p[2]), mat="wall", **R)
    tri_fill(b, [c[-1, YI0, ZF], c[1, YI0, ZF]] + floor_edge + [c[1, YI1, ZF], c[-1, YI1, ZF]], [], lambda p: (p[0], p[1]), mat="floor", **R)
    b.f([c[-1, YI0, ZC], c[1, YI0, ZC], c[1, YI1, ZC], c[-1, YI1, ZC]], mat="ceil", **R)
    b.f([c[-1, YI0, ZF], c[-1, YI1, ZF], c[-1, YI1, ZC], c[-1, YI0, ZC]], mat="wall", **R)
    b.f([c[-1, YI0, ZF], c[1, YI0, ZF], c[1, YI0, ZC], c[-1, YI0, ZC]], mat="wall", **R)
    b.f([c[-1, YI1, ZF], c[1, YI1, ZF], c[1, YI1, ZC], c[-1, YI1, ZC]], mat="wall", **R)
    # wheel housings: boxes over the wheel wells, seats on top of them
    zh = 1.13
    for sx, sn in ((-1, "L"), (1, "R")):
        for ax, ya in (("F", YA_F), ("R", YA_R)):
            xa, xb = sorted((sx * 0.55, sx * (XI + 0.01)))
            b.box(xa, xb, ya - 0.63, ya + 0.63, ZF - 0.01, zh, "interior", "housing_%s%s" % (ax, sn), "housing", dens=0.3)
    # double seats along both walls (rear edge y0 of each), on the floor or on a wheel housing
    zf = ZF - 0.01
    rule_poles = None
    if SPEC.get("interior_rule"):       # the kit's rule (shape_guide.seat_rows, door_poles) instead of placed rows
        sys.path.insert(0, HERE)
        import shape_guide as sg
        housings = [(ya - 0.63, ya + 0.63) for ya in (YA_F, YA_R)]
        start = YI0 + 0.02 + 0.50 + 0.05            # in front of the rear bench
        rows = {-1: sg.seat_rows(start, YF - 1.70, 0.80, 0.44, housings=housings),
                1: sg.seat_rows(start, YF - 1.70, 0.80, 0.44, keep_out=DOORS, housings=housings)}
        rule_poles = sg.door_poles(DOORS, 0.50, YF - 1.65)
    else:
        rows = {-1: [(3.0, True), (1.95, False), (1.15, False), (0.35, False), (-0.45, False), (-1.25, False), (-2.9, True), (-4.1, False)],
                1: [(3.0, True), (1.75, False), (-0.75, False), (-1.55, False), (-2.9, True)]}
    for sx, sn in ((-1, "L"), (1, "R")):
        xa, xb = sorted((sx * 0.24, sx * 1.12))
        for k, (y0, high) in enumerate(rows[sx]):
            if high:
                seat(b, "seat_%s%d" % (sn, k), xa, xb, y0, zh - 0.01, zh + 0.26, zh + 0.74)
            else:
                seat(b, "seat_%s%d" % (sn, k), xa, xb, y0, zf, ZF + 0.46, ZF + 0.94)
    seat(b, "seat_bench", -1.12, 1.12, YI0 + 0.02, zf, ZF + 0.60, ZF + 1.12, depth=0.50)
    # the driver's place: seat, dashboard, steering wheel, a partition behind the seat
    seat(b, "seat_driver", -0.85, -0.37, YF - 1.50, zf, ZF + 0.52, ZF + 1.12, depth=0.46)
    b.box(-1.14, -0.08, YF - 0.93, YF - 0.58, zf, 1.10, "interior", "dash", "dash", dens=0.3)
    b.box(-1.14, -0.30, YF - 1.65, YF - 1.61, zf, 1.95, "interior", "partition", "housing", dens=0.3)
    cen, axis = Vector((-0.61, YF - 0.97, 1.06)), Vector((0.0, -0.5, 0.866))
    e1, e2 = Vector((1, 0, 0)), Vector((0.0, 0.866, 0.5))
    sw = [[tuple(cen + axis * o + 0.21 * (math.cos(2 * math.pi * i / 10) * e1 + math.sin(2 * math.pi * i / 10) * e2)) for i in range(10)]
          for o in (-0.02, 0.02)]
    t = dict(obj="interior", part="steering", mat="dash", dens=0.3)
    b.rings(sw, lambda s, i, q: dict(t), cap0=dict(t), cap1=dict(t))
    # handrail poles at the doors and two rails under the ceiling
    poles = rule_poles if rule_poles is not None else [(0.50, 3.92), (0.50, -0.10), (0.50, 1.25), (0.50, -4.62), (0.50, -3.32), (-0.20, 0.78), (-0.20, -2.00)]
    for k, (x, y) in enumerate(poles):
        b.box(x - 0.02, x + 0.02, y - 0.02, y + 0.02, zf, ZC + 0.01, "interior", "pole%d" % k, "pole", dens=0.3, bottom="pole")
    for k, x in enumerate((-0.50, 0.50)):
        b.box(x - 0.017, x + 0.017, YR + 1.06, YF - 1.75, 2.06, 2.10, "interior", "rail%d" % k, "pole", dens=0.3, bottom="pole")
    for k, (x, y) in enumerate([(-0.50, YR + 1.06), (-0.50, YF - 1.75), (0.50, YR + 1.06), (0.50, YF - 1.75), (-0.50, -0.3), (0.50, -1.2)]):
        b.box(x - 0.015, x + 0.015, y - 0.015, y + 0.015, 2.09, ZC + 0.01, "interior", "hanger%d" % k, "pole", dens=0.3, bottom="pole")


def build_all():
    b = Builder()
    build_body(b)
    build_roof_and_mirrors(b)
    build_wheel(b, "wheel_FL", YA_F, -1, False)
    build_wheel(b, "wheel_FR", YA_F, 1, False)
    build_wheel(b, "wheel_RL", YA_R, -1, True)
    build_wheel(b, "wheel_RR", YA_R, 1, True)
    for k, (y0, y1) in enumerate(DOORS):
        ym = (y0 + y1) / 2
        build_door_leaf(b, "door_%da" % (k + 1), ym, y1)
        build_door_leaf(b, "door_%db" % (k + 1), y0, ym)
    if SPEC.get("interior", True):
        build_interior(b)
    return b


# ---------------------------------------------------------------------------------------------------------------------
# bmesh: consistent, outward normals per shell (the interior room faces inward)
# ---------------------------------------------------------------------------------------------------------------------
def make_bmesh(b):
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in b.verts]
    bm.verts.ensure_lookup_table()
    tags = []
    for idx, t in b.faces:
        bm.faces.new([vs[i] for i in idx])
        tags.append(t)
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    seen = set()
    flipped = shells = 0
    for f0 in bm.faces:
        if f0.index in seen:
            continue
        comp = [f0]
        seen.add(f0.index)
        k = 0
        while k < len(comp):
            f = comp[k]
            k += 1
            for e in f.edges:
                for g in e.link_faces:
                    if g.index not in seen:
                        seen.add(g.index)
                        comp.append(g)
        vol = 0.0
        for f in comp:
            vv = [l.vert.co for l in f.loops]
            for i in range(1, len(vv) - 1):
                vol += vv[0].dot(vv[i].cross(vv[i + 1])) / 6.0
        shells += 1
        inward = any(tags[f.index].get("inward") for f in comp)
        if (vol < 0) != inward:
            bmesh.ops.reverse_faces(bm, faces=comp)
            flipped += 1
    bm.normal_update()
    print("BUILD shells: %d, flipped: %d" % (shells, flipped))
    return bm, tags


# ---------------------------------------------------------------------------------------------------------------------
# UV charts and packing
# ---------------------------------------------------------------------------------------------------------------------
VIEWS = {
    "-X": lambda p: (-p[1], p[2]), "+X": lambda p: (p[1], p[2]),
    "+Y": lambda p: (-p[0], p[2]), "-Y": lambda p: (p[0], p[2]),
    "+Z": lambda p: (p[1], -p[0]), "-Z": lambda p: (p[1], p[0]),
}
NAT = {  # natural 2D world coords (h, v) -> chart coords (a, b)
    "-X": lambda h, v: (-h, v), "+X": lambda h, v: (h, v),      # (y, z)
    "+Y": lambda h, v: (-h, v), "-Y": lambda h, v: (h, v),      # (x, z)
    "+Z": lambda h, v: (v, -h), "-Z": lambda h, v: (v, h),      # (x, y)
}


def auto_view(n):
    w = (1.02, 1.01, 1.0)
    ax = max(range(3), key=lambda i: abs(n[i]) * w[i])
    return ("+" if n[ax] > 0 else "-") + "XYZ"[ax]


class Island:
    def __init__(self, key):
        self.key = key
        self.faces = []        # (face_index, [(a, b) ...], mat)
        self.view = None
        self.dens = 0.0
        self.obj = None

    def finish(self):
        aa = [c[0] for _, cs, _ in self.faces for c in cs]
        bb = [c[1] for _, cs, _ in self.faces for c in cs]
        self.amin, self.amax, self.bmin, self.bmax = min(aa), max(aa), min(bb), max(bb)
        area = {}
        for _, cs, m in self.faces:
            ar = 0.0
            for i in range(len(cs)):
                x0, y0 = cs[i]
                x1, y1 = cs[(i + 1) % len(cs)]
                ar += x0 * y1 - x1 * y0
            area[m] = area.get(m, 0.0) + abs(ar)
        self.bg = max(area, key=area.get)

    def size(self, scale):
        ppm = BASE_PPM * self.dens * scale
        w = max(2, int(math.ceil((self.amax - self.amin) * ppm))) + 2 * PAD
        h = max(2, int(math.ceil((self.bmax - self.bmin) * ppm))) + 2 * PAD
        return w, h, ppm


def make_islands(bm, tags):
    isl = {}
    for f in bm.faces:
        t = tags[f.index]
        view = t.get("view", "auto")
        if "luv" in t:
            cs = [t["luv"][l.vert.index] for l in f.loops]
            key, vname = t["chart"], None
        else:
            if view == "auto":
                view = auto_view(f.normal)
            cs = [VIEWS[view](l.vert.co) for l in f.loops]
            vname = view
            key = t.get("chart") or (t["part"], view)
        I = isl.get(key)
        if I is None:
            I = isl[key] = Island(key)
            I.view = vname
            I.obj = t["obj"]
        I.faces.append((f.index, cs, t["mat"]))
        I.dens = max(I.dens, t.get("dens", 1.0))
    for I in isl.values():
        I.finish()
    return isl


def maxrects_pack(sizes, wb, hb):
    free = np.array([[0, 0, wb, hb]], dtype=np.int64)       # x, y, w, h
    out = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: (-max(sizes[i]), -sizes[i][0] * sizes[i][1]))
    for i in order:
        w, h = sizes[i]
        fits = (free[:, 2] >= w) & (free[:, 3] >= h)
        if not fits.any():
            return None
        cand = free[fits]
        k = np.lexsort((cand[:, 0], cand[:, 1], np.minimum(cand[:, 2] - w, cand[:, 3] - h)))[0]
        px, py = int(cand[k, 0]), int(cand[k, 1])
        out[i] = (px, py)
        fx, fy, fw, fh = free[:, 0], free[:, 1], free[:, 2], free[:, 3]
        hit = ~((px >= fx + fw) | (px + w <= fx) | (py >= fy + fh) | (py + h <= fy))
        new = []
        for (ax, ay, aw, ah) in free[hit].tolist():
            if px > ax:
                new.append((ax, ay, px - ax, ah))
            if px + w < ax + aw:
                new.append((px + w, ay, ax + aw - (px + w), ah))
            if py > ay:
                new.append((ax, ay, aw, py - ay))
            if py + h < ay + ah:
                new.append((ax, py + h, aw, ay + ah - (py + h)))
        allr = free[~hit]
        if new:
            allr = np.concatenate([allr, np.array(new, dtype=np.int64).reshape(-1, 4)])
        if len(allr) == 0:
            free = allr
            continue
        allr = np.unique(allr, axis=0)
        x0, y0 = allr[:, 0], allr[:, 1]
        x1, y1 = x0 + allr[:, 2], y0 + allr[:, 3]
        cont = (x0[:, None] >= x0[None, :]) & (y0[:, None] >= y0[None, :]) & (x1[:, None] <= x1[None, :]) & (y1[:, None] <= y1[None, :])
        np.fill_diagonal(cont, False)
        free = allr[~cont.any(axis=1)]
    return out


def pack_islands(isl):
    L = list(isl.values())
    lo, hi = 0.3, 2.0
    best = None
    for _ in range(14):
        mid = (lo + hi) / 2
        sizes = [I.size(mid)[:2] for I in L]
        res = maxrects_pack(sizes, TEX, TEX)
        if res is None:
            hi = mid
        else:
            lo = mid
            best = (mid, res, sizes)
    scale, res, sizes = best
    for I, (x, y), (w, h) in zip(L, res, sizes):
        I.x, I.y, I.w, I.h = x, y, w, h
        I.ppm = BASE_PPM * I.dens * scale
    used = sum(w * h for (w, h) in sizes)
    print("BUILD atlas %d px: %d charts, main density %.1f px/m, rect fill %.1f %%" % (TEX, len(L), BASE_PPM * scale, 100.0 * used / TEX / TEX))
    return scale


def write_uvs(bm, isl):
    uv = bm.loops.layers.uv.new("UVMap")
    for I in isl.values():
        for fi, cs, _ in I.faces:
            f = bm.faces[fi]
            for l, (a, c) in zip(f.loops, cs):
                px = I.x + PAD + (a - I.amin) * I.ppm
                py = I.y + PAD + (I.bmax - c) * I.ppm
                l[uv].uv = (px / TEX, 1.0 - py / TEX)


# ---------------------------------------------------------------------------------------------------------------------
# painting
# ---------------------------------------------------------------------------------------------------------------------
class Canvas:
    def __init__(self, n):
        self.n = n
        self.col = np.zeros((n, n, 3), np.float32)
        self.col[:] = C["under"]
        self.cov = np.zeros((n, n), bool)      # texels that belong to a face
        self.clip = (0, 0, n, n)
        self.mark = False

    def _apply(self, x0, y0, x1, y1, m, col):
        if self.mark:
            self.cov[y0:y1, x0:x1] |= m
        if col is not None:
            self.col[y0:y1, x0:x1][m] = np.asarray(col, np.float32)

    def _bbox(self, xs, ys):
        cx0, cy0, cx1, cy1 = self.clip
        return (max(cx0, int(math.floor(min(xs)))), max(cy0, int(math.floor(min(ys)))),
                min(cx1, int(math.ceil(max(xs))) + 1), min(cy1, int(math.ceil(max(ys))) + 1))

    def tri(self, p0, p1, p2, col):
        x0, y0, x1, y1 = self._bbox((p0[0], p1[0], p2[0]), (p0[1], p1[1], p2[1]))
        if x1 <= x0 or y1 <= y0:
            return
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(area) < 1e-7:
            return
        s = 1.0 if area > 0 else -1.0
        X = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
        Y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
        m = None
        for a, c in ((p0, p1), (p1, p2), (p2, p0)):
            e = s * ((c[0] - a[0]) * (Y - a[1]) - (c[1] - a[1]) * (X - a[0])) >= -1e-3
            m = e if m is None else (m & e)
        self._apply(x0, y0, x1, y1, m, col)

    def poly(self, pts, col):
        if len(pts) == 3:
            self.tri(pts[0], pts[1], pts[2], col)
            return
        for t in tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in pts]]):
            self.tri(pts[t[0]], pts[t[1]], pts[t[2]], col)

    def circle(self, cx, cy, r, col, r_in=0.0):
        x0, y0, x1, y1 = self._bbox((cx - r, cx + r), (cy - r, cy + r))
        if x1 <= x0 or y1 <= y0:
            return
        X = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
        Y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        m = d2 <= r * r
        if r_in > 0:
            m &= d2 >= r_in * r_in
        self._apply(x0, y0, x1, y1, m, col)


class Painter:
    def __init__(self, cv, I):
        self.cv, self.I = cv, I
        self.k = I.ppm * SS
        self.ox = (I.x + PAD) * SS - I.amin * self.k
        self.oy = (I.y + PAD) * SS + I.bmax * self.k
        cv.clip = (I.x * SS, I.y * SS, (I.x + I.w) * SS, (I.y + I.h) * SS)
        self.nat = NAT.get(I.view, lambda h, v: (h, v))

    def px(self, a, c):
        return (self.ox + a * self.k, self.oy - c * self.k)

    def P(self, h, v):
        return self.px(*self.nat(h, v))

    def fill(self, col):
        x0, y0, x1, y1 = self.cv.clip
        self.cv.col[y0:y1, x0:x1] = np.asarray(col, np.float32)

    def poly(self, pts, col):
        self.cv.poly([self.P(h, v) for (h, v) in pts], col)

    def rect(self, h0, v0, h1, v1, col):
        self.poly([(h0, v0), (h1, v0), (h1, v1), (h0, v1)], col)

    def rrect(self, h0, v0, h1, v1, r, col, n=5):
        pts = []
        for (cx, cy, a0) in ((h1 - r, v1 - r, 0), (h0 + r, v1 - r, 90), (h0 + r, v0 + r, 180), (h1 - r, v0 + r, 270)):
            for i in range(n + 1):
                t = math.radians(a0 + 90.0 * i / n)
                pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
        self.poly(pts, col)

    def circle(self, h, v, r, col, r_in=0.0):
        cx, cy = self.P(h, v)
        self.cv.circle(cx, cy, r * self.k, col, r_in * self.k)

    def line(self, p0, p1, width, col):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * width / 2, dx / ln * width / 2
        self.poly([(p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny), (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)], col)

    def frame(self, h0, v0, h1, v1, width, col):
        for a, c in (((h0, v0), (h1, v0)), ((h1, v0), (h1, v1)), ((h1, v1), (h0, v1)), ((h0, v1), (h0, v0))):
            self.line(a, c, width, col)


GREY = (0.42, 0.43, 0.43)       # panel seams


def livery_side(P, sx):
    """under the windows and doors: the yellow band at the sill, the black window band, the arch trims."""
    P.rect(Y_SIDE0, 1.10, Y_SIDE1, 1.25, C["yellow"])
    P.rect(BAND_Y0[sx], 1.25, 7.0, 2.50, C["black"])
    for ya in (YA_F, YA_R):
        P.circle(ya, WHEEL_R, RA + 0.035, C["trim"], r_in=RA - 0.05)


def livery_rear(P):
    P.rrect(-1.06, 1.54, 1.06, 2.54, 0.09, C["black"])      # the frame around the rear window


def detail_front(P):
    P.rect(-0.27, 0.375, 0.27, 0.505, GREY)
    P.rect(-0.26, 0.385, 0.26, 0.495, C["plate"])           # blank number plate
    P.frame(-1.06, 2.535, 1.06, 2.705, 0.012, (0.22, 0.22, 0.23))      # the bezel of the destination display
    for sx in (-1, 1):
        for xc in (0.73, 0.89):                             # two round lenses per side, the indicator at the corner
            P.circle(sx * xc, 0.78, 0.072, (0.55, 0.55, 0.53), r_in=0.058)
        xa, xb = sorted((sx * 0.99, sx * 1.105))
        P.rect(xa, 0.69, xb, 0.87, C["amber"])
        xa, xb = sorted((sx * 0.08, sx * 0.86))             # wipers, parked
        P.line((xa if sx > 0 else xb, 1.235), (xb if sx > 0 else xa, 1.29), 0.022, C["black"])


def detail_rear(P):
    P.frame(-0.94, 0.625, 0.94, 1.50, 0.012, GREY)          # engine cover seam
    P.rect(-0.27, 0.935, 0.27, 1.065, GREY)
    P.rect(-0.26, 0.945, 0.26, 1.055, C["plate"])
    P.rect(-0.30, 2.60, 0.30, 2.635, C["tail"])             # high brake light
    for sx in (-1, 1):
        xa, xb = sorted((sx * 0.995, sx * 1.11))
        P.rect(xa, 1.085, xb, 1.21, C["amber"])
        P.rect(xa, 0.94, xb, 1.065, C["revers"])
    for z0 in (0.69, 1.13):                                 # slats of the two grilles
        for k in range(5):
            z = z0 + 0.02 + k * 0.05
            P.rect(-0.57, z, 0.57, z + 0.012, (0.16, 0.16, 0.165))


def detail_side(P, sx):
    if sx < 0:
        for k in range(15):                                 # slats of the engine air intake
            z = 1.44 + k * 0.057
            P.rect(YR + 0.54, z, YR + 1.15, z + 0.014, (0.16, 0.16, 0.165))
        P.frame(YR + 1.66, 0.62, YR + 2.11, 0.95, 0.01, GREY)       # fuel flap
    for y in (Y_SIDE0 + 0.9, -0.9, Y_SIDE1 - 0.5):          # small amber side markers on the skirt
        P.rect(y - 0.05, 0.60, y + 0.05, 0.635, C["amber"])


def detail_ac(P):
    P.rrect(-0.78, 1.62, 0.78, 4.08, 0.05, (0.60, 0.61, 0.62))
    for yc in (2.10, 2.85, 3.60):
        P.circle(0.0, yc, 0.30, (0.20, 0.20, 0.21))
        P.circle(0.0, yc, 0.27, (0.36, 0.37, 0.38))
        for i in range(6):
            t = math.pi * i / 6
            P.line((-0.27 * math.cos(t), yc - 0.27 * math.sin(t)), (0.27 * math.cos(t), yc + 0.27 * math.sin(t)), 0.014, (0.20, 0.20, 0.21))
        P.circle(0.0, yc, 0.06, (0.20, 0.20, 0.21))


def detail_pod(P):
    P.rrect(-0.58, YR + 0.88, 0.58, YR + 1.64, 0.03, (0.24, 0.24, 0.25))
    for k in range(9):
        y = YR + 0.91 + k * 0.08
        P.rect(-0.55, y, 0.55, y + 0.03, (0.50, 0.51, 0.52))


def detail_hatch(P, yc):
    P.rrect(-0.34, yc - 0.34, 0.34, yc + 0.34, 0.05, (0.56, 0.57, 0.58))
    P.rrect(-0.30, yc - 0.30, 0.30, yc + 0.30, 0.04, C["hatch"])


def detail_wheel(P, ya, twin):
    zc = WHEEL_R
    P.circle(ya, zc, 0.405, (0.11, 0.11, 0.115), r_in=0.395)            # a ring on the tyre wall
    P.circle(ya, zc, 0.30, (0.50, 0.51, 0.52), r_in=0.285)              # the shadow under the rim lip
    for i in range(8):                                                  # hand holes in the disc
        t = 2 * math.pi * (i + 0.5) / 8
        P.circle(ya + 0.21 * math.cos(t), zc + 0.21 * math.sin(t), 0.026, (0.10, 0.10, 0.105))
    P.circle(ya, zc, 0.085, (0.30, 0.30, 0.31))
    for i in range(10):                                                 # wheel nuts
        t = 2 * math.pi * i / 10
        P.circle(ya + 0.115 * math.cos(t), zc + 0.115 * math.sin(t), 0.012, (0.70, 0.71, 0.72))


def paint_all(isl):
    n = TEX * SS
    cv = Canvas(n)
    for I in isl.values():
        P = Painter(cv, I)
        key = I.key
        livery = None
        if key in (("body", "-X"), ("body", "+X")):
            sx = -1 if key[1] == "-X" else 1
            livery = lambda P=P, sx=sx: livery_side(P, sx)
        elif key == ("body", "-Y"):
            livery = lambda P=P: livery_rear(P)
        elif key in (("body", "+Y"), ("body", "+Z")):
            livery = lambda: None
        P.fill(C["skin"] if livery is not None else C[I.bg])
        if livery is not None:
            livery()
        cv.mark = True
        for fi, cs, m in I.faces:
            skip = (m == "skin") if livery is not None else (m == I.bg)
            cv.poly([P.px(a, c) for (a, c) in cs], None if skip else C[m])
        cv.mark = False
        if key == ("body", "+Y"):
            detail_front(P)
        elif key == ("body", "-Y"):
            detail_rear(P)
        elif key in (("body", "-X"), ("body", "+X")):
            detail_side(P, -1 if key[1] == "-X" else 1)
        elif key == ("ac", "+Z"):
            detail_ac(P)
        elif key == ("pod", "+Z"):
            detail_pod(P)
        elif key == ("hatch0", "+Z"):
            detail_hatch(P, HATCHES[0])
        elif key == ("hatch1", "+Z"):
            detail_hatch(P, HATCHES[1])
        elif isinstance(key, str) and key.startswith("wheel_") and key.endswith("_out"):
            detail_wheel(P, YA_F if key[6] == "F" else YA_R, key[6] == "R")
        elif isinstance(key, str) and key.endswith("_tr"):
            x0, y0, x1, y1 = cv.clip                        # tread blocks
            A = ((np.arange(x0, x1, dtype=np.float32) + 0.5) - P.ox) / P.k
            stripes = (np.mod(A * 12.0, 1.0) < 0.5)[None, :, None]
            sub = cv.col[y0:y1, x0:x1]
            sub[:] = sub * (1.0 - 0.35 * stripes) + np.asarray((0.12, 0.12, 0.125), np.float32) * (0.35 * stripes)
    cv.clip = (0, 0, n, n)
    col = cv.col.reshape(TEX, SS, TEX, SS, 3).mean(axis=(1, 3))
    # a texel counts as painted only if all of it lies on a face: the mixed texels along a chart's edge are
    # overwritten from inside by dilate(), so no line of the background colour shows along a seam
    cov = cv.cov.reshape(TEX, SS, TEX, SS).all(axis=(1, 3))
    return col, cov


def box_blur(a, r):
    k = 2 * r + 1
    for ax in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float64)
        a = ((c[k:] - c[:-k]) if ax == 0 else (c[:, k:] - c[:, :-k])) / k
    return a.astype(np.float32)


def dilate(col, cov, passes):
    """spread the charts' colours into their gutters, so the atlas does not bleed at mip levels."""
    cov = cov.copy()
    for _ in range(passes):
        new = cov.copy()
        for sh in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            src_cov = np.roll(cov, sh, axis=(0, 1))
            m = (~new) & src_cov
            if m.any():
                col[m] = np.roll(col, sh, axis=(0, 1))[m]
                new |= m
        cov = new
    return col


def save_png(arr, path, name):
    h, w, _ = arr.shape
    img = bpy.data.images.new(name + "_tmp", w, h, alpha=False)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[:, :, :3] = np.clip(arr, 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(path)
    img.name = name
    return img


# ---------------------------------------------------------------------------------------------------------------------
# objects, checks, export
# ---------------------------------------------------------------------------------------------------------------------
def enable_gpu(scene):
    scene.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == "OPTIX")
        scene.cycles.device = "GPU"
    except Exception as e:
        print("BUILD GPU setup failed, CPU:", e)


def mesh_object(scene, bm, tags, name, pivot, keep, mat):
    bm2 = bm.copy()
    lay = bm2.faces.layers.int["tag"]
    drop = [f for f in bm2.faces if not keep(tags[f[lay]])]
    if drop:
        bmesh.ops.delete(bm2, geom=drop, context="FACES")
    bmesh.ops.translate(bm2, verts=bm2.verts[:], vec=-Vector(pivot))
    me = bpy.data.meshes.new(name)
    bm2.to_mesh(me)
    bm2.free()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = pivot
    scene.collection.objects.link(ob)
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    bpy.context.view_layer.update()
    try:      # weighted normals, so the big flat panels stay flat next to the rounded edges
        mod = ob.modifiers.new("WN", "WEIGHTED_NORMAL")
        mod.keep_sharp = True
        mod.weight = 60
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print("BUILD weighted normal failed on %s: %s" % (name, e))
    return ob


def check_uv_overlap(me, n=2048):
    uv = me.uv_layers.active.data
    uvs = np.array([d.uv[:] for d in uv])
    cnt = np.zeros((n, n), np.uint8)
    for p in me.polygons:
        P = [(uvs[i][0] * n, uvs[i][1] * n) for i in p.loop_indices]
        p0, p1, p2 = P[0], P[1], P[2]
        area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(area) < 1e-6:
            continue
        s = 1.0 if area > 0 else -1.0
        x0, x1 = max(0, int(math.floor(min(q[0] for q in P)))), min(n, int(math.ceil(max(q[0] for q in P))) + 1)
        y0, y1 = max(0, int(math.floor(min(q[1] for q in P)))), min(n, int(math.ceil(max(q[1] for q in P))) + 1)
        X = (np.arange(x0, x1) + 0.5)[None, :]
        Y = (np.arange(y0, y1) + 0.5)[:, None]
        m = None
        for a, c in ((p0, p1), (p1, p2), (p2, p0)):
            ln = math.hypot(c[0] - a[0], c[1] - a[1]) or 1.0
            e = s * ((c[0] - a[0]) * (Y - a[1]) - (c[1] - a[1]) * (X - a[0])) / ln > 0.75      # inside by 0.75 px
            m = e if m is None else (m & e)
        cnt[y0:y1, x0:x1] += m.astype(np.uint8)
    print("BUILD uv range u %.4f..%.4f v %.4f..%.4f | texels of a %d grid claimed by more than one triangle: %d of %d" % (
        uvs[:, 0].min(), uvs[:, 0].max(), uvs[:, 1].min(), uvs[:, 1].max(), n, int((cnt > 1).sum()), int((cnt > 0).sum())))


def check_shells(bm, tags):
    lay = bm.faces.layers.int["tag"]
    by_obj = {}
    for e in bm.edges:
        if len(e.link_faces) != 2:
            o = tags[e.link_faces[0][lay]]["obj"] if e.link_faces else "?"
            by_obj[o] = by_obj.get(o, 0) + 1
    print("BUILD open or non-manifold edges per object (the door openings of body and interior are open on purpose):", by_obj or "none")


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    os.makedirs(OUT, exist_ok=True)
    b = build_all()
    bm, tags = make_bmesh(b)
    isl = make_islands(bm, tags)
    pack_islands(isl)
    write_uvs(bm, isl)
    lay = bm.faces.layers.int.new("tag")
    for f in bm.faces:
        f[lay] = f.index
        f.smooth = True
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle() > math.radians(35.0):
            e.smooth = False
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.faces.ensure_lookup_table()
    check_shells(bm, tags)

    mat = bpy.data.materials.new("M_" + NAME)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Roughness"].default_value = 0.55
    bsdf.inputs["Metallic"].default_value = 0.0

    joined = mesh_object(scene, bm, tags, NAME, (0.0, 0.0, 0.0), lambda t: True, mat)
    check_uv_overlap(joined.data)

    print("BUILD painting ... (%.1f s)" % (time.time() - T0))
    col, cov = paint_all(isl)
    print("BUILD painted (%.1f s)" % (time.time() - T0))
    if DO_AO:
        enable_gpu(scene)
        scene.cycles.samples = 64
        scene.cycles.use_denoising = False
        if scene.world is None:
            scene.world = bpy.data.worlds.new("W")
        scene.world.light_settings.distance = 1.2
        gm = bpy.data.meshes.new("tmp_ground")
        gm.from_pydata([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [], [(0, 1, 2, 3)])
        gob = bpy.data.objects.new("tmp_ground", gm)
        scene.collection.objects.link(gob)
        aoimg = bpy.data.images.new("tmp_ao", TEX, TEX, alpha=False)
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = aoimg
        nt.nodes.active = node
        scene.render.bake.margin = 8
        try:
            scene.render.bake.margin_type = "EXTEND"
        except Exception:
            pass
        bpy.ops.object.select_all(action="DESELECT")
        joined.select_set(True)
        bpy.context.view_layer.objects.active = joined
        bpy.ops.object.bake(type="AO")
        buf = np.empty(TEX * TEX * 4, np.float32)
        aoimg.pixels.foreach_get(buf)
        ao = box_blur(buf.reshape(TEX, TEX, 4)[::-1, :, 0].copy(), 3)
        del buf
        nt.nodes.remove(node)
        bpy.data.images.remove(aoimg)
        bpy.data.objects.remove(gob)
        bpy.data.meshes.remove(gm)
        for I in isl.values():          # wheels turn: no baked shadow on them
            if I.obj.startswith("wheel"):
                ao[I.y:I.y + I.h, I.x:I.x + I.w] = 1.0
        print("BUILD AO baked (%.1f s), mean %.3f" % (time.time() - T0, float(ao.mean())))
        col *= (1.0 - 0.5 * (1.0 - np.clip(ao, 0, 1)))[:, :, None]
    col = dilate(col, cov, PAD + 2)
    tex_path = os.path.join(OUT, NAME + "_basecolor.png")
    img = save_png(col, tex_path, NAME + "_basecolor")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    tex.location = (-420, 300)
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    def export(path, objs):
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_yup=True, export_apply=True)
        print("BUILD wrote %s (%.2f MB)" % (path, os.path.getsize(path) / 1e6))

    co = np.array([v.co[:] for v in joined.data.vertices])
    mn, mx = co.min(axis=0), co.max(axis=0)
    total = len(joined.data.polygons)
    print("BUILD joined: %d triangles, %d vertices, bounds min %s max %s, size %s" % (
        total, len(co), np.round(mn, 3), np.round(mx, 3), np.round(mx - mn, 3)))
    export(os.path.join(OUT, NAME + "_joined.glb"), [joined])
    jm = joined.data
    bpy.data.objects.remove(joined)
    bpy.data.meshes.remove(jm)

    parts = []
    for name in OBJECTS:
        ob = mesh_object(scene, bm, tags, name, PIVOTS[name], lambda t, name=name: t["obj"] == name, mat)
        parts.append(ob)
        print("BUILD part %-9s origin (%7.3f, %7.3f, %6.3f)  triangles %5d  vertices %5d" % (
            name, ob.location.x, ob.location.y, ob.location.z, len(ob.data.polygons), len(ob.data.vertices)))
    assert sum(len(o.data.polygons) for o in parts) == total
    bm.free()
    export(os.path.join(OUT, NAME + "_parts.glb"), parts)
    img.pack()
    blend = os.path.join(OUT, NAME + "_parts.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    if os.path.isfile(blend + "1"):       # Blender's backup of the file a rebuild replaced
        os.remove(blend + "1")
    print("BUILD wrote %s (%.2f MB), done in %.1f s" % (blend, os.path.getsize(blend) / 1e6, time.time() - T0))


main()
