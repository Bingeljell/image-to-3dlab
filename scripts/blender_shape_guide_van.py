# Outbreak / method comparison M1 (Fable, hand-scripted Blender model)
# Box ambulance van: procedural mesh + UV atlas + painted 1024 texture + AO bake.
# Run:  blender -b -P build_ambulance.py -- [--no-ao]
import bpy, bmesh, math, sys, os, time
import numpy as np
from mathutils import Matrix, Vector
from mathutils.geometry import tessellate_polygon

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def arg(name, default=None):
    return ARGS[ARGS.index(name) + 1] if name in ARGS else default


OUT = os.path.abspath(arg("--out", os.path.dirname(HERE)))
os.makedirs(OUT, exist_ok=True)
DO_AO = "--no-ao" not in ARGS

TEX = int(arg("--tex", "1024"))   # final atlas size
SS = 4 if TEX <= 1024 else 1      # supersampling for painting (a 4096 atlas needs none)
PAD = 3 * TEX // 1024             # gutter px (each side of every island), 3 at 1024
NAME = arg("--name", "ambulance")
# the kit's parts: each wheel is its own object on its axle (Fable's tags wFL.. -> the kit's names)
WHEELS = {"wFL": "wheel_FL", "wFR": "wheel_FR", "wRL": "wheel_RL", "wRR": "wheel_RR"}
BASE_PPM = 120.0 * TEX / 1024   # px per metre before the auto-fit scale (120 at a 1024 atlas)
WHEEL_SEGS = 24

# ----------------------------------------------------------------------------
# colours (sRGB display values)
# ----------------------------------------------------------------------------
C = dict(
    white=(0.86, 0.865, 0.85), roofwhite=(0.83, 0.835, 0.82), alu=(0.70, 0.71, 0.72),
    yellow=(0.84, 0.91, 0.10), red=(0.85, 0.11, 0.09), black=(0.055, 0.055, 0.06),
    dark=(0.11, 0.11, 0.115), glass=(0.045, 0.06, 0.075), glass_hi=(0.10, 0.135, 0.165),
    blue=(0.10, 0.27, 0.92), silver=(0.66, 0.68, 0.70), tyre=(0.075, 0.075, 0.08),
    rim=(0.58, 0.59, 0.60), hub=(0.33, 0.34, 0.35), under=(0.045, 0.045, 0.047),
    ac=(0.74, 0.75, 0.75), mirror=(0.42, 0.50, 0.56), amber=(0.95, 0.55, 0.08),
    lamp=(0.84, 0.86, 0.88), tail=(0.70, 0.05, 0.05), dirt=(0.36, 0.32, 0.27),
    plate=(0.88, 0.88, 0.86),
)
EMIT = dict(blue=(0.06, 0.28, 1.0), tail=(0.55, 0.02, 0.02))
E_BLUE = EMIT['blue']
E_RED = (0.85, 0.04, 0.03)

# the van family's presets (the kit's numbers; the ambulance is the hand-built original)
AMBULANCE = dict(box=(1.025, -3.35, 0.62, 0.40, 2.74), rear_axle=-1.825, livery="ambulance", lightbar=True)
PRESETS = {
    "ambulance": AMBULANCE,
    # a long panel van: the box 0.8 m longer with the rear axle following, a plain livery for the picture model
    "van_long": dict(AMBULANCE, box=(1.025, -4.15, 0.62, 0.40, 2.74), rear_axle=-2.30, livery="plain", lightbar=False),
    # a box van: a wider, taller and longer box
    "box_van": dict(AMBULANCE, box=(1.10, -3.95, 0.62, 0.40, 3.05), rear_axle=-2.15, livery="plain", lightbar=False),
}
SPEC = PRESETS[arg("--preset", "ambulance")]
LIVERY = SPEC["livery"]
if LIVERY == "plain":      # no ambulance markings: red and yellow become the body's white, the blue lights go dark
    C = dict(C, yellow=C["white"], red=C["white"], blue=C["dark"])
    EMIT = dict(tail=EMIT["tail"])

# ----------------------------------------------------------------------------
# builder
# ----------------------------------------------------------------------------
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
        """rings: list of closed loops (lists of xyz). Quads between consecutive loops, n-gon caps."""
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

    def box(self, x0, x1, y0, y1, z0, z1, part, mat, dens=1.0, top_inset=0.0, zmid=None, mats=None):
        """axis box, optionally with a tapered (chamfered) top section starting at zmid."""
        def rect(z, ins):
            return [(x0 + ins, y0 + ins, z), (x1 - ins, y0 + ins, z), (x1 - ins, y1 - ins, z), (x0 + ins, y1 - ins, z)]
        rr = [rect(z0, 0.0)]
        if top_inset > 0:
            rr += [rect(zmid, 0.0), rect(z1, top_inset)]
        else:
            rr += [rect(z1, 0.0)]
        t = dict(part=part, mat=mat, dens=dens)
        self.rings(rr, lambda s, i, q: dict(t), cap0=dict(part=part, mat='under', dens=dens * 0.3), cap1=dict(t))


def lerp(a, b, t):
    return a + (b - a) * t


def tab(table, x):
    if x <= table[0][0]:
        return table[0][1]
    for (x0, v0), (x1, v1) in zip(table, table[1:]):
        if x <= x1:
            return lerp(v0, v1, (x - x0) / (x1 - x0))
    return table[-1][1]


# ----------------------------------------------------------------------------
# dimensions
# ----------------------------------------------------------------------------
BX, BY0, BY1, BZ0, BZ1 = SPEC["box"]                         # the box (the ambulance's patient box)
BR, BNS = 0.07, 3                                            # box edge radius / segments
RA_Y = SPEC["rear_axle"]
RA_Z, RA_R, RA_N = 0.36, 0.46, 12                            # rear wheel arch
FA_Y, FA_Z, FA_R = 2.50, 0.36, 0.44                          # front wheel arch
WHEEL_R, WHEEL_W = 0.36, 0.26
CAB_W = 0.97
CAB_ST = [0.55, 1.45, 1.75, 2.06, 2.10, 2.19, 2.33, 2.50, 2.67, 2.81, 2.90, 2.94, 3.12, 3.30, 3.40, 3.455, 3.48]
YR, YC = 2.06, 2.67                                          # windscreen top / cowl station
ZR = [(0.55, 2.43), (1.45, 2.42), (1.75, 2.395), (2.06, 2.32)]
ZH = [(2.67, 1.50), (2.81, 1.458), (2.94, 1.415), (3.12, 1.34), (3.30, 1.24), (3.40, 1.16), (3.455, 1.10), (3.48, 1.05)]
WM = [(0.55, CAB_W), (2.94, CAB_W), (3.12, 0.965), (3.30, 0.93), (3.40, 0.885), (3.455, 0.84), (3.48, 0.79)]
BELT = 1.40


def cab_zbot(y):
    dy = abs(y - FA_Y)
    if dy >= FA_R - 1e-6:
        return 0.30
    return max(0.30, FA_Z + math.sqrt(FA_R ** 2 - dy ** 2))


def cabin_upper(wm, zr):   # rows 3..10
    return [(wm - 0.008, BELT), (wm - 0.115, zr - 0.20), (wm - 0.145, zr - 0.10), (wm - 0.18, zr - 0.055),
            (wm - 0.24, zr - 0.022), (wm - 0.34, zr - 0.006), (wm * 0.45, zr + 0.008), (0.0, zr + 0.015)]


def hood_upper(wm, zh, d):
    ze = zh - d
    return [(wm - 0.008, ze), (wm - 0.03, ze + 0.35 * d), (wm - 0.065, ze + 0.60 * d), (wm - 0.11, ze + 0.78 * d),
            (wm - 0.17, ze + 0.86 * d), (wm - 0.32, zh - 0.006), (wm * 0.45, zh - 0.002), (0.0, zh)]


def cab_section(y):
    wm = tab(WM, y)
    if y <= YR:
        up = cabin_upper(wm, tab(ZR, y))
    elif y >= YC:
        d = 0.10 - 0.03 * (y - YC) / (CAB_ST[-1] - YC)
        up = hood_upper(wm, tab(ZH, y), d)
    else:
        t = (y - YR) / (YC - YR)
        a = cabin_upper(CAB_W, tab(ZR, YR))
        b = hood_upper(CAB_W, tab(ZH, YC), 0.10)
        up = [(lerp(p[0], q[0], t), lerp(p[1], q[1], t)) for p, q in zip(a, b)]
    zb = cab_zbot(y)
    zw = min(0.92, up[0][1] - 0.10)
    k = (zw - zb) / 0.62
    low = [(wm - 0.035 * k, zb), (wm - 0.010 * k, zb + 0.4 * (zw - zb)), (wm, zw)]
    return low + up     # 11 points, last = centre line


# ----------------------------------------------------------------------------
# parts
# ----------------------------------------------------------------------------
def build_box(b):
    xb, y0, y1, zb, zt = BX, BY0, BY1, BZ0, BZ1
    R, ns = BR, BNS
    th0 = math.asin((zb - RA_Z) / RA_R)

    def upper(inset, rc):
        pts = [(y0 + inset, zb)]
        cy, cz = y0 + inset + rc, zt - inset - rc
        for k in range(ns + 1):
            t = math.pi - k * (math.pi / 2) / ns
            pts.append((cy + rc * math.cos(t), cz + rc * math.sin(t)))
        cy = y1 - inset - rc
        for k in range(ns + 1):
            t = math.pi / 2 - k * (math.pi / 2) / ns
            pts.append((cy + rc * math.cos(t), cz + rc * math.sin(t)))
        pts.append((y1 - inset, zb))
        return pts

    arch = []
    for j in range(RA_N + 1):
        t = th0 + (math.pi - 2 * th0) * j / RA_N
        arch.append((RA_Y + RA_R * math.cos(t), RA_Z + RA_R * math.sin(t)))   # front -> rear
    yf, yr = arch[0][0], arch[-1][0]
    s_f = y1 - yf
    arclen = RA_R * (math.pi - 2 * th0)

    def under_uv(p):
        x, y, z = p
        if z > zb + 1e-5:
            t = math.atan2(z - RA_Z, y - RA_Y)
            s = s_f + RA_R * (t - th0)
        elif y > RA_Y:
            s = y1 - y
        else:
            s = s_f + arclen + (yr - y)
        return (s, x)

    U = dict(part='box', mat='under', chart='box_under', luvf=under_uv, dens=0.14)
    Wt = dict(part='box', mat='white')
    ring = {}
    A = {}
    for sx in (-1, 1):
        ring[sx] = []
        for k in range(ns + 1):
            ph = k * (math.pi / 2) / ns
            x = xb - R + R * math.sin(ph)
            inset = R * (1 - math.cos(ph))
            rc = R - 0.75 * inset
            ring[sx].append([b.v(sx * x, y, z) for (y, z) in upper(inset, rc)])
        A[sx] = [b.v(sx * xb, y, z) for (y, z) in arch]
    NU = len(ring[1][0])
    L, Rr = ring[-1], ring[1]
    for i in range(NU - 1):
        b.f([L[0][i], L[0][i + 1], Rr[0][i + 1], Rr[0][i]], **Wt)
    for sx in (-1, 1):
        r = ring[sx]
        for k in range(ns):
            for i in range(NU - 1):
                b.f([r[k][i], r[k][i + 1], r[k + 1][i + 1], r[k + 1][i]], **Wt)
        b.f(r[ns] + A[sx], part='box', mat='white', view='+X' if sx > 0 else '-X')
    e = NU - 1
    b.f([L[k][e] for k in range(ns + 1)] + [A[-1][0], A[1][0]] + [Rr[k][e] for k in range(ns, -1, -1)], **U)
    b.f([Rr[k][0] for k in range(ns + 1)] + [A[1][RA_N], A[-1][RA_N]] + [L[k][0] for k in range(ns, -1, -1)], **U)
    for j in range(RA_N):
        b.f([A[-1][j], A[-1][j + 1], A[1][j + 1], A[1][j]], **U)


def build_cab(b):
    rings = []
    for y in CAB_ST:
        half = cab_section(y)
        rings.append([(x, y, z) for (x, z) in half] + [(-x, y, z) for (x, z) in reversed(half[:-1])])
    n = len(rings[0])            # 21
    s_r, s_c = CAB_ST.index(YR), CAB_ST.index(YC)
    top = cabin_upper(CAB_W, tab(ZR, YR))[-1][1]
    bot = hood_upper(CAB_W, tab(ZH, YC), 0.10)[-1][1]
    bv = Vector((0.0, YR - YC, top - bot)).normalized()
    ws_view = ('custom', Vector((-1, 0, 0)), bv)

    def tag(s, i, q):
        if i == n - 1:
            return dict(part='cab', mat='under', chart='cab_under', view='-Z', dens=0.14)
        rl = i if i <= 9 else 19 - i
        if s_r <= s < s_c and rl >= 7:
            return dict(part='cab', mat='glass', chart='cab_ws', view=ws_view, dens=0.8)
        return dict(part='cab', mat='white')

    b.rings(rings, tag, cap0=dict(part='cab', mat='under', chart='cab_back', view='-Y', dens=0.1),
            cap1=dict(part='cab', mat='yellow', view='+Y'))


def build_wheel(b, name, yc, xo, sx):
    R, W, n = WHEEL_R, WHEEL_W, WHEEL_SEGS
    prof = [(0.30, W), (R, W - 0.03), (R, 0.035), (0.335, 0.0), (0.225, 0.004), (0.205, 0.03), (0.10, 0.05), (0.065, 0.03)]
    rings = []
    for (r, d) in prof:
        rings.append([(sx * (xo - d), yc + r * math.cos(2 * math.pi * i / n), WHEEL_R + r * math.sin(2 * math.pi * i / n))
                      for i in range(n)])
    vout = '+X' if sx > 0 else '-X'
    vin = '-X' if sx > 0 else '+X'
    du = 2 * math.pi * R / n
    wv = prof[1][1] - prof[2][1]

    def tag(s, i, q):
        if s == 0:
            return dict(part=name, mat='tyre', chart=name + '_in', view=vin, dens=0.14)
        if s == 1:
            luv = {q[0]: (i * du, 0.0), q[1]: ((i + 1) * du, 0.0), q[2]: ((i + 1) * du, wv), q[3]: (i * du, wv)}
            return dict(part=name, mat='tyre', chart=name + '_tr', luv=luv, dens=0.45)
        mat = 'tyre' if s <= 3 else ('rim' if s <= 5 else 'hub')
        return dict(part=name, mat=mat, chart=name + '_out', view=vout, dens=0.7)

    b.rings(rings, tag, cap0=dict(part=name, mat='tyre', chart=name + '_in', view=vin, dens=0.14),
            cap1=dict(part=name, mat='hub', chart=name + '_out', view=vout, dens=0.7))


def build_flare(b, name, yc, zc, r0, r1, x_out, x_body, x_in, sx, n=8):
    rings = []
    a0, a1 = math.radians(-8.0), math.radians(188.0)
    for j in range(n + 1):
        t = a0 + (a1 - a0) * j / n
        c, s = math.cos(t), math.sin(t)
        rings.append([(sx * x_out, yc + r0 * c, zc + r0 * s), (sx * (x_body + 0.004), yc + r1 * c, zc + r1 * s),
                      (sx * x_in, yc + r0 * c, zc + r0 * s)])
    vout = '+X' if sx > 0 else '-X'
    dth = (a1 - a0) / n

    def tag(s, i, q):
        if i == 0:
            return dict(part=name, mat='black', chart=name + '_o', view=vout, dens=0.7)
        if i == 1:
            luv = {q[0]: (s * dth * r1, 0.0), q[1]: (s * dth * r1, 0.12), q[2]: ((s + 1) * dth * r1, 0.12), q[3]: ((s + 1) * dth * r1, 0.0)}
            return dict(part=name, mat='under', chart=name + '_b', luv=luv, dens=0.14)
        luv = {q[0]: (s * dth * r0, 0.0), q[1]: (s * dth * r0, x_out - x_in), q[2]: ((s + 1) * dth * r0, x_out - x_in), q[3]: ((s + 1) * dth * r0, 0.0)}
        return dict(part=name, mat='dark', chart=name + '_i', luv=luv, dens=0.3)

    capt = dict(part=name, mat='black', dens=0.3, view='-Z')
    b.rings(rings, tag, cap0=dict(capt), cap1=dict(capt))


def build_lightbar(b):
    prof = [(-0.15, 0.0), (-0.15, 0.07), (-0.11, 0.122), (-0.04, 0.14), (0.04, 0.14), (0.11, 0.122), (0.15, 0.07), (0.15, 0.0)]
    xs = [-0.62, -0.55, -0.20, -0.18, 0.18, 0.20, 0.55, 0.62]
    sc = [0.78, 1.0, 1.0, 0.85, 0.85, 1.0, 1.0, 0.78]
    yc, zb = 1.28, 2.438
    rings = [[(x, yc + dy * s, zb + dz * s) for (dy, dz) in prof] for x, s in zip(xs, sc)]
    mats = ['blue', 'blue', 'dark', 'silver', 'dark', 'blue', 'blue']

    def tag(s, i, q):
        if i == len(prof) - 1:
            return dict(part='lbar', mat='under', dens=0.3)
        return dict(part='lbar', mat=mats[s])

    b.rings(rings, tag, cap0=dict(part='lbar', mat='blue'), cap1=dict(part='lbar', mat='blue'))
    b.box(-0.50, 0.50, yc - 0.09, yc + 0.09, 2.37, 2.446, 'lbar_base', 'dark', dens=0.5)


def build_all():
    b = Builder()
    build_box(b)
    build_cab(b)
    # wheels (FL, FR, RL, RR) and arch flares
    for name, yc, xo, sx in (('wFL', FA_Y, 0.955, -1), ('wFR', FA_Y, 0.955, 1), ('wRL', RA_Y, 1.0, -1), ('wRR', RA_Y, 1.0, 1)):
        build_wheel(b, name, yc, xo, sx)
    for sx, sn in ((-1, 'L'), (1, 'R')):
        build_flare(b, 'flF' + sn, FA_Y, FA_Z, 0.425, 0.505, CAB_W + 0.028, CAB_W, CAB_W - 0.09, sx)
        build_flare(b, 'flR' + sn, RA_Y, RA_Z, 0.452, 0.53, BX + 0.028, BX, BX - 0.09, sx)
    if SPEC["lightbar"]:
        build_lightbar(b)
        # corner beacons on the box roof
        for sx, sn in ((-1, 'L'), (1, 'R')):
            xa, xb_ = sorted((sx * 0.77, sx * (BX + 0.012)))
            b.box(xa, xb_, BY1 - 0.29, BY1 + 0.012, BZ1 - 0.10, BZ1 + 0.11, 'clF' + sn, 'blue', top_inset=0.04, zmid=BZ1 + 0.06)
            b.box(xa, xb_, BY0 - 0.012, BY0 + 0.29, BZ1 - 0.10, BZ1 + 0.11, 'clR' + sn, 'blue', top_inset=0.04, zmid=BZ1 + 0.06)
    # roof equipment
    b.box(-0.48, 0.48, BY1 - 0.64, BY1 - 0.12, BZ1 - 0.04, BZ1 + 0.105, 'ac', 'ac', dens=0.85, top_inset=0.035, zmid=BZ1 + 0.06)
    b.box(-0.27, 0.27, BY0 + 0.10, BY0 + 0.36, BZ1 - 0.04, BZ1 + 0.045, 'vent', 'ac', dens=0.85, top_inset=0.025, zmid=BZ1 + 0.015)
    # front bumper (prism following the nose)
    half = [(1.0, 3.02), (1.0, 3.12), (0.965, 3.30), (0.92, 3.41), (0.87, 3.47), (0.80, 3.50)]
    poly = half + [(-x, y) for (x, y) in reversed(half)]
    def btag(s, i, q):
        if i == len(poly) - 1:
            return dict(part='bumpF', mat='under', chart='bumpF_back', view='-Y', dens=0.1)
        return dict(part='bumpF', mat='black', dens=0.8)
    b.rings([[(x, y, 0.26) for (x, y) in poly], [(x, y, 0.61) for (x, y) in poly]], btag,
            cap0=dict(part='bumpF', mat='under', dens=0.2), cap1=dict(part='bumpF', mat='black', dens=0.8))
    # rear step / bumper, tail light housings
    b.box(-0.94, 0.94, BY0 - 0.15, BY0 + 0.02, 0.33, 0.43, 'stepR', 'black', dens=0.6)
    for sx, sn in ((-1, 'L'), (1, 'R')):
        xa, xb_ = sorted((sx * 0.74, sx * 1.0))
        b.box(xa, xb_, BY0 - 0.035, BY0 + 0.01, 0.47, 0.67, 'tail' + sn, 'tail', dens=0.9)
        # mud flaps behind rear wheels, cab entry steps
        xa, xb_ = sorted((sx * 0.72, sx * 1.01))
        b.box(xa, xb_, RA_Y - 0.55, RA_Y - 0.525, 0.10, 0.42, 'flap' + sn, 'black', dens=0.3)
        xa, xb_ = sorted((sx * 0.80, sx * 0.995))
        b.box(xa, xb_, 1.02, 2.0, 0.24, 0.31, 'cstep' + sn, 'black', dens=0.4)
        # mirrors: arm + housing
        xa, xb_ = sorted((sx * 0.92, sx * 1.05))
        b.box(xa, xb_, 2.385, 2.425, 1.49, 1.53, 'mirA' + sn, 'black', dens=0.4)
        xa, xb_ = sorted((sx * 1.03, sx * 1.175))
        hs = [[(xa, 2.36, 1.44), (xb_, 2.36, 1.44), (xb_, 2.36, 1.74), (xa, 2.36, 1.74)],
              [(xa, 2.43, 1.44), (xb_, 2.43, 1.44), (xb_, 2.43, 1.74), (xa, 2.43, 1.74)],
              [(xa + 0.025, 2.46, 1.47), (xb_ - 0.025, 2.46, 1.47), (xb_ - 0.025, 2.46, 1.71), (xa + 0.025, 2.46, 1.71)]]
        b.rings(hs, lambda s, i, q, sn=sn: dict(part='mir' + sn, mat='black', dens=0.6),
                cap0=dict(part='mir' + sn, mat='mirror', dens=0.6), cap1=dict(part='mir' + sn, mat='black', dens=0.6))
        # cab door handles
        xa, xb_ = sorted((sx * 0.955, sx * 0.985))
        b.box(xa, xb_, 1.08, 1.25, 1.215, 1.255, 'hdl' + sn, 'black', dens=0.4)
    # rear door lock bars
    for x in (-0.07, 0.07):
        b.box(x - 0.015, x + 0.015, BY0 - 0.022, BY0 + 0.005, 1.25, BZ1 - 0.44, 'bar%d' % (x > 0), 'silver', dens=0.4)
    # chassis block
    b.box(-0.42, 0.42, BY0 + 0.10, 3.0, 0.20, 0.45, 'chassis', 'under', dens=0.1)
    return b


# ----------------------------------------------------------------------------
# bmesh creation
# ----------------------------------------------------------------------------
def make_bmesh(b):
    bm = bmesh.new()
    vs = [bm.verts.new(p) for p in b.verts]
    bm.verts.ensure_lookup_table()
    tags = []
    for idx, t in b.faces:
        f = bm.faces.new([vs[i] for i in idx])
        tags.append(t)
    bm.faces.ensure_lookup_table()
    bm.verts.index_update()
    bm.faces.index_update()
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.normal_update()
    # per-shell signed volume check -> flip inverted shells
    seen = set()
    flipped = 0
    shells = 0
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
        if vol < 0:
            bmesh.ops.reverse_faces(bm, faces=comp)
            flipped += 1
    bm.normal_update()
    print("shells:", shells, "flipped:", flipped)
    return bm, tags


# ----------------------------------------------------------------------------
# UV islands + packing
# ----------------------------------------------------------------------------
VIEWS = {
    '-X': lambda p: (-p[1], p[2]), '+X': lambda p: (p[1], p[2]),
    '+Y': lambda p: (-p[0], p[2]), '-Y': lambda p: (p[0], p[2]),
    '+Z': lambda p: (p[1], -p[0]), '-Z': lambda p: (p[1], p[0]),
}
NAT = {  # natural 2D world coords (h, v) -> (a, b)
    '-X': lambda h, v: (-h, v), '+X': lambda h, v: (h, v),      # (y, z)
    '+Y': lambda h, v: (-h, v), '-Y': lambda h, v: (h, v),      # (x, z)
    '+Z': lambda h, v: (v, -h), '-Z': lambda h, v: (v, h),      # (x, y)
}


def auto_view(n):
    w = (1.02, 1.01, 1.0)
    ax = max(range(3), key=lambda i: abs(n[i]) * w[i])
    return ('+' if n[ax] > 0 else '-') + 'XYZ'[ax]


class Island:
    def __init__(self, key):
        self.key = key
        self.faces = []        # (face_index, [(a,b)...], mat)
        self.view = None
        self.dens = 0.0
        self.part = None

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
        view = t.get('view', 'auto')
        if 'luv' in t:
            cs = [t['luv'][l.vert.index] for l in f.loops]
            key, vname = t['chart'], None
        elif 'luvf' in t:
            cs = [t['luvf'](tuple(l.vert.co)) for l in f.loops]
            key, vname = t['chart'], None
        else:
            if view == 'auto':
                view = auto_view(f.normal)
            if isinstance(view, tuple):
                cs = [(l.vert.co.dot(view[1]), l.vert.co.dot(view[2])) for l in f.loops]
                vname = None
            else:
                cs = [VIEWS[view](l.vert.co) for l in f.loops]
                vname = view
            key = t.get('chart') or (t['part'], view)
        I = isl.get(key)
        if I is None:
            I = isl[key] = Island(key)
            I.view = vname
            I.part = t['part']
        I.faces.append((f.index, cs, t['mat']))
        I.dens = max(I.dens, t.get('dens', 1.0))
    for I in isl.values():
        I.finish()
    return isl


def maxrects_pack(sizes, W, H):
    free = [(0, 0, W, H)]
    out = [None] * len(sizes)
    order = sorted(range(len(sizes)), key=lambda i: (-max(sizes[i]), -sizes[i][0] * sizes[i][1]))
    for i in order:
        w, h = sizes[i]
        best = None
        for (fx, fy, fw, fh) in free:
            if w <= fw and h <= fh:
                sc = (min(fw - w, fh - h), fy, fx)
                if best is None or sc < best[0]:
                    best = (sc, fx, fy)
        if best is None:
            return None
        px, py = best[1], best[2]
        out[i] = (px, py)
        nf = []
        for (fx, fy, fw, fh) in free:
            if px >= fx + fw or px + w <= fx or py >= fy + fh or py + h <= fy:
                nf.append((fx, fy, fw, fh))
                continue
            if px > fx:
                nf.append((fx, fy, px - fx, fh))
            if px + w < fx + fw:
                nf.append((px + w, fy, fx + fw - (px + w), fh))
            if py > fy:
                nf.append((fx, fy, fw, py - fy))
            if py + h < fy + fh:
                nf.append((fx, py + h, fw, fy + fh - (py + h)))
        # prune contained
        nf = list(set(nf))
        keep = []
        for a in nf:
            ok = True
            for c in nf:
                if a is not c and a != c and c[0] <= a[0] and c[1] <= a[1] and c[0] + c[2] >= a[0] + a[2] and c[1] + c[3] >= a[1] + a[3]:
                    ok = False
                    break
            if ok:
                keep.append(a)
        free = keep
    return out


def pack_islands(isl):
    L = list(isl.values())
    lo, hi = 0.3, 2.0
    best = None
    for _ in range(18):
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
    print("pack scale %.3f -> main density %.1f px/m, %d islands, rect fill %.1f%%" % (scale, BASE_PPM * scale, len(L), 100.0 * used / TEX / TEX))
    return scale


def write_uvs(bm, isl):
    uv = bm.loops.layers.uv.new("UVMap")
    for I in isl.values():
        for fi, cs, _ in I.faces:
            f = bm.faces[fi]
            for l, (a, b) in zip(f.loops, cs):
                px = I.x + PAD + (a - I.amin) * I.ppm
                py = I.y + PAD + (I.bmax - b) * I.ppm
                l[uv].uv = (px / TEX, 1.0 - py / TEX)


# ----------------------------------------------------------------------------
# painting
# ----------------------------------------------------------------------------
class Canvas:
    def __init__(self, n):
        self.n = n
        self.col = np.zeros((n, n, 3), np.float32)
        self.col[:] = C['under']
        self.emi = np.zeros((n, n, 3), np.float32)
        self.clip = (0, 0, n, n)

    def _apply(self, x0, y0, x1, y1, m, col, alpha, emit):
        sub = self.col[y0:y1, x0:x1]
        colv = np.asarray(col, np.float32)
        if alpha >= 1.0:
            sub[m] = colv
            self.emi[y0:y1, x0:x1][m] = np.asarray(emit if emit is not None else (0, 0, 0), np.float32)
        else:
            sub[m] = sub[m] * (1.0 - alpha) + colv * alpha

    def _bbox(self, xs, ys):
        cx0, cy0, cx1, cy1 = self.clip
        x0 = max(cx0, int(math.floor(min(xs))))
        x1 = min(cx1, int(math.ceil(max(xs))) + 1)
        y0 = max(cy0, int(math.floor(min(ys))))
        y1 = min(cy1, int(math.ceil(max(ys))) + 1)
        return x0, y0, x1, y1

    def tri(self, p0, p1, p2, col, alpha=1.0, emit=None):
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
        for a, b in ((p0, p1), (p1, p2), (p2, p0)):
            e = s * ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) >= -1e-3
            m = e if m is None else (m & e)
        self._apply(x0, y0, x1, y1, m, col, alpha, emit)

    def poly(self, pts, col, alpha=1.0, emit=None):
        if len(pts) == 3:
            self.tri(pts[0], pts[1], pts[2], col, alpha, emit)
            return
        if alpha < 1.0:
            # single mask to avoid double blending on shared edges
            x0, y0, x1, y1 = self._bbox([p[0] for p in pts], [p[1] for p in pts])
            if x1 <= x0 or y1 <= y0:
                return
            X = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
            Y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
            m = np.zeros((y1 - y0, x1 - x0), bool)
            for t in tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in pts]]):
                p0, p1, p2 = pts[t[0]], pts[t[1]], pts[t[2]]
                area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
                if abs(area) < 1e-7:
                    continue
                s = 1.0 if area > 0 else -1.0
                mm = None
                for a, b in ((p0, p1), (p1, p2), (p2, p0)):
                    e = s * ((b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0])) >= -1e-3
                    mm = e if mm is None else (mm & e)
                m |= mm
            self._apply(x0, y0, x1, y1, m, col, alpha, emit)
            return
        for t in tessellate_polygon([[Vector((p[0], p[1], 0.0)) for p in pts]]):
            self.tri(pts[t[0]], pts[t[1]], pts[t[2]], col, alpha, emit)

    def circle(self, cx, cy, r, col, alpha=1.0, emit=None, r_in=0.0):
        x0, y0, x1, y1 = self._bbox((cx - r, cx + r), (cy - r, cy + r))
        if x1 <= x0 or y1 <= y0:
            return
        X = (np.arange(x0, x1, dtype=np.float32) + 0.5)[None, :]
        Y = (np.arange(y0, y1, dtype=np.float32) + 0.5)[:, None]
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        m = d2 <= r * r
        if r_in > 0:
            m &= d2 >= r_in * r_in
        self._apply(x0, y0, x1, y1, m, col, alpha, emit)


_TEXT = {}


def text_tris(body, bold, spacing):
    key = (body, bold, spacing)
    if key in _TEXT:
        return _TEXT[key]
    cu = bpy.data.curves.new('tmp_txt', 'FONT')
    cu.body = body
    cu.size = 1.0
    cu.offset = bold
    cu.space_character = spacing
    ob = bpy.data.objects.new('tmp_txt', cu)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    obe = ob.evaluated_get(dg)
    me = obe.to_mesh()
    me.calc_loop_triangles()
    co = [(v.co.x, v.co.y) for v in me.vertices]
    tris = [[co[i] for i in lt.vertices] for lt in me.loop_triangles]
    obe.to_mesh_clear()
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(cu)
    xs = [p[0] for t in tris for p in t]
    ys = [p[1] for t in tris for p in t]
    _TEXT[key] = (tris, (min(xs), max(xs), min(ys), max(ys)))
    return _TEXT[key]


class Painter:
    def __init__(self, cv, I):
        self.cv, self.I = cv, I
        self.k = I.ppm * SS
        self.ox = (I.x + PAD) * SS - I.amin * self.k
        self.oy = (I.y + PAD) * SS + I.bmax * self.k
        cv.clip = (I.x * SS, I.y * SS, (I.x + I.w) * SS, (I.y + I.h) * SS)
        self.nat = NAT.get(I.view, lambda h, v: (h, v))

    def px(self, a, b):
        return (self.ox + a * self.k, self.oy - b * self.k)

    def P(self, h, v):
        return self.px(*self.nat(h, v))

    def fill(self, col):
        x0, y0, x1, y1 = self.cv.clip
        self.cv.col[y0:y1, x0:x1] = np.asarray(col, np.float32)
        self.cv.emi[y0:y1, x0:x1] = 0.0

    def poly(self, pts, col, alpha=1.0, emit=None):
        self.cv.poly([self.P(h, v) for (h, v) in pts], col, alpha, emit)

    def poly_ab(self, pts, col, alpha=1.0, emit=None):
        self.cv.poly([self.px(a, b) for (a, b) in pts], col, alpha, emit)

    def rect(self, h0, v0, h1, v1, col, alpha=1.0, emit=None):
        self.poly([(h0, v0), (h1, v0), (h1, v1), (h0, v1)], col, alpha, emit)

    def rrect(self, h0, v0, h1, v1, r, col, alpha=1.0, emit=None, n=5):
        pts = []
        for (cx, cy, a0) in ((h1 - r, v1 - r, 0), (h0 + r, v1 - r, 90), (h0 + r, v0 + r, 180), (h1 - r, v0 + r, 270)):
            for i in range(n + 1):
                t = math.radians(a0 + 90.0 * i / n)
                pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
        self.poly(pts, col, alpha, emit)

    def circle(self, h, v, r, col, alpha=1.0, emit=None, r_in=0.0):
        cx, cy = self.P(h, v)
        self.cv.circle(cx, cy, r * self.k, col, alpha, emit, r_in * self.k)

    def line(self, p0, p1, width, col, alpha=1.0, emit=None):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        ln = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / ln * width / 2, dx / ln * width / 2
        self.poly([(p0[0] + nx, p0[1] + ny), (p1[0] + nx, p1[1] + ny), (p1[0] - nx, p1[1] - ny), (p0[0] - nx, p0[1] - ny)], col, alpha, emit)

    def polyline(self, pts, width, col, closed=False, alpha=1.0):
        q = list(pts) + ([pts[0]] if closed else [])
        for a, b in zip(q, q[1:]):
            self.line(a, b, width, col, alpha)

    def text(self, s, h, v, cap, col, ang=0.0, bold=0.028, spacing=1.06, maxw=None, emit=None):
        tris, (x0, x1, y0, y1) = text_tris(s, bold, spacing)
        sc = cap / (y1 - y0)
        if maxw is not None and (x1 - x0) * sc > maxw:
            sc = maxw / (x1 - x0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a0, b0 = self.nat(h, v)
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        for t in tris:
            pp = []
            for (x, y) in t:
                x, y = (x - cx) * sc, (y - cy) * sc
                pp.append(self.px(a0 + x * ca - y * sa, b0 + x * sa + y * ca))
            self.cv.tri(pp[0], pp[1], pp[2], col, 1.0, emit)
        return (x1 - x0) * sc, (y1 - y0) * sc

    def star(self, h, v, R, col, ang=0.0, inner=None):
        a0, b0 = self.nat(h, v)
        w = 0.285 * R

        def rot(x, y, deg):
            c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
            return (a0 + x * c - y * s, b0 + x * s + y * c)

        for d in (0, 60, 120):
            self.poly_ab([rot(-w, -R, ang + d), rot(w, -R, ang + d), rot(w, R, ang + d), rot(-w, R, ang + d)], col)
        if inner is not None:
            sw = 0.035 * R
            self.poly_ab([rot(-sw, -0.62 * R, ang), rot(sw, -0.62 * R, ang), rot(sw, 0.62 * R, ang), rot(-sw, 0.62 * R, ang)], inner)
            self.cv.circle(*self.px(*rot(0, 0.66 * R, ang)), 0.07 * R * self.k, inner)
            pts = []
            for i in range(25):
                t = i / 24.0
                yy = lerp(0.50 * R, -0.50 * R, t)
                xx = 0.15 * R * math.sin(t * math.pi * 3.0) * (1.0 - 0.45 * t)
                pts.append(rot(xx, yy, ang))
            for p, q in zip(pts, pts[1:]):
                dx, dy = q[0] - p[0], q[1] - p[1]
                ln = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / ln * 0.024 * R, dx / ln * 0.024 * R
                self.poly_ab([(p[0] + nx, p[1] + ny), (q[0] + nx, q[1] + ny), (q[0] - nx, q[1] - ny), (p[0] - nx, p[1] - ny)], inner)
                self.cv.circle(*self.px(*q), 0.024 * R * self.k, inner)

    # pixel-space helpers for dirt passes ------------------------------------
    def grid(self):
        x0, y0, x1, y1 = self.cv.clip
        A = ((np.arange(x0, x1, dtype=np.float32) + 0.5) - self.ox) / self.k
        B = (self.oy - (np.arange(y0, y1, dtype=np.float32) + 0.5)) / self.k
        return (x0, y0, x1, y1), A[None, :], B[:, None]

    def blend(self, alpha, col):
        x0, y0, x1, y1 = self.cv.clip
        sub = self.cv.col[y0:y1, x0:x1]
        al = np.clip(alpha, 0.0, 1.0)[:, :, None]
        sub[:] = sub * (1.0 - al) + np.asarray(col, np.float32)[None, None, :] * al


def fractal_noise(n, seed, octs=((192, 1.0), (64, 0.6), (20, 0.35), (6, 0.2))):
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n), np.float32)
    tot = 0.0
    for cell, amp in octs:
        g = rng.random((n // cell + 3, n // cell + 3)).astype(np.float32)
        idx = np.arange(n, dtype=np.float32) / cell
        i0 = idx.astype(np.int32)
        f = idx - i0
        f = f * f * (3 - 2 * f)
        rows = g[i0] * (1 - f)[:, None] + g[i0 + 1] * f[:, None]
        out += amp * (rows[:, i0] * (1 - f)[None, :] + rows[:, i0 + 1] * f[None, :])
        tot += amp
    out /= tot
    return out


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# --- per chart paint functions ---------------------------------------------
W_, Y_, R_, K_, D_ = C['white'], C['yellow'], C['red'], C['black'], C['dark']
BAND0, BAND1 = 2.40, 2.645
CHK = [(0.86, 1.18), (1.18, 1.50)]
N_BLK = 5
TXT_Z = 1.95


def blue_light(P, h0, v0, h1, v1):
    if LIVERY == "plain":
        return
    P.rrect(h0 - 0.02, v0 - 0.02, h1 + 0.02, v1 + 0.02, 0.02, D_)
    P.rrect(h0, v0, h1, v1, 0.015, C['blue'], emit=E_BLUE)


def paint_box_side(P, left):
    ya, yb = BY0 - 0.3, BY1 + 0.3
    P.rect(ya, BAND0, yb, BAND1, Y_)
    P.rect(ya, BAND1, yb, 3.0, C['alu'])
    P.rect(ya, 0.2, yb, 0.80, Y_)
    P.rect(ya, 0.80, yb, 0.86, K_)
    bw = (BY1 - BY0) / N_BLK
    for r, (z0, z1) in enumerate(CHK):
        for c in range(-1, N_BLK + 1):
            col = R_ if (r + c) % 2 == 0 else Y_
            P.rect(BY0 + c * bw, z0, BY0 + (c + 1) * bw, z1, col)
    # white ring around the wheel arch (as on the reference)
    P.circle(RA_Y, RA_Z, 0.585, W_, r_in=0.30)
    P.rect(RA_Y - 0.62, 0.2, RA_Y + 0.62, 0.405, Y_)
    # aluminium corner posts + bottom rail
    for y in (BY0, BY1 - 0.05):
        P.rect(y - 0.3 if y == BY0 else y, 0.2, y + 0.05 if y == BY0 else y + 0.3, 3.0, C['alu'])
    P.rect(ya, 0.2, yb, 0.425, C['alu'])
    if left:
        # two-leaf side door on the rear half (reference A)
        d0, d1, dz0, dz1 = -3.22, -1.74, 0.86, 2.37
        dm = (d0 + d1) / 2
        for y in (d0, dm, d1):
            P.line((y, dz0), (y, dz1), 0.016, (0.42, 0.43, 0.43))
        P.line((d0, dz1), (d1, dz1), 0.016, (0.42, 0.43, 0.43))
        for y in (dm - 0.09, d1 - 0.09):
            P.rrect(y - 0.03, 1.56, y + 0.03, 1.76, 0.012, K_)
        ty, sy, sz, sr = -0.76, -2.86, 2.03, 0.25
    else:
        ty, sy, sz, sr = -0.95, -2.78, 1.98, 0.30
    P.text("AMBULANCE", ty, TXT_Z, 0.37, R_, bold=0.042, spacing=1.09, maxw=2.50)
    P.star(sy, sz, sr, R_, inner=W_)
    # lights on the top band: blue near both ends, white scene light in the middle
    zl0, zl1 = 2.47, 2.585
    blue_light(P, BY1 - 0.42, zl0, BY1 - 0.16, zl1)
    blue_light(P, BY0 + 0.16, zl0, BY0 + 0.42, zl1)
    P.rrect(-1.56, zl0 + 0.01, -1.17, zl1 - 0.01, 0.015, D_)
    P.rrect(-1.54, zl0 + 0.025, -1.19, zl1 - 0.025, 0.01, C['lamp'])
    # small amber side markers on the skirt
    for y in (BY0 + 0.25, -0.6, BY1 - 0.25):
        P.rect(y - 0.05, 0.52, y + 0.05, 0.56, C['amber'])


def paint_box_roof(P):
    P.fill(C['roofwhite'])
    P.star(0.0, -2.08, 0.74, R_, ang=0.0, inner=W_)          # top view: a=+Y, main bar along the vehicle axis
    # "112": custom 1 glyphs (the font 1 loses its flag when emboldened) + font 2; reads with the front up
    ch, cy = 0.74, -0.72
    for cx in (-0.66, -0.20):
        g = [(0.05, 0.0), (0.24, 0.0), (0.24, ch), (0.085, ch), (-0.13, ch - 0.19), (-0.13, ch - 0.37), (0.05, ch - 0.25)]
        P.poly([(cx + gx, cy - ch / 2 + gy) for gx, gy in g], R_)
    P.text("2", 0.44, cy, ch, R_, ang=-90.0, bold=0.05)
    # thin aluminium perimeter trim and a centre seam
    for (h0, v0, h1, v1) in ((-BX, BY0, BX, BY0 + 0.05), (-BX, BY1 - 0.05, BX, BY1), (-BX, BY0, -BX + 0.05, BY1), (BX - 0.05, BY0, BX, BY1)):
        P.rect(h0, v0, h1, v1, C['alu'])


def paint_box_rear(P):
    P.rect(-1.2, BAND0, 1.2, BAND1, Y_)
    P.rect(-1.2, BAND1, 1.2, 3.0, C['alu'])
    for x in (-BX - 0.2, BX - 0.05):
        P.rect(x, 0.2, x + 0.25, 3.0, C['alu'])
    # yellow strips beside the doors at check height
    P.rect(-BX + 0.05, 0.62, -0.86, 1.50, Y_)
    P.rect(0.86, 0.62, BX - 0.05, 1.50, Y_)
    for i in range(4):
        z = 0.62 + i * 0.22
        P.poly([(-BX + 0.05, z), (-0.86, z + 0.11), (-0.86, z + 0.22), (-BX + 0.05, z + 0.11)], R_)
        P.poly([(BX - 0.05, z), (0.86, z + 0.11), (0.86, z + 0.22), (BX - 0.05, z + 0.11)], R_)
    # doors
    seam = (0.42, 0.43, 0.43)
    P.polyline([(-0.84, 0.60), (-0.84, 2.34), (0.84, 2.34), (0.84, 0.60), (-0.84, 0.60)], 0.018, seam)
    P.line((0.0, 0.60), (0.0, 2.34), 0.018, seam)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 0.16, sx * 0.68))
        P.rrect(x0 - 0.025, 1.50, x1 + 0.025, 2.21, 0.07, K_)
        P.rrect(x0, 1.525, x1, 2.185, 0.05, C['glass'])
        P.poly([(x0 + 0.05, 1.95), (x1 - 0.05, 2.12), (x1 - 0.05, 2.17), (x0 + 0.05, 2.06)], C['glass_hi'], alpha=0.5)
        for z in (0.80, 1.45, 2.15):
            P.rect(sx * 0.80 - 0.035, z - 0.05, sx * 0.80 + 0.035, z + 0.05, C['alu'])
    P.rect(-0.80, 0.94, 0.80, 1.22, Y_)
    P.text("AMBULANCE", 0.0, 1.08, 0.165, R_, bold=0.042, spacing=1.09, maxw=1.48)
    P.rrect(-0.11, 1.27, 0.11, 1.40, 0.015, K_)
    # high-level lights on the band
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 0.36, sx * 0.62))
        P.rrect(x0 - 0.015, 2.455, x1 + 0.015, 2.60, 0.02, D_)
        P.rrect(x0, 2.47, x1, 2.585, 0.015, C['tail'], emit=E_RED)
    P.rrect(-0.20, 2.47, 0.20, 2.585, 0.015, D_)
    P.rrect(-0.18, 2.485, 0.18, 2.57, 0.01, C['lamp'])
    # lower valance + number plate
    P.rect(-1.2, 0.2, 1.2, 0.60, D_)
    P.rect(-0.27, 0.455, 0.27, 0.575, C['plate'])
    P.text("RTW 112", 0.0, 0.515, 0.065, K_, bold=0.02)


def paint_box_front(P):
    P.rect(-1.2, BAND0, 1.2, BAND1, Y_)
    P.rect(-1.2, BAND1, 1.2, 3.0, C['alu'])
    for x in (-BX - 0.2, BX - 0.05):
        P.rect(x, 0.2, x + 0.25, 3.0, C['alu'])
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 0.66, sx * 0.92))
        blue_light(P, x0, 2.47, x1, 2.585)
    P.rect(-0.92, 0.2, 0.92, 2.30, (0.30, 0.30, 0.30))   # behind the cab (hidden)


def cab_glass_poly():
    zt = lambda y: tab(ZR, y) - 0.235
    pts = [(1.04, BELT + 0.035), (2.555, BELT + 0.03)]
    pts += [(2.05, zt(2.06) + 0.0), (1.75, zt(1.75)), (1.45, zt(1.45)), (1.04, zt(1.04))]
    return pts


def paint_cab_side(P, left):
    # yellow front wing + bonnet side; everything above the belt behind the cowl stays white
    P.rect(2.02, 0.60, 3.7, 1.80, Y_)
    P.poly([(2.02, BELT - 0.01), (YC, BELT - 0.012), (YC, 1.70), (2.02, 2.8)], W_)
    P.rect(1.9, BELT - 0.01, 2.02, 2.8, W_)
    P.rect(2.02, 0.2, 3.7, 0.60, W_)
    # door: battenburg below the belt, black sill stripe
    bw = (2.02 - 0.55) / 3
    rows = [(0.78, 1.09), (1.09, BELT - 0.005)]
    for r, (z0, z1) in enumerate(rows):
        for c in range(-1, 3):
            P.rect(0.55 + c * bw, z0, 0.55 + (c + 1) * bw, z1, R_ if (r + c) % 2 == 1 else Y_)
    P.rect(0.2, 0.60, 2.02, 0.78, K_)
    # red wedge behind the headlight
    P.poly([(2.985, 0.60), (3.42, 0.60), (3.38, 0.84), (3.10, 0.97), (2.985, 0.97)], R_)
    # headlight wrap-around + indicator
    P.poly([(3.30, 0.85), (3.50, 0.81), (3.50, 0.99), (3.36, 1.01), (3.30, 0.975)], K_)
    P.poly([(3.325, 0.87), (3.50, 0.835), (3.50, 0.97), (3.37, 0.99), (3.325, 0.96)], C['lamp'])
    P.rect(2.82, 1.14, 2.90, 1.175, C['amber'])
    # door seams, handle recess
    seam = (0.40, 0.41, 0.41)
    P.line((0.995, 0.34), (0.995, 2.22), 0.014, seam)
    P.line((2.02, 0.34), (2.02, BELT), 0.014, seam)
    P.line((0.995, 0.34), (2.02, 0.34), 0.012, seam)
    P.rrect(1.06, 1.20, 1.27, 1.27, 0.02, (0.20, 0.20, 0.20))
    # side glass with rubber frame
    g = cab_glass_poly()
    P.polyline(g, 0.07, K_, closed=True)
    for p in g:
        P.circle(p[0], p[1], 0.035, K_)
    P.poly(g, C['glass'])
    P.poly([(1.20, 1.86), (1.85, 2.08), (1.62, 2.14), (1.20, 2.10)], C['glass_hi'], alpha=0.35)
    P.line((1.13, BELT + 0.035), (1.13, tab(ZR, 1.13) - 0.235), 0.03, K_)      # quarter light bar


def paint_cab_top(P):
    P.fill(W_)
    P.rect(-1.2, 2.40, 1.2, 3.7, Y_)
    P.rect(-1.2, 2.60, 1.2, 2.705, K_)                 # cowl / wiper tray
    for sx in (-1, 1):
        P.poly([(sx * 0.28, 2.80), (sx * 0.56, 2.80), (sx * 0.52, 2.93), (sx * 0.28, 2.93)], D_)


def paint_cab_front(P):
    P.fill(Y_)
    P.rect(-1.2, BELT - 0.03, 1.2, 3.0, W_)             # A-pillar faces that fall into the front chart
    P.rect(-1.2, 0.2, 1.2, 0.635, K_)
    P.rrect(-0.44, 0.65, 0.44, 0.965, 0.05, C['silver'])
    P.rrect(-0.415, 0.67, 0.415, 0.945, 0.04, K_)
    for z in (0.735, 0.80, 0.865):
        P.rect(-0.39, z - 0.009, 0.39, z + 0.009, (0.22, 0.22, 0.23))
    for sx in (-1, 1):
        o = [(sx * 0.47, 0.79), (sx * 0.80, 0.81), (sx * 0.80, 0.995), (sx * 0.66, 1.0), (sx * 0.47, 0.94)]
        P.poly(o, K_)
        i = [(sx * 0.49, 0.81), (sx * 0.80, 0.83), (sx * 0.80, 0.975), (sx * 0.66, 0.98), (sx * 0.49, 0.925)]
        P.poly(i, C['lamp'])
        P.circle(sx * 0.59, 0.885, 0.048, (0.60, 0.63, 0.66))
        xa, xb = sorted((sx * 0.73, sx * 0.80))
        P.rect(xa, 0.84, xb, 0.96, C['amber'])
        x0, x1 = sorted((sx * 0.19, sx * 0.34))
        P.rrect(x0, 0.77, x1, 0.835, 0.012, C['blue'], emit=E_BLUE)
    P.rect(-0.075, 0.775, 0.075, 0.83, C['silver'])       # plain badge bar, no brand


def paint_windscreen(P):
    I = P.I
    n = 8
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        col = tuple(lerp(C['glass'][k] * 0.9, C['glass_hi'][k], (t0 + t1) / 2) for k in range(3))
        P.poly_ab([(I.amin - 1, lerp(I.bmin - 0.05, I.bmax + 0.05, t0)), (I.amax + 1, lerp(I.bmin - 0.05, I.bmax + 0.05, t0)),
                   (I.amax + 1, lerp(I.bmin - 0.05, I.bmax + 0.05, t1)), (I.amin - 1, lerp(I.bmin - 0.05, I.bmax + 0.05, t1))], col)
    P.poly_ab([(I.amin - 1, I.bmax - 0.07), (I.amax + 1, I.bmax - 0.07), (I.amax + 1, I.bmax + 1), (I.amin - 1, I.bmax + 1)], K_)
    P.poly_ab([(I.amin - 1, I.bmin - 1), (I.amax + 1, I.bmin - 1), (I.amax + 1, I.bmin + 0.05), (I.amin - 1, I.bmin + 0.05)], K_)
    for a0 in (0.58, -0.10):
        P.poly_ab([(a0, I.bmin + 0.05), (a0, I.bmin + 0.075), (a0 - 0.52, I.bmin + 0.165), (a0 - 0.52, I.bmin + 0.14)], K_)


def paint_wheel(P, yc):
    zc = WHEEL_R
    P.circle(yc, zc, 0.232, D_, r_in=0.212)
    P.circle(yc, zc, 0.205, C['rim'])
    P.circle(yc, zc, 0.20, (0.50, 0.51, 0.52), r_in=0.185)
    for i in range(8):
        t = 2 * math.pi * (i + 0.5) / 8
        P.circle(yc + 0.148 * math.cos(t), zc + 0.148 * math.sin(t), 0.026, (0.10, 0.10, 0.105))
    P.circle(yc, zc, 0.085, C['hub'])
    P.circle(yc, zc, 0.045, (0.20, 0.20, 0.21))
    for i in range(5):
        t = 2 * math.pi * i / 5 + 0.3
        P.circle(yc + 0.066 * math.cos(t), zc + 0.066 * math.sin(t), 0.009, (0.62, 0.63, 0.64))
    # tyre lettering ring hint
    P.circle(yc, zc, 0.30, (0.11, 0.11, 0.115), r_in=0.292)


def paint_ac_top(P):
    P.rrect(-0.40, 0.06, 0.40, 0.42, 0.03, (0.62, 0.63, 0.64))
    for i in range(7):
        y = 0.095 + i * 0.048
        P.rect(-0.37, y, -0.02, y + 0.022, (0.20, 0.20, 0.21))
    P.circle(0.20, 0.24, 0.14, (0.20, 0.20, 0.21))
    P.circle(0.20, 0.24, 0.12, (0.38, 0.39, 0.40))
    for i in range(6):
        t = math.pi * i / 6
        P.line((0.20 - 0.12 * math.cos(t), 0.24 - 0.12 * math.sin(t)), (0.20 + 0.12 * math.cos(t), 0.24 + 0.12 * math.sin(t)), 0.012, (0.20, 0.20, 0.21))
    P.circle(0.20, 0.24, 0.035, (0.20, 0.20, 0.21))


def paint_vent_top(P):
    P.rrect(-0.21, -3.20, 0.21, -3.04, 0.02, (0.25, 0.25, 0.26))
    for i in range(4):
        y = -3.185 + i * 0.037
        P.rect(-0.19, y, 0.19, y + 0.016, (0.50, 0.51, 0.52))


def paint_bumper_front(P):
    P.rect(-0.27, 0.36, 0.27, 0.475, C['plate'])
    P.text("RTW 112", 0.0, 0.4175, 0.062, K_, bold=0.02)
    P.rrect(-0.55, 0.275, 0.55, 0.33, 0.015, (0.02, 0.02, 0.02))
    for sx in (-1, 1):
        P.circle(sx * 0.63, 0.40, 0.05, (0.30, 0.31, 0.32))
        P.circle(sx * 0.63, 0.40, 0.036, (0.62, 0.64, 0.66))


def paint_tail(P):
    I = P.I
    P.rect(-1.1, 0.47, 1.1, 0.67, C['tail'], emit=E_RED)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * 0.74, sx * 0.81))
        P.rect(x0, 0.47, x1, 0.67, C['amber'])
        x0, x1 = sorted((sx * 0.81, sx * 0.86))
        P.rect(x0, 0.47, x1, 0.67, C['lamp'])


def paint_step_top(P):
    P.fill((0.13, 0.13, 0.135))
    P.rect(-0.94, -3.50, 0.94, -3.47, Y_)
    for i in range(24):
        x = -0.90 + i * 0.078
        P.line((x, -3.455), (x + 0.04, -3.345), 0.012, (0.22, 0.22, 0.23))


def dirt_side(P, noise, k=0.42, ztop=1.15):
    (x0, y0, x1, y1), A, B = P.grid()
    nz = noise[y0:y1, x0:x1]
    al = k * (1.0 - sstep(0.30, ztop, B)) ** 1.6 * (0.45 + 0.75 * nz)
    al = al + 0.10 * sstep(0.45, 0.8, nz)
    P.blend(al, C['dirt'])


def dirt_flat(P, noise, k=0.16):
    (x0, y0, x1, y1), A, B = P.grid()
    nz = noise[y0:y1, x0:x1]
    P.blend(k * sstep(0.30, 0.85, nz) + 0.025, C['dirt'])


def paint_all(isl):
    n = TEX * SS
    cv = Canvas(n)
    noise = fractal_noise(n, 7)
    noise2 = fractal_noise(n, 11, octs=((150, 0.7), (48, 1.0), (14, 0.8), (5, 0.5)))
    for I in isl.values():
        P = Painter(cv, I)
        P.fill(C[I.bg])
        if EMIT.get(I.bg):
            x0, y0, x1, y1 = cv.clip
            cv.emi[y0:y1, x0:x1] = np.asarray(EMIT[I.bg], np.float32)
        multi = len(set(m for _, _, m in I.faces)) > 1
        if multi:
            for fi, cs, m in I.faces:
                if m != I.bg:
                    cv.poly([P.px(a, b) for (a, b) in cs], C[m], 1.0, EMIT.get(m))
        key = I.key
        if key == ('box', '-X'):
            paint_box_side(P, True); dirt_side(P, noise)
        elif key == ('box', '+X'):
            paint_box_side(P, False); dirt_side(P, noise)
        elif key == ('box', '+Z'):
            paint_box_roof(P); dirt_flat(P, noise2, 0.11)
        elif key == ('box', '-Y'):
            paint_box_rear(P); dirt_side(P, noise, 0.50, 1.3)
        elif key == ('box', '+Y'):
            paint_box_front(P)
        elif key == ('cab', '-X'):
            paint_cab_side(P, True); dirt_side(P, noise, 0.45, 1.0)
        elif key == ('cab', '+X'):
            paint_cab_side(P, False); dirt_side(P, noise, 0.45, 1.0)
        elif key == ('cab', '+Z'):
            paint_cab_top(P); dirt_flat(P, noise2, 0.10)
        elif key == ('cab', '+Y'):
            paint_cab_front(P); dirt_side(P, noise, 0.35, 0.9)
        elif key == 'cab_ws':
            paint_windscreen(P); dirt_flat(P, noise2, 0.10)
        elif isinstance(key, str) and key.endswith('_out') and key[0] == 'w':
            paint_wheel(P, FA_Y if key[1] == 'F' else RA_Y)
            dirt_flat(P, noise2, 0.35)
        elif isinstance(key, str) and key.endswith('_tr'):
            (x0, y0, x1, y1), A, B = P.grid()
            stripes = (np.mod(A * 14.0, 1.0) < 0.5).astype(np.float32) * np.ones_like(B)
            P.blend(0.35 * stripes, (0.12, 0.12, 0.125))
            dirt_flat(P, noise2, 0.35)
        elif key == ('ac', '+Z'):
            paint_ac_top(P); dirt_flat(P, noise2, 0.15)
        elif key == ('vent', '+Z'):
            paint_vent_top(P)
        elif key == ('bumpF', '+Y'):
            paint_bumper_front(P); dirt_side(P, noise, 0.30, 0.7)
        elif key in (('tailL', '-Y'), ('tailR', '-Y')):
            paint_tail(P)
        elif key == ('stepR', '+Z'):
            paint_step_top(P)
        elif isinstance(key, tuple) and key[0].startswith('cl') and key[1] != '-Z':
            # beacon: darker base line
            if key[1] in ('+X', '-X', '+Y', '-Y'):
                h0 = -9
                P.rect(-9, 2.60, 9, 2.665, D_)
    cv.clip = (0, 0, n, n)
    col = cv.col.reshape(TEX, SS, TEX, SS, 3).mean(axis=(1, 3))
    emi = cv.emi.reshape(TEX, SS, TEX, SS, 3).mean(axis=(1, 3))
    return col, emi


def save_png(arr, path, name, srgb=True):
    h, w, _ = arr.shape
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, w, h, alpha=False)
    rgba = np.ones((h, w, 4), np.float32)
    rgba[:, :, :3] = np.clip(arr, 0, 1)
    img.pixels.foreach_set(rgba[::-1].ravel())
    img.filepath_raw = path
    img.file_format = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    img = bpy.data.images.load(path)
    img.name = name
    return img


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def enable_gpu(scene):
    scene.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'OPTIX'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == 'OPTIX')
        scene.cycles.device = 'GPU'
    except Exception as e:
        print("GPU setup failed, CPU fallback:", e)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    b = build_all()
    bm, tags = make_bmesh(b)
    tag_layer = bm.faces.layers.int.new("tag")     # survives triangulation: which part each face is
    for i, f in enumerate(bm.faces):
        f[tag_layer] = i
    isl = make_islands(bm, tags)
    pack_islands(isl)
    write_uvs(bm, isl)
    # shading: smooth with sharp edges over 40 deg
    for f in bm.faces:
        f.smooth = True
    nsharp = 0
    for e in bm.edges:
        if len(e.link_faces) == 2 and e.calc_face_angle() > math.radians(40.0):
            e.smooth = False
            nsharp += 1
    ngons = sum(1 for f in bm.faces if len(f.verts) > 4)
    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method='BEAUTY', ngon_method='BEAUTY')
    me = bpy.data.meshes.new("Ambulance")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("Ambulance", me)
    scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    bpy.context.view_layer.update()
    print("mesh: %d verts, %d tris (ngons before triangulation: %d, sharp edges: %d)" % (len(me.vertices), len(me.polygons), ngons, nsharp))
    # weighted normals so the big flat panels stay flat next to the rounded edges
    try:
        mod = ob.modifiers.new("WN", 'WEIGHTED_NORMAL')
        mod.keep_sharp = True
        mod.weight = 60
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print("weighted normal failed:", e)

    print("painting ... (%.1fs)" % (time.time() - T0))
    col, emi = paint_all(isl)
    print("painted (%.1fs)" % (time.time() - T0))
    alb_path = os.path.join(OUT, "ambulance_albedo.png")
    emi_path = os.path.join(OUT, "ambulance_emission.png")

    # material
    mat = bpy.data.materials.new("M_Ambulance")
    try:
        mat.use_nodes = True
    except Exception:
        pass
    nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
    me.materials.append(mat)

    ao = None
    if DO_AO:
        enable_gpu(scene)
        scene.cycles.samples = 192
        scene.cycles.use_denoising = False
        if scene.world is None:
            scene.world = bpy.data.worlds.new("W")
        scene.world.light_settings.distance = 1.2
        # ground plane for contact occlusion
        gm = bpy.data.meshes.new("tmp_ground")
        gm.from_pydata([(-30, -30, 0), (30, -30, 0), (30, 30, 0), (-30, 30, 0)], [], [(0, 1, 2, 3)])
        gob = bpy.data.objects.new("tmp_ground", gm)
        scene.collection.objects.link(gob)
        aoimg = bpy.data.images.new("tmp_ao", TEX, TEX, alpha=False)
        node = nt.nodes.new('ShaderNodeTexImage')
        node.image = aoimg
        nt.nodes.active = node
        scene.render.bake.margin = 6
        try:
            scene.render.bake.margin_type = 'EXTEND'
        except Exception:
            pass
        bpy.ops.object.select_all(action='DESELECT')
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.bake(type='AO')
        buf = np.empty(TEX * TEX * 4, np.float32)
        aoimg.pixels.foreach_get(buf)
        ao = buf.reshape(TEX, TEX, 4)[::-1, :, 0].copy()
        nt.nodes.remove(node)
        bpy.data.images.remove(aoimg)
        bpy.data.objects.remove(gob)
        bpy.data.meshes.remove(gm)
        print("AO baked (%.1fs) mean %.3f" % (time.time() - T0, float(ao.mean())))
        aom = 1.0 - 0.62 * (1.0 - np.clip(ao, 0, 1)) ** 1.0
        col = col * aom[:, :, None]

    alb = save_png(col, alb_path, "ambulance_albedo")
    emg = save_png(emi, emi_path, "ambulance_emission")
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = alb
    tex.location = (-420, 300)
    tex2 = nt.nodes.new('ShaderNodeTexImage')
    tex2.image = emg
    tex2.location = (-420, -60)
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    nt.links.new(tex2.outputs['Color'], bsdf.inputs['Emission Color'])
    bsdf.inputs['Emission Strength'].default_value = 1.0
    bsdf.inputs['Roughness'].default_value = 0.55
    bsdf.inputs['Metallic'].default_value = 0.0
    export_kit(scene, ob, tags)
    blend = os.path.join(OUT, "ambulance.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    alb.filepath = "//ambulance_albedo.png"
    emg.filepath = "//ambulance_emission.png"
    bpy.ops.wm.save_mainfile()
    print("saved", blend, "(%.1fs)" % (time.time() - T0))



def export_kit(scene, ob, tags):
    """The kit's two GLBs: <name>_joined.glb (one mesh, transforms applied, what the rig renders and Pixel Match
    paints) and <name>_parts.glb (the body, and each wheel as its own object with its origin on its axle)."""
    def export(path, objs):
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True, export_yup=True, export_apply=True)
        print("KIT wrote %s (%.2f MB)" % (path, os.path.getsize(path) / 1e6))

    ob.name = NAME
    export(os.path.join(OUT, NAME + "_joined.glb"), [ob])
    attr = ob.data.attributes["tag"].data
    part_of = [tags[a.value].get("part", "") for a in attr]
    parts = []
    for name, keep in [("body", lambda p: p not in WHEELS)] + [(WHEELS[w], lambda p, w=w: p == w) for w in WHEELS]:
        dup = ob.copy()
        dup.data = ob.data.copy()
        dup.name = name
        scene.collection.objects.link(dup)
        dup.data.polygons.foreach_set("select", [not keep(p) for p in part_of])
        bpy.ops.object.select_all(action="DESELECT")
        dup.select_set(True)
        bpy.context.view_layer.objects.active = dup
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_mode(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")
        dup.data.polygons.foreach_set("select", [not keep(p) for p in part_of])
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.delete(type="FACE")
        bpy.ops.object.mode_set(mode="OBJECT")
        if name != "body":       # the origin goes to the wheel's centre, which is its axle
            co = np.array([v.co[:] for v in dup.data.vertices])
            pivot = Vector(((co.min(axis=0) + co.max(axis=0)) / 2).tolist())
            dup.data.transform(Matrix.Translation(-pivot))
            dup.location = pivot
        print("KIT part %-9s origin (%7.3f, %7.3f, %6.3f)  triangles %5d" % (
            name, dup.location.x, dup.location.y, dup.location.z, len(dup.data.polygons)))
        parts.append(dup)
    assert sum(len(o.data.polygons) for o in parts) == len(ob.data.polygons)
    export(os.path.join(OUT, NAME + "_parts.glb"), parts)
    for o in parts:
        bpy.data.objects.remove(o)
    ob.name = "Ambulance"


main()
