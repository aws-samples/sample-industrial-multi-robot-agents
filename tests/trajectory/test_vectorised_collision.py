"""The vectorised moving-obstacle check answers exactly what the loop answered.

``robot_kinematics._config_is_free`` checked every arm sphere against every other
arm's sphere in a Python double loop — 8.3 ms per state with 150 parked-arm
spheres on welding_cell_2_small, ~95 % of the compound search's validity check. It is now
one distance matrix (``_moving_hit``). Same rule: a state collides when any
``|p - o| < r_o + sphere_radius``; the diagnosis gets the same FIRST hit
(obstacle-major order) as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

np = pytest.importorskip("numpy")


@pytest.fixture(params=["numpy", "numba"])
def rk(request, monkeypatch):
    """Every equivalence test runs against BOTH implementations: the NumPy
    reference and the compiled kernels (skipped when numba is not installed)."""
    mod = pytest.importorskip("agents.trajectory.tools.robot_kinematics")
    from agents.trajectory.tools import fastgeom
    if request.param == "numba":
        if not fastgeom.AVAILABLE:
            pytest.skip(fastgeom.status())
    else:
        monkeypatch.setattr(fastgeom, "AVAILABLE", False)
    return mod


def _loop_first_hit(points, obstacles, r):
    """The old code, verbatim in behaviour: obstacle-major, first hit wins."""
    for k, (o, ro) in enumerate(obstacles):
        for i, p in enumerate(points):
            gap = float(np.linalg.norm(p - o)) - (ro + r)
            if gap < 0:
                return k, i, gap
    return None


def _case(rng, n_pts, n_obs, spread):
    pts = [rng.uniform(-spread, spread, 3) for _ in range(n_pts)]
    obs = [(rng.uniform(-spread, spread, 3), float(rng.uniform(0.05, 0.4)))
           for _ in range(n_obs)]
    return pts, obs


def test_same_answer_and_same_first_hit_on_random_states(rk):
    rng = np.random.default_rng(7)
    hits = 0
    for trial in range(3000):
        pts, obs = _case(rng, int(rng.integers(1, 60)), int(rng.integers(1, 160)),
                         spread=float(rng.choice([0.5, 2.0, 6.0])))
        want = _loop_first_hit(pts, obs, 0.12)
        got = rk._moving_hit(pts, list(obs), 0.12)
        assert (want is None) == (got is None), trial
        if want is not None:
            hits += 1
            assert got[:2] == want[:2], trial
            assert got[2] == pytest.approx(want[2], abs=1e-12)
    assert 300 < hits < 2700, "the cases must exercise both answers"


def test_boundary_is_strict_like_the_loop(rk):
    o = np.array([0.0, 0.0, 0.0])
    p = np.array([0.5, 0.0, 0.0])
    assert rk._moving_hit([p], [(o, 0.38)], 0.12) is None          # exactly touching
    assert rk._moving_hit([p], [(o, 0.3800001)], 0.12) is not None


def test_empty_inputs_are_clear(rk):
    assert rk._moving_hit([], [(np.zeros(3), 1.0)], 0.12) is None
    assert rk._moving_hit([np.zeros(3)], [], 0.12) is None


def test_the_obstacle_array_is_reused_for_one_list_and_rebuilt_for_another(rk):
    obs = [(np.zeros(3), 0.1), (np.ones(3), 0.2)]
    a = rk._obstacle_arrays(obs)
    assert rk._obstacle_arrays(obs)[0] is a[0], "same list: built once"
    obs.append((np.full(3, 2.0), 0.3))
    assert len(rk._obstacle_arrays(obs)[0]) == 3, "a list that grew is rebuilt"
    other = list(obs)
    assert rk._obstacle_arrays(other)[0] is not a[0]


# ── frame origins in world coordinates, all at once ──────────────────────────

class _SE3:
    def __init__(self, t):
        self.translation = np.asarray(t, float)


class _Data:
    def __init__(self, ts):
        self.oMf = [_SE3(t) for t in ts]


def test_all_frames_at_once_equal_one_frame_at_a_time(rk):
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(1, 40))
        data = _Data(rng.uniform(-3, 3, (n, 3)))
        base, yaw = rng.uniform(-2, 2, 3), float(rng.uniform(-np.pi, np.pi))
        got = rk._frames_world(data, n, base, yaw)
        want = [rk._frame_world(data, i, base, yaw) for i in range(n)]
        assert len(got) == n
        assert np.allclose(np.asarray(got), np.asarray(want), atol=1e-12)
    assert rk._frames_world(_Data([]), 0, np.zeros(3), 0.0) == []


# ── the first breached box: same box, same sphere, same clearance ─────────────

def _brute_first_box(P, R, own, comp, extra):
    best = None
    for m in range(len(comp["box_c"])):
        for i in range(len(P)):
            c, h = comp["box_c"][m], comp["box_h"][m]
            cl = np.clip(P[i] - c, -h, h)
            d = float(np.linalg.norm(P[i] - c - cl))
            marg = comp["box_m"][m] if np.isnan(own[i]) else own[i]
            need = R[i] + marg + extra
            if d < need:
                return m, i, d - need
    return best


def test_the_box_check_finds_the_brute_force_answer(rk):
    rng = np.random.default_rng(11)
    hits = 0
    for trial in range(400):
        n, k = int(rng.integers(1, 30)), int(rng.integers(1, 60))
        P = rng.uniform(-2, 2, (n, 3))
        R = rng.uniform(0.02, 0.15, n)
        own = np.where(rng.random(n) < 0.3, rng.uniform(0.0, 0.05, n), np.nan)
        comp = {"box_c": rng.uniform(-2, 2, (k, 3)), "box_h": rng.uniform(0.01, 0.4, (k, 3)),
                "box_m": rng.uniform(0.0, 0.05, k)}
        extra = float(rng.choice([0.0, 0.03]))

        def _m(obstacle_margins, _own=own):
            return np.where(np.isnan(_own)[:, None], obstacle_margins[None, :], _own[:, None])

        got = rk._first_box_hit(P, R, own, comp, extra, _m)
        want = _brute_first_box(P, R, own, comp, extra)
        assert (got is None) == (want is None), trial
        if want is not None:
            hits += 1
            assert (got[0], got[1]) == (want[0], want[1]), trial
            assert got[3] == pytest.approx(want[2], abs=1e-9)
    assert 40 < hits < 360


def test_numba_can_be_switched_off_completely(monkeypatch):
    """NUMBA_GEOM=0 at import time is a full rollback to the NumPy path."""
    import importlib
    from agents.trajectory.tools import fastgeom
    monkeypatch.setenv("NUMBA_GEOM", "0")
    off = importlib.reload(fastgeom)
    try:
        assert off.AVAILABLE is False and "NUMBA_GEOM=0" in off.status()
    finally:
        monkeypatch.delenv("NUMBA_GEOM")
        importlib.reload(fastgeom)


# ── the robot-vs-robot minimum gap, and the link-fill spheres ────────────────

def test_the_pair_check_gives_the_exact_minimum_gap(rk):
    rng = np.random.default_rng(5)
    for trial in range(300):
        na, nb = int(rng.integers(1, 80)), int(rng.integers(1, 80))
        A = rng.uniform(-1, 1, (na, 3)) + rng.uniform(-2, 2, 3)
        B = rng.uniform(-1, 1, (nb, 3))
        stop1, stop2 = rng.uniform(0, 0.05, na), rng.uniform(0, 0.05, nb)
        radii1, radii2 = rng.uniform(0.05, 0.15, na), rng.uniform(0.05, 0.15, nb)

        def groups(n):
            cuts = sorted(set(rng.integers(1, n, size=min(4, max(0, n - 1))).tolist())) if n > 1 else []
            edges = [0, *cuts, n]
            return [slice(a, b) for a, b in zip(edges, edges[1:]) if b > a]

        g1, g2 = (groups(na), groups(nb)) if trial % 2 else (None, None)
        hit, gap = rk._check_robot_pair_collision(list(A), list(B), 0.12, stop1, stop2,
                                                  g1, g2, radii1, radii2)
        brute = (np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
                 - radii1[:, None] - radii2[None] - stop1[:, None] - stop2[None]).min()
        assert gap == pytest.approx(brute, abs=1e-9), trial
        assert hit == (brute < 0)


class _JData:
    def __init__(self, ts):
        self.oMi = [_SE3(t) for t in ts]


class _Model:
    def __init__(self, nj):
        self.njoints = nj


def test_the_link_fill_points_are_the_same_points(rk):
    rng = np.random.default_rng(9)
    for _ in range(200):
        nj = int(rng.integers(2, 10))
        data = _JData(rng.uniform(-1.5, 1.5, (nj, 3)))
        base, yaw = rng.uniform(-2, 2, 3), float(rng.uniform(-np.pi, np.pi))
        step = float(rng.choice([0.05, 0.1, 0.17]))
        got = rk._link_fill_positions(_Model(nj), data, base, yaw, step)
        cy, sy = np.cos(yaw), np.sin(yaw)
        W = [np.array([base[0] + p[0] * cy - p[1] * sy, base[1] + p[0] * sy + p[1] * cy,
                       base[2] + p[2]]) for p in (o.translation for o in data.oMi)]
        want = []
        for j in range(1, nj - 1):
            a, b = W[j], W[j + 1]
            seg = float(np.linalg.norm(b - a))
            n = int(seg / step)
            if n < 1:
                continue
            d = (b - a) / seg
            for k in range(1, n + 1):
                if k * step >= seg:
                    break
                want.append(a + d * k * step)
        assert len(got) == len(want)
        if want:
            assert np.allclose(np.asarray(got), np.asarray(want), atol=1e-12)


# ── the coordinator's conflict test without the full distance matrix ─────────

def test_cloud_conflict_matches_the_full_matrix(rk):
    from agents.trajectory.tools.coordination import _clouds_within
    rng = np.random.default_rng(21)
    yes = 0
    for trial in range(2000):
        pa = rng.uniform(-0.6, 0.6, (int(rng.integers(1, 300)), 3))
        pb = rng.uniform(-0.6, 0.6, (int(rng.integers(1, 300)), 3)) \
            + rng.uniform(-2.5, 2.5, 3)
        band, radius = 0.2, 0.12
        d = np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(-1)).min()
        want = bool(d - 2 * radius <= band)
        assert _clouds_within(pa, pb, band + 2 * radius) == want, trial
        yes += want
    assert 200 < yes < 1800, "both answers must be exercised"


def test_cloud_conflict_counts_an_exact_touch_as_a_conflict(rk):
    from agents.trajectory.tools.coordination import _clouds_within
    a = np.zeros((1, 3))
    b = np.array([[0.5, 0.0, 0.0]])
    assert _clouds_within(a, b, 0.5) is True      # d == thr: "<=" as before
    assert _clouds_within(a, b, 0.4999999) is False
