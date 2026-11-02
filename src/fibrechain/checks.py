"""
Direct ray traces used as references and cross-checks.

uniform_rays      : the OpticStudio geometric vignetting number. Rays on a
                    square grid inside the unit pupil (px, py), i.e. uniform
                    over the object-NA disc; fraction reaching the detector and
                    losses grouped by the surface that stops them.
working_fnumber   : working F/# from the real marginal rays in image space,
                    tangential (py = +-1) and sagittal (px = +-1), apertures
                    ignored, as in the OpticStudio F/# Data report:
                    F = 1 / (2 n sin(alpha / 2)), alpha = angle between the two
                    marginal rays.
direct_hexagon    : the same tracer fed with rays drawn uniformly over the
                    lenslet hexagon in fibre-output angle and scrambled (each
                    sample rotated by the golden angle times its index). It is
                    the 'without K, FRD 0' fibre output, traced ray by ray; it
                    must reproduce the map-weighted budget (budget.py).
"""
import numpy as np

__all__ = ["uniform_rays", "hexagon_samples", "direct_trace", "working_fnumber"]


def _lost(b, group_of):
    lost = {}
    for sid in np.unique(b[b != -1]):
        g = group_of.get(int(sid), str(int(sid)))
        lost[g] = lost.get(g, 0) + np.mean(b == sid)
    return lost


def uniform_rays(system, pupil, fields, wavelengths, group_of, n=181):
    u = np.linspace(-1, 1, n)
    PX, PY = np.meshgrid(u, u)
    m = PX ** 2 + PY ** 2 <= 1
    rows = []
    for fx, fy in fields:
        for wi, w in enumerate(wavelengths):
            p0, d0 = pupil.pupil_rays(fx, fy, PX[m], PY[m])
            _, _, b = system.trace(p0, d0, wi, w)
            rows.append((fx, fy, float(w), float(np.mean(b == -1)), _lost(b, group_of)))
    return rows


def hexagon_samples(th_o, n=160):
    """Regular grid inside the hexagon (apothem th_o, flats normal to x), each
    point rotated about the axis by the golden angle times its index."""
    rc = 2 * th_o / np.sqrt(3)
    g = np.arange(-rc, rc, rc / n)
    GX, GY = np.meshgrid(g + 0.5 * (g[1] - g[0]), g + 0.5 * (g[1] - g[0]))
    m = np.ones_like(GX, bool)
    for k in range(6):
        t = k * np.pi / 3
        m &= GX * np.cos(t) + GY * np.sin(t) <= th_o
    th = np.hypot(GX[m], GY[m])
    ph = np.mod(np.arctan2(GY[m], GX[m]) + np.arange(m.sum()) * np.pi * (3 - np.sqrt(5)),
                2 * np.pi)
    return th, ph


def direct_trace(system, pupil, fields, wavelengths, group_of, theta, phi):
    rows = []
    for fx, fy in fields:
        p, d = pupil.fibre_rays(fx, fy, theta, phi)
        for wi, w in enumerate(wavelengths):
            _, _, b = system.trace(p, d, wi, w)
            rows.append((fx, fy, float(w), float(np.mean(b == -1)), _lost(b, group_of)))
    return rows


def working_fnumber(system, pupil, fx, fy, wl_index, wl_um):
    """(tangential, sagittal) working F/# in image space; nan if a marginal ray
    leaves the domain of a surface sag."""
    px = np.array([0.0, 0.0, 1.0, -1.0])
    py = np.array([1.0, -1.0, 0.0, 0.0])
    p, d = pupil.pupil_rays(fx, fy, px, py)
    _, dd, _ = system.trace(p, d, wl_index, wl_um)
    n = system.S[-999].get("n", np.ones(wl_index + 1))[wl_index]

    def f(a, b):
        return 1 / (2 * n * np.sin(np.arccos(np.clip(a @ b, -1, 1)) / 2))
    return f(dd[0], dd[1]), f(dd[2], dd[3])
