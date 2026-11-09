"""
Surface geometry: sag and gradient (sphere / conic / even asphere), ray-surface
intersection (Newton iterations from the vertex plane), aperture tests.
All functions are vectorised over rays and work in the LOCAL surface frame
(z along the surface axis, vertex at the origin).
"""
import numpy as np

__all__ = ["sag_and_grad", "intersect", "aperture_ok"]


def _noll(j):
    """(n, m) of the Noll index j (m < 0: sine term)."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    k = j - n * (n + 1) // 2 - 1                    # 0 .. n
    ms = [m for m in range(n % 2, n + 1, 2)]
    ms = sorted(set(ms))
    # Noll: within an order, |m| increases; even j -> cosine, odd j -> sine
    seq = []
    for m in ms:
        seq += [m] if m == 0 else [m, m]
    mabs = seq[k]
    if mabs == 0:
        return n, 0
    return n, (mabs if j % 2 == 0 else -mabs)


def _radial(n, m, rho):
    from math import factorial
    m = abs(m)
    R = np.zeros_like(rho)
    dR = np.zeros_like(rho)
    for s_ in range((n - m) // 2 + 1):
        c = (-1) ** s_ * factorial(n - s_) / (factorial(s_) * factorial((n + m) // 2 - s_)
                                              * factorial((n - m) // 2 - s_))
        p = n - 2 * s_
        R = R + c * rho ** p
        if p > 0:
            dR = dR + c * p * rho ** (p - 1)
    return R, dR


def zernike_sag(s, x, y):
    """Zernike Standard Sag term (OpticStudio SZERNSAG): sum A_j Z_j(rho, phi),
    Noll ordering and normalisation, rho = r / R_norm, phi from the local x axis.
    Returns z, dz/dx, dz/dy."""
    zern = s.get("zern", {})
    R0 = s.get("zern_rnorm", 1.0)
    xx = (x - s.get("Zernike Decenter X", 0.0)) / R0
    yy = (y - s.get("Zernike Decenter Y", 0.0)) / R0
    rho = np.hypot(xx, yy)
    phi = np.arctan2(yy, xx)
    z = np.zeros_like(rho); dzr = np.zeros_like(rho); dzp = np.zeros_like(rho)
    for j, a in zern.items():
        if a == 0:
            continue
        n, m = _noll(j)
        N = np.sqrt(n + 1) if m == 0 else np.sqrt(2 * (n + 1))
        R, dR = _radial(n, m, rho)
        if m == 0:
            ang, dang = 1.0, 0.0
        elif m > 0:
            ang, dang = np.cos(m * phi), -m * np.sin(m * phi)
        else:
            ang, dang = np.sin(-m * phi), -m * np.cos(-m * phi)
        z = z + a * N * R * ang
        dzr = dzr + a * N * dR * ang
        dzp = dzp + a * N * R * dang
    with np.errstate(divide="ignore", invalid="ignore"):
        cphi, sphi = np.cos(phi), np.sin(phi)
        inv = np.where(rho > 0, 1 / np.where(rho > 0, rho, 1), 0.0)
        dzx = (dzr * cphi - dzp * sphi * inv) / R0
        dzy = (dzr * sphi + dzp * cphi * inv) / R0
    return z, dzx, dzy


def sag_and_grad(s, x, y):
    """Sag z(x, y) and its partial derivatives for a STANDARD (sphere/conic),
    EVENASPH or SZERNSAG surface. Returns z, dz/dx, dz/dy and a mask of points
    outside the conic's domain."""
    c = 0.0 if not np.isfinite(s.get("R", np.inf)) else 1.0 / s["R"]
    k = s.get("conic", 0.0)
    r2 = x * x + y * y
    arg = 1 - (1 + k) * c * c * r2
    bad = arg < 0
    arg = np.where(bad, 0.0, arg)
    sq = np.sqrt(arg)
    z = c * r2 / (1 + sq)
    # d z / d(r2)
    with np.errstate(divide="ignore", invalid="ignore"):
        dzdr2 = np.where(sq > 0, c / (2 * sq), np.inf if c else 0.0)
    if s["type"] == "EVENASPH":
        for p, a in s.get("asph", {}).items():
            if a == 0:
                continue
            z = z + a * r2 ** (p / 2)
            dzdr2 = dzdr2 + a * (p / 2) * r2 ** (p / 2 - 1)
    dzdx, dzdy = 2 * x * dzdr2, 2 * y * dzdr2
    if s["type"] == "SZERNSAG" and s.get("zern"):
        zz, zx, zy = zernike_sag(s, x, y)
        z, dzdx, dzdy = z + zz, dzdx + zx, dzdy + zy
    return z, dzdx, dzdy, bad


def intersect(s, p, d):
    """p, d: local frame, shape (N,3). Returns point, unit normal, fail mask."""
    t = -p[:, 2] / d[:, 2]
    plane = s["type"] == "DGRATING" or (
        not np.isfinite(s.get("R", np.inf)) and s["type"] not in ("EVENASPH", "SZERNSAG"))
    if not plane:
        for _ in range(30):
            q = p + t[:, None] * d
            z, gx, gy, bad = sag_and_grad(s, q[:, 0], q[:, 1])
            f = q[:, 2] - z
            fp = d[:, 2] - gx * d[:, 0] - gy * d[:, 1]
            dt = f / fp
            t = t - dt
            if not np.any(np.abs(dt) >= 1e-10):      # converged (NaN rays ignored)
                break
    q = p + t[:, None] * d
    if plane:
        gx = gy = np.zeros(len(q)); bad = np.zeros(len(q), bool)
    else:
        _, gx, gy, bad = sag_and_grad(s, q[:, 0], q[:, 1])
    n = np.stack([-gx, -gy, np.ones_like(gx)], 1)
    n /= np.linalg.norm(n, axis=1)[:, None]
    fail = bad | ~np.isfinite(q).all(1) | ~np.isfinite(n).all(1)
    return q, n, fail


def aperture_ok(s, q):
    at = s.get("ap_type")
    if at is None or "None" in at:
        return np.ones(len(q), bool)
    x = q[:, 0] - s.get("X- Decenter", 0.0)
    y = q[:, 1] - s.get("Y- Decenter", 0.0)
    if at.startswith("Circular Aperture"):
        r = np.hypot(x, y)
        return (r <= s.get("Maximum Radius", np.inf)) & (r >= s.get("Minimum Radius", 0.0))
    if at.startswith("Floating Aperture"):
        return np.hypot(x, y) <= s.get("Maximum Radius", np.inf)
    if at.startswith("Rectangular Aperture"):
        return (np.abs(x) <= s["X Half Width"]) & (np.abs(y) <= s["Y Half Width"])
    if at.startswith("Rectangular Obscuration"):
        return ~((np.abs(x) <= s["X Half Width"]) & (np.abs(y) <= s["Y Half Width"]))
    raise ValueError(f"aperture type {at} not handled")


