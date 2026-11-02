"""
Surface geometry: sag and gradient (sphere / conic / even asphere), ray-surface
intersection (Newton iterations from the vertex plane), aperture tests.
All functions are vectorised over rays and work in the LOCAL surface frame
(z along the surface axis, vertex at the origin).
"""
import numpy as np

__all__ = ["sag_and_grad", "intersect", "aperture_ok"]


def sag_and_grad(s, x, y):
    """Sag z(x, y) and its partial derivatives for a STANDARD (sphere/conic) or
    EVENASPH surface. Returns z, dz/dx, dz/dy and a mask of points outside the
    conic's domain."""
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
    return z, dzdx, dzdy, bad


def intersect(s, p, d):
    """p, d: local frame, shape (N,3). Returns point, unit normal, fail mask."""
    t = -p[:, 2] / d[:, 2]
    plane = s["type"] in ("DGRATING", "SZERNSAG") or (
        not np.isfinite(s.get("R", np.inf)) and s["type"] != "EVENASPH")
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


