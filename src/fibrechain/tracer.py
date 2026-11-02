"""
Sequential real ray tracer.

Rays are traced in GLOBAL coordinates. At each surface of the path they are
moved to the local frame (p_l = R^T (p - o), d_l = R^T d), intersected with the
surface sag, tested against the surface aperture, refracted (vector Snell law,
total internal reflection stops the ray), reflected (mirrors) or diffracted
(DGRATING, plane VPH / grating, lines parallel to the local x axis), and moved
back to the global frame. For every ray the tracer returns the id of the first
surface that stops it (-1 if it reaches the image), so that losses can be
attributed to a surface along the path.

Supported surface types: STANDARD, EVENASPH, SZERNSAG (traced as its base
plane: the Zernike sag terms matter for image quality, not for vignetting),
DGRATING. COORDBRK surfaces carry no optics and are skipped (their effect is
already contained in the global frames of the export).
"""
import numpy as np
from .zemax import parse_prescription
from .optics import intersect, aperture_ok

__all__ = ["System"]


class System:
    """Optical system read from a prescription export.

    path          : prescription .txt file
    first_surface : first surface traced (the object side of the traced part,
                    e.g. the slit plate); earlier surfaces are ignored
    grating_sign  : sign convention of the grating equation (+1 reproduces
                    OpticStudio for a validated VPH; -1 makes every ray
                    evanescent there, which is an easy check)
    """

    def __init__(self, path, first_surface=3, grating_sign=+1):
        self.S = parse_prescription(path)
        ids = sorted(k for k in self.S if k is not None and k >= first_surface)
        self.path = [k for k in ids if self.S[k]["type"] != "COORDBRK"] + [-999]
        self.grating_sign = grating_sign

    def surface(self, sid):
        return self.S[sid]

    def trace(self, p0, d0, wl_index, wl_um, record=None, stop_at=None):
        """p0, d0 global (N,3) at the object side, already pointing forward.
        Returns final points, directions, and index of the blocking surface
        (-1 = reached the image)."""
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            return self._trace(p0, d0, wl_index, wl_um, record, stop_at)

    def _trace(self, p0, d0, wl_index, wl_um, record, stop_at):
        p, d = p0.copy(), d0 / np.linalg.norm(d0, axis=1)[:, None]
        blocked = np.full(len(p), -1)
        alive = np.ones(len(p), bool)
        n_prev = 1.0
        for sid in self.path:
            s = self.S[sid]
            R, o = s["Rg"], s["og"]
            pl = (p - o) @ R          # = R^T (p - o)
            dl = d @ R
            q, nrm, fail = intersect(s, pl, dl)
            ok = aperture_ok(s, q) & ~fail
            newly = alive & ~ok
            blocked[newly] = sid
            alive &= ok
            # orient the normal along the propagation direction
            sgn = np.sign(np.sum(nrm * dl, 1))
            sgn[sgn == 0] = 1
            nrm = nrm * sgn[:, None]
            n_next = s["n"][wl_index] if "n" in s else n_prev
            if s.get("mirror"):
                dl = dl - 2 * np.sum(dl * nrm, 1)[:, None] * nrm
            elif s["type"] == "DGRATING":
                # n2 * (d2 . y) = n1 * (d1 . y) + sign * m * lambda * T ; x kept
                m, T = s["order"], s["lpum"]
                ky = n_prev * dl[:, 1] + self.grating_sign * m * wl_um * T
                kx = n_prev * dl[:, 0]
                kz2 = n_next ** 2 - kx ** 2 - ky ** 2
                evan = kz2 < 0
                blocked[alive & evan] = sid
                alive &= ~evan
                kz = np.sqrt(np.clip(kz2, 0, None)) * np.sign(dl[:, 2])
                dl = np.stack([kx, ky, kz], 1) / n_next
            elif abs(n_next - n_prev) > 1e-12:
                eta = n_prev / n_next
                ci = np.sum(dl * nrm, 1)
                st2 = eta ** 2 * (1 - ci ** 2)
                tir = st2 > 1
                blocked[alive & tir] = sid
                alive &= ~tir
                dl = eta * dl + (np.sqrt(np.clip(1 - st2, 0, None)) - eta * ci)[:, None] * nrm
            dl /= np.linalg.norm(dl, axis=1)[:, None]
            p = q @ R.T + o
            d = dl @ R.T
            n_prev = n_next
            if record is not None:
                record[sid] = (p.copy(), alive.copy())
            if stop_at is not None and sid == stop_at:
                break
        return p, d, blocked
