"""
Linearised ("unfolded") layout of the traced system, for drawings.

Every surface hit X_k is mapped with the frame of the chief ray arriving on
surface k:  s = S_k + c_k (X - C_k).d_k  and  h = (X - C_k).u_k (meridional,
dispersion plane) or (X - C_k).v_k (sagittal, slit plane), S_k being the chief
path length. The transverse basis is parallel-transported through refractions
and reflected at mirrors, so mirrors are unfolded; from surface `fold_from` on
the system is drawn as built (one fixed frame), e.g. a Schmidt camera with
the detector inside the beam. Faces tilted by more than `tmax_deg` to the
chief ray (dichroic, prisms, VPH) have their tilt compressed to `tmax_deg` so
that the dispersers do not overlap; heights and path lengths are true.
"""
import numpy as np
from .optics import sag_and_grad

__all__ = ["UnfoldedLayout"]


def _frames(S, keep, Pc, u0=None):
    dirs = []
    for k in range(len(Pc) - 1):
        d = Pc[k + 1] - Pc[k]
        L = np.linalg.norm(d)
        dirs.append((d / L if L > 1e-9 else None, L))
    for k in range(len(dirs)):
        if dirs[k][0] is None:
            j = next(j for j in range(k + 1, len(dirs)) if dirs[j][0] is not None)
            dirs[k] = (dirs[j][0], 0.0)
    d = dirs[0][0]
    u = np.array([0.0, 1.0, 0.0]) if u0 is None else u0
    u = u - (u @ d) * d; u /= np.linalg.norm(u)
    v = np.cross(d, u)
    segs, s0 = [], 0.0
    path = [None] + list(keep)
    for k, (dk, L) in enumerate(dirs):
        if k > 0 and S[path[k]].get("mirror"):
            nrm = dk - d; nrm /= np.linalg.norm(nrm)
            u = u - 2 * (u @ nrm) * nrm
            v = v - 2 * (v @ nrm) * nrm
        if k > 0:
            u = u - (u @ dk) * dk; u /= np.linalg.norm(u)
            v = v - (v @ dk) * dk - (v @ u) * u; v /= np.linalg.norm(v)
        d = dk
        segs.append(dict(C=Pc[k], d=d, u=u.copy(), v=v.copy(), s0=s0, L=L))
        s0 += L
    return segs


class UnfoldedLayout:
    def __init__(self, system, pupil, wavelengths, keep, fold_from=None, tmax_deg=14.0,
                 parallel_faces=None, bodies=(), ref_field=(0, 0), ref_wl=None):
        """keep: surface ids drawn (and used for the chief polyline, virtual
        dummy surfaces must be left out); parallel_faces: {ref id: [ids]}
        faces drawn parallel to the reference face (e.g. a VPH sandwich);
        bodies: pairs of faces whose interior is filled."""
        self.sys, self.pupil, self.WL = system, pupil, np.asarray(wavelengths)
        self.keep, self.fold, self.bodies = list(keep), fold_from, list(bodies)
        self.tmax = np.radians(tmax_deg)
        ref_wl = len(self.WL) // 2 if ref_wl is None else ref_wl
        P, _ = self.trace_record(*ref_field, np.array([0.0]), ref_wl)
        S = self.sys.S
        self.segs = _frames(S, self.keep, P[:, 0, :])
        self.C = P[1:, 0, :]
        self.S_SURF = {sid: self.segs[k]["s0"] + self.segs[k]["L"]
                       for k, sid in enumerate(self.keep)}
        self.c, sgn = [], {}
        for k, sid in enumerate(self.keep):
            n = S[sid]["Rg"][:, 2]
            seg = self.segs[k]
            a0 = np.arctan2(n @ seg["u"], n @ seg["d"])
            a0 = (a0 + np.pi / 2) % np.pi - np.pi / 2
            a = abs(a0)
            sgn[sid] = np.sign(a0)
            flat = not np.isfinite(S[sid].get("R", np.inf)) or S[sid]["type"] in ("DGRATING", "SZERNSAG")
            self.c.append(np.tan(self.tmax) / np.tan(a) if (flat and a > self.tmax) else 1.0)
        for ref, ids in (parallel_faces or {}).items():
            for sid in ids:
                k = self.keep.index(sid)
                if sgn[sid] != sgn[ref]:
                    self.c[k] = -self.c[k]

    # ---------------------------------------------------------------- tracing
    def trace_record(self, fx, fy, py, wi, px=None):
        py = np.asarray(py, float)
        px = np.zeros_like(py) if px is None else np.asarray(px, float)
        p0, d0 = self.pupil.pupil_rays(fx, fy, px, py)
        rec = {}
        self.sys.trace(p0, d0, wi, self.WL[wi], record=rec)
        P = np.stack([p0] + [rec[s][0] for s in self.keep], 0)
        A = np.stack([np.ones(len(py), bool)] + [rec[s][1] for s in self.keep], 0)
        return P, A

    # ---------------------------------------------------------------- mapping
    def map(self, k, X, plane="u"):
        kf = self.keep.index(self.fold) if self.fold is not None else None
        if kf is not None and k >= kf:
            seg, C, S0, c = self.segs[kf], self.C[kf], self.S_SURF[self.fold], 1.0
        else:
            seg, C, S0, c = self.segs[k], self.C[k], self.S_SURF[self.keep[k]], self.c[k]
        D = X - C
        return S0 + c * (D @ seg["d"]), D @ (seg["u"] if plane == "u" else seg["v"])

    def spos(self, sid):
        """Drawn path coordinate of the chief ray on surface sid."""
        k = self.keep.index(sid)
        return float(self.map(k, self.C[k])[0])

    def fan(self, fx, fy, wi, plane="u", n=2):
        t = np.linspace(-1, 1, n)
        if plane == "u":
            P, A = self.trace_record(fx, fy, t, wi)
        else:
            P, A = self.trace_record(fx, fy, np.zeros(n), wi, px=t)
        pts = np.zeros((len(self.keep) + 1, n, 2))
        seg = self.segs[0]
        D = P[0] - self.C[0]
        pts[0, :, 0] = D @ seg["d"]
        pts[0, :, 1] = D @ (seg["u"] if plane == "u" else seg["v"])
        for k in range(len(self.keep)):
            pts[k + 1, :, 0], pts[k + 1, :, 1] = self.map(k, P[k + 1], plane)
        return pts, A

    def profile(self, sid, plane="u", n=120):
        k = self.keep.index(sid)
        s = self.sys.S[sid]
        ax_ = "Y" if plane == "u" else "X"
        c = s.get(f"{ax_}- Decenter", 0.0)
        h = s.get(f"{ax_} Half Width", s.get("Maximum Radius", None))
        if h is None:
            return None
        t = np.linspace(c - h, c + h, n)
        x, y = (np.zeros_like(t), t) if plane == "u" else (t, np.zeros_like(t))
        if s["type"] in ("DGRATING", "SZERNSAG") or not np.isfinite(s.get("R", np.inf)):
            z = np.zeros_like(t)
        else:
            z = sag_and_grad(s, x, y)[0]
        G = np.stack([x, y, z], 1) @ s["Rg"].T + s["og"]
        return self.map(k, G, plane)

    # ---------------------------------------------------------------- drawing
    def draw_beam(self, ax, fx, fy, wi, col, plane="u", alpha=0.16, lw=0.6, z=2):
        pts, _ = self.fan(fx, fy, wi, plane, n=2)
        a, b = pts[:, 0], pts[:, 1]
        for k in range(len(pts) - 1):
            ax.fill([a[k, 0], a[k + 1, 0], b[k + 1, 0], b[k, 0]],
                    [a[k, 1], a[k + 1, 1], b[k + 1, 1], b[k, 1]], color=col,
                    alpha=alpha, lw=0, zorder=z)
        for r in (a, b):
            ax.plot(r[:, 0], r[:, 1], color=col, lw=lw, zorder=z + 1)
        return pts

    def draw_surfaces(self, ax, colours, plane="u", lw=1.3, z=6, thick=()):
        for sid in self.keep:
            if sid not in colours:
                continue
            pr = self.profile(sid, plane)
            if pr is None:
                continue
            ax.plot(pr[0], pr[1], color=colours[sid], lw=3.0 if sid in thick else lw,
                    solid_capstyle="butt", zorder=z + (2 if sid in thick else 0))
        for a, b in self.bodies:
            pa, pb = self.profile(a, plane), self.profile(b, plane)
            if pa is None or pb is None:
                continue
            ax.fill(np.r_[pa[0], pb[0][::-1]], np.r_[pa[1], pb[1][::-1]], color=colours[a],
                    alpha=0.25, lw=0, zorder=z - 1)
