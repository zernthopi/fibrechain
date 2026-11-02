"""
Energy budget: the fibre output encircled energy EE(theta) weighted by the
azimuthally averaged spectrograph transmission of each stage.

  t_X(theta_i) = fraction of the nphi directions of ring i that pass stage X
                 X = pre (not stopped before the camera), obs (not stopped by
                 the obscuration either), all (reach the detector)
  w_i          = EE(theta_i + dtheta/2) - EE(theta_i - dtheta/2)
  F_X          = sum_i w_i t_X(theta_i)        flux after stage X (lenslet = 1)
  tail         = 1 - EE(theta_max)             flux outside the traced cone
The azimuthal average is exact because the fibre output is scrambled
(circularly symmetric).
"""
import csv
import numpy as np

__all__ = ["stage_transmissions", "ring_weights", "energy_budget", "summarise",
           "write_csv"]


def stage_transmissions(blocked, stages):
    """blocked: (ntheta, nphi) -> t_pre, t_obs, t_all (ntheta,)"""
    pre_lost = np.isin(blocked, list(stages["pre"]))
    obs_lost = np.isin(blocked, list(stages["obs"]))
    t_pre = 1 - pre_lost.mean(1)
    t_obs = t_pre - obs_lost.mean(1)
    t_all = (blocked == -1).mean(1)
    return t_pre, t_obs, t_all


def ring_weights(ee, theta, TH, DTH):
    edges = np.r_[0.0, TH + DTH / 2]
    E = np.interp(edges, theta, ee)
    return np.diff(E), 1 - E[-1]


def energy_budget(maps, fibre, fields, field_weights, stages, mirror_fields=True):
    """One row per (kernel variant, FRD, field, wavelength), values in % of
    the flux entering the lenslet."""
    B, TH, DTH = maps["blocked"], maps["TH"], float(maps["DTH"])
    UF = [tuple(int(v) for v in f) for f in maps["fields"]]
    WL = maps["WL"]

    def ufield(fx, fy):
        return UF.index((-abs(fx), fy) if mirror_fields else (fx, fy))
    T = {(fi, wi): stage_transmissions(B[fi, wi], stages)
         for fi in range(len(UF)) for wi in range(len(WL))}
    rows = []
    for var in ("K", "noK"):
        for d in [float(v) for v in fibre["FRD_FWHM_DEG"]]:
            w, tail = ring_weights(fibre[f"EE_frd_{var}_{d:g}"], fibre["THETA"], TH, DTH)
            for (fx, fy), fw in zip(fields, field_weights):
                fi = ufield(fx, fy)
                for wi, wl in enumerate(WL):
                    t_pre, t_obs, t_all = T[(fi, wi)]
                    a_pre, a_obs, fin = (np.sum(w * t) for t in (t_pre, t_obs, t_all))
                    rows.append(dict(kernel=var, frd=d, fx=fx, fy=fy, field_w=fw, wl=float(wl),
                                     after_pre=100 * a_pre, after_obs=100 * a_obs,
                                     final=100 * fin, loss_pre=100 * (1 - tail - a_pre),
                                     loss_obs=100 * (a_pre - a_obs), loss_cam=100 * (a_obs - fin),
                                     tail=100 * tail, obs_of_coll=100 * (a_pre - a_obs) / a_pre,
                                     det_of_coll=100 * fin / a_pre))
    stage_t = {f"t_{k}_{fi}_{wi}": v[j] for (fi, wi), v in T.items()
               for j, k in enumerate(("pre", "obs", "all"))}
    return rows, dict(TH=TH, **stage_t)


def summarise(rows):
    """On-axis and slit-weighted averages (field weights), mean over wavelength."""
    out = []
    for where in ("on-axis", "slit avg"):
        for var in ("K", "noK"):
            for d in sorted({r["frd"] for r in rows}):
                R = [r for r in rows if r["kernel"] == var and r["frd"] == d and
                     (where != "on-axis" or (r["fx"], r["fy"]) == (0, 0))]
                W = np.array([r["field_w"] for r in R], float)
                a = {k: float(np.sum(W * np.array([r[k] for r in R])) / W.sum())
                     for k in ("after_pre", "after_obs", "final", "loss_pre", "loss_obs",
                               "loss_cam", "tail")}
                a["det_of_coll"] = 100 * a["final"] / a["after_pre"]
                a["obs_of_coll"] = 100 * a["loss_obs"] / a["after_pre"]
                out.append(dict(where=where, kernel=var, frd=d, **a))
    return out


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
