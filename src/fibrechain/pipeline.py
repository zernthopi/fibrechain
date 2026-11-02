"""
End-to-end computation for one configuration. Writes to `outdir`:

  fibre_output.npz        fibre output EE(theta) and maps (fibre.py)
  trans_maps.npz          blocking surface per fibre-output direction (maps.py)
  stage_transmissions.npz azimuthally averaged t_pre, t_obs, t_all (budget.py)
  budget.csv              per kernel / FRD / field / wavelength
  budget_summary.csv      on-axis and slit-weighted, wavelength-averaged
  uniform_rays.npy        OpticStudio-like uniform-ray reference (checks.py)
  hexagon_direct.npy      direct trace of the scrambled lenslet hexagon
  fnumber_check.npy       working F/#: OpticStudio F/# Data vs this tracer
  validation.json         consistency checks (see validate())
"""
import json
import time
from pathlib import Path
import numpy as np
from .config import load_config
from .fibre import fibre_output
from .maps import transmission_maps
from .budget import energy_budget, summarise, write_csv
from .checks import uniform_rays, hexagon_samples, direct_trace, working_fnumber
from .zemax import parse_fnumber_data

__all__ = ["run", "validate"]


def _log(msg, t0):
    print(f"[{time.time() - t0:6.1f} s] {msg}", flush=True)


def run(config_path, outdir, steps=("fibre", "maps", "budget", "uniform", "hexagon", "validate")):
    cfg = load_config(config_path)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sysm = cfg.system()
    _log(f"{cfg.name}: {len(sysm.path)} surfaces traced, {len(cfg.fields)} fields, "
         f"{len(cfg.wavelengths)} wavelengths, lenslet half-angle {np.degrees(cfg.th_o):.3f} deg", t0)
    if "fibre" in steps:
        fo = fibre_output(cfg.th_o, cfg.f_tel, cfg.eps_tel, cfg.frd_fwhm_deg, cfg.frd_nominal,
                          cfg.grid_n, cfg.grid_half)
        np.savez_compressed(out / "fibre_output.npz", **fo)
        _log("fibre output distributions -> fibre_output.npz", t0)
    if "maps" in steps:
        tm = transmission_maps(sysm, cfg.pupil, cfg.unique_fields(), cfg.wavelengths,
                               cfg.dtheta_deg, cfg.theta_max_deg, cfg.nphi)
        np.savez_compressed(out / "trans_maps.npz", **tm)
        _log(f"transmission maps ({tm['blocked'].size:,} rays) -> trans_maps.npz", t0)
    if "budget" in steps:
        fo, tm = np.load(out / "fibre_output.npz"), np.load(out / "trans_maps.npz")
        rows, st = energy_budget(tm, fo, cfg.fields, cfg.field_weights, cfg.stages,
                                 cfg.mirror_fields)
        write_csv(rows, out / "budget.csv")
        write_csv(summarise(rows), out / "budget_summary.csv")
        np.savez_compressed(out / "stage_transmissions.npz", **st)
        _log("energy budget -> budget.csv, budget_summary.csv", t0)
    if "uniform" in steps:
        u = uniform_rays(sysm, cfg.pupil, cfg.fields, cfg.wavelengths, cfg.group_of(),
                         cfg.uniform_n)
        np.save(out / "uniform_rays.npy", np.array(u, dtype=object), allow_pickle=True)
        _log("uniform-ray reference -> uniform_rays.npy", t0)
    if "hexagon" in steps:
        th, ph = hexagon_samples(cfg.th_o, cfg.hex_samples)
        h = direct_trace(sysm, cfg.pupil, cfg.fields, cfg.wavelengths, cfg.group_of(), th, ph)
        np.save(out / "hexagon_direct.npy", np.array(h, dtype=object), allow_pickle=True)
        _log(f"direct trace of the scrambled hexagon ({len(th)} rays per case) -> "
             "hexagon_direct.npy", t0)
    if "validate" in steps:
        v = validate(cfg, out)
        (out / "validation.json").write_text(json.dumps(v, indent=2))
        _log("validation -> validation.json", t0)
        for k, val in v.items():
            print(f"    {k}: {val}")
    return out


def validate(cfg, out):
    """Internal consistency checks (no reference to OpticStudio needed,
    except the uniform-ray number that can be compared with it)."""
    import csv
    out = Path(out)
    fo = np.load(out / "fibre_output.npz")
    U = np.load(out / "uniform_rays.npy", allow_pickle=True)
    H = np.load(out / "hexagon_direct.npy", allow_pickle=True)
    rows = [r for r in csv.DictReader(open(out / "budget.csv"))]
    W = dict(zip(cfg.fields, cfg.field_weights))
    uni = {(u[0], u[1], round(u[2], 4)): 100 * u[3] for u in U}
    slit = sum(W[(u[0], u[1])] * 100 * u[3] for u in U) / (sum(cfg.field_weights) * len(cfg.wavelengths))
    onax = np.mean([100 * u[3] for u in U if (u[0], u[1]) == (0, 0)])
    sym = max(abs(uni[(fx, fy, round(w, 4))] - uni[(-fx, fy, round(w, 4))])
              for fx, fy in cfg.fields if fx > 0 for w in cfg.wavelengths)
    mp = {(int(float(r["fx"])), int(float(r["fy"])), round(float(r["wl"]), 4)): float(r["final"])
          for r in rows if r["kernel"] == "noK" and float(r["frd"]) == 0}
    d = [100 * h[3] - mp[(h[0], h[1], round(h[2], 4))] for h in H]
    zf = parse_fnumber_data(cfg.prescription)
    sysm = cfg.system()
    fdev = []
    ftab = []
    for (fx, fy, w), (zt, zs) in zf.items():
        wi = int(np.argmin(np.abs(cfg.wavelengths - w)))
        t, s_ = working_fnumber(sysm, cfg.pupil, fx, fy, wi, w)
        ftab.append([fx, fy, w, zt, zs, float(t), float(s_)])
        fdev += [abs(t / zt - 1), abs(s_ / zs - 1)]
    fdev = np.array(fdev)
    np.save(out / "fnumber_check.npy", np.array(ftab, float))
    th_na = np.arcsin(cfg.pupil.r_stop / np.hypot(cfg.pupil.r_stop, cfg.pupil.z_stop))
    return {
        "uniform_rays_slit_average_pct": round(float(slit), 2),
        "uniform_rays_on_axis_pct": round(float(onax), 2),
        "uniform_rays_max_plus_minus_x_difference_pt": round(float(sym), 3),
        "working_fnumber_vs_opticstudio_max_rel_dev_pct": round(100 * float(np.nanmax(fdev)), 2),
        "working_fnumber_vs_opticstudio_median_rel_dev_pct": round(100 * float(np.nanmedian(fdev)), 2),
        "working_fnumber_cases_compared": f"{int(np.sum(np.isfinite(fdev)))} of {len(fdev)}",
        "hexagon_direct_vs_map_budget_max_abs_pt": round(float(np.max(np.abs(d))), 3),
        "hexagon_direct_vs_map_budget_mean_pt": round(float(np.mean(d)), 3),
        "fibre_flux_within_grid_K": round(float(fo["EE_geo"][-1]), 6),
        "flux_beyond_traced_cone_max_pct": round(max(float(r["tail"]) for r in rows), 4),
        "lenslet_hexagon_apothem_deg": round(float(np.degrees(cfg.th_o)), 3),
        "lenslet_hexagon_corner_deg": round(float(np.degrees(2 * cfg.th_o / np.sqrt(3))), 3),
        "object_NA_half_angle_deg": round(float(np.degrees(th_na)), 3),
        "hexagon_to_NA_disc_solid_angle": round(float(2 * np.sqrt(3) * cfg.th_o ** 2 /
                                                      (np.pi * th_na ** 2)), 4),
    }
