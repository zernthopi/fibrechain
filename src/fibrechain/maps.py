"""
Spectrograph transmission in FIBRE-OUTPUT-ANGLE space.

For each field and wavelength, a polar grid of directions (theta, phi) about
the fibre axis is traced from the fibre exit through every aperture of the
system; for each direction the id of the first surface that stops it is kept
(-1 = reaches the detector). theta: rings of width dtheta from 0 to theta_max
(ring centres), phi: nphi equal steps. This map does not depend on the fibre;
the fibre output is applied afterwards (budget.py), so FRD / kernel studies
never require a new trace.
"""
import time
import numpy as np

__all__ = ["polar_grid", "transmission_maps"]


def polar_grid(dtheta_deg=0.05, theta_max_deg=13.0, nphi=180):
    dth = np.radians(dtheta_deg)
    th = (np.arange(int(round(np.radians(theta_max_deg) / dth))) + 0.5) * dth
    phi = (np.arange(nphi) + 0.5) * 2 * np.pi / nphi
    return th, phi, dth


def transmission_maps(system, pupil, fields, wavelengths, dtheta_deg=0.05,
                      theta_max_deg=13.0, nphi=180, verbose=False):
    """Returns dict(blocked[field, wl, theta, phi] int16, TH, PHI, DTH,
    fields, WL)."""
    TH, PHI, DTH = polar_grid(dtheta_deg, theta_max_deg, nphi)
    TT, PP = np.meshgrid(TH, PHI, indexing="ij")
    blocked = np.zeros((len(fields), len(wavelengths), len(TH), len(PHI)), np.int16)
    t0 = time.time()
    for fi, (fx, fy) in enumerate(fields):
        p, d = pupil.fibre_rays(fx, fy, TT, PP)
        for wi, w in enumerate(wavelengths):
            _, _, b = system.trace(p, d, wi, w)
            blocked[fi, wi] = b.reshape(len(TH), len(PHI))
        if verbose:
            print(f"  map field ({fx}, {fy}) traced, {time.time() - t0:.0f} s")
    return dict(blocked=blocked, TH=TH, PHI=PHI, DTH=DTH, fields=np.array(fields),
                WL=np.asarray(wavelengths))
