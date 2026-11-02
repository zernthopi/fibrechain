"""
Angular distribution at the fibre output (the spectrograph pupil), from the
lenslet to the fibre exit, on a square grid of output angles (radians).

  O   = 1_Hex(th_o) / A_hex            lenslet top hat (flux 1 = 100 %): the
                                       uniformly illuminated hexagonal lenslet
                                       seen from the fibre core
  I   = O (x) K_geo                    far field at the fibre input; K_geo =
                                       telescope pupil annulus in angle, outer
                                       radius 1/(2 F_tel), inner eps/(2 F_tel)
                                       (geometric FRD: every point of the
                                       lenslet sees the whole pupil cone)
  P   = < I (x) K_FRD >_az             fibre FRD (Gaussian, FWHM convention of
                                       Zhang, Zheng & Saunders 2016) and
                                       azimuthal scrambling by the fibre
The 'noK' variant skips K_geo (I = O). Convolutions are done by FFT on an
N x N grid; the encircled energy EE(theta) of the scrambled output is the
cumulative sum of the 2-D map sorted by radius (exact for a circularly
symmetric output, no fit involved).
"""
import numpy as np
from numpy.fft import rfft2, irfft2, fftshift

__all__ = ["AngleGrid", "fibre_output", "FWHM_TO_SIGMA"]

FWHM_TO_SIGMA = 1 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


class AngleGrid:
    def __init__(self, n=4096, half=0.40):
        self.n, self.half = n, half
        self.x = np.linspace(-half, half, n)
        self.dx = self.x[1] - self.x[0]
        self.X, self.Y = np.meshgrid(self.x, self.x)
        self.R = np.hypot(self.X, self.Y)
        self.order = np.argsort(self.R.ravel())
        self.rs = self.R.ravel()[self.order]

    def hexagon(self, apothem):
        """Regular hexagon, flats normal to x (as the lenslet in the slit)."""
        m = np.ones_like(self.X, dtype=bool)
        for k in range(6):
            t = k * np.pi / 3
            m &= (self.X * np.cos(t) + self.Y * np.sin(t)) <= apothem
        return m.astype(float)

    def annulus(self, r_out, r_in):
        return ((self.R <= r_out) & (self.R >= r_in)).astype(float)

    def gaussian(self, sigma):
        g = np.exp(-(self.R ** 2) / (2 * sigma ** 2))
        return g / g.sum()

    @staticmethod
    def conv(a, b):
        o = fftshift(irfft2(rfft2(a) * rfft2(b), s=a.shape))
        o[o < 0] = 0
        return o

    def encircled(self, img, radii):
        cs = np.cumsum(img.ravel()[self.order])
        return np.interp(radii, self.rs, cs)

    def scramble(self, img):
        """Azimuthal mean (perfect scrambling) on the same grid."""
        rb = np.minimum((self.R / self.dx).astype(np.int64), int(self.half / self.dx) + 2)
        prof = np.bincount(rb.ravel(), img.ravel()) / np.maximum(np.bincount(rb.ravel()), 1)
        return _norm(prof[rb])


def _norm(img):
    return img / img.sum()


def fibre_output(th_o, f_tel, eps_tel, frd_fwhm_deg, frd_nominal=1.0, n=4096, half=0.40,
                 map_half=0.23, map_step=3, n_theta=3000, verbose=False):
    """Fibre output distributions for both variants ('K', 'noK') and every
    fibre FRD level. Returns a dict of arrays (saved as fibre_output.npz):
      THETA, EE_hex, EE_geo, EE_frd_{K|noK}_{fwhm}   encircled energy vs angle
      HEX, GEO, SCR_nom_{K|noK}, map_extent        maps for the figures
      th_o, FRD_FWHM_DEG, FRD_NOMINAL"""
    G = AngleGrid(n, half)
    th_k = 1 / (2 * f_tel)
    KER = _norm(G.annulus(th_k, eps_tel * th_k))
    HEX = _norm(G.hexagon(th_o))
    GEO = _norm(G.conv(HEX, KER))
    THETA = np.linspace(0, half, n_theta)
    out = {"THETA": THETA, "FRD_FWHM_DEG": np.array(frd_fwhm_deg, float),
           "FRD_NOMINAL": frd_nominal, "th_o": th_o,
           "EE_hex": G.encircled(HEX, THETA), "EE_geo": G.encircled(GEO, THETA)}
    crop = int((half - map_half) / G.dx)

    def for_map(v):
        return v[crop:n - crop, crop:n - crop][::map_step, ::map_step]
    out["map_extent"] = np.array([G.x[crop], G.x[n - crop - 1]])
    out["HEX"], out["GEO"] = for_map(HEX), for_map(GEO)
    for var, base in (("K", GEO), ("noK", HEX)):
        for deg in frd_fwhm_deg:
            img = base if deg == 0 else _norm(G.conv(base, G.gaussian(np.radians(deg) * FWHM_TO_SIGMA)))
            out[f"EE_frd_{var}_{deg:g}"] = G.encircled(img, THETA)
            if deg == frd_nominal:
                out[f"SCR_nom_{var}"] = for_map(G.scramble(img))
            if verbose:
                print(f"  fibre output {var:3s} FRD {deg:g} deg done")
    return out
