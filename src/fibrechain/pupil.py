"""
Object-side ray generation for a fibre-fed spectrograph whose slit (the fibre
exits) is curved about the stop.

Zemax model reproduced here ("object NA" aperture with real ray aiming): the object surface is a sphere of radius R_obj centred on the stop,
the stop has radius R_stop for the object NA, and the pupil is decentred by the
vignetting factor VDY (same for all fields). Two ray sets are needed:

  * Zemax-like rays, from normalised pupil coordinates (px, py): used to
    reproduce the OpticStudio geometric (uniform-ray) vignetting number;
  * fibre-output rays, from angles (theta, phi) about the FIBRE AXIS, taken as
    the chief ray of the vignetted pupil (towards the stop point
    (0, VDY * R_stop)); theta = fibre output angle, phi measured from the slit
    direction (global x projected perpendicular to the axis).
"""
from dataclasses import dataclass
import numpy as np

__all__ = ["SlitPupil"]


@dataclass
class SlitPupil:
    r_obj: float                 # object (curved slit) radius of curvature [mm]
    z_stop: float                # stop position relative to the slit vertex [mm]
    r_stop: float                # stop radius giving the object NA [mm]
    vdy: float                   # pupil decentre (vignetting factor VDY)

    def object_point(self, fx, fy):
        """Point of the curved slit at field (fx, fy) [mm]."""
        c = 1 / self.r_obj
        r2 = fx * fx + fy * fy
        z = c * r2 / (1 + np.sqrt(1 - c * c * r2))
        return np.array([fx, fy, z], float)

    def pupil_rays(self, fx, fy, px, py):
        """Zemax-like rays for normalised pupil coordinates (px, py), VDY
        included, aimed at the real stop. Returns origins and unit directions."""
        px, py = np.asarray(px, float), np.asarray(py, float)
        P0 = self.object_point(fx, fy)
        S = np.stack([px * self.r_stop, (py + self.vdy) * self.r_stop,
                      np.full_like(px, self.z_stop)], 1)
        d = P0[None, :] - S
        d /= np.linalg.norm(d, axis=1)[:, None]
        return np.repeat(P0[None, :], len(px), 0), d

    def fibre_axis(self, fx, fy):
        """Fibre exit point and fibre axis (chief ray of the vignetted pupil)."""
        P0 = self.object_point(fx, fy)
        S0 = np.array([0.0, self.vdy * self.r_stop, self.z_stop])
        a = P0 - S0
        return P0, a / np.linalg.norm(a)

    def fibre_frame(self, fx, fy):
        """Fibre exit point P0, axis a and transverse unit vectors e1 (slit
        direction, global x projected perpendicular to a) and e2 = a x e1."""
        P0, a = self.fibre_axis(fx, fy)
        e1 = np.array([1.0, 0, 0]) - a[0] * a
        e1 /= np.linalg.norm(e1)
        return P0, a, e1, np.cross(a, e1)

    def fibre_rays(self, fx, fy, theta, phi):
        """Rays leaving the fibre exit at output angles (theta, phi) [rad]."""
        P0, a, e1, e2 = self.fibre_frame(fx, fy)
        theta, phi = np.ravel(theta), np.ravel(phi)
        d = (np.cos(theta)[:, None] * a + np.sin(theta)[:, None] *
             (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2))
        return np.repeat(P0[None, :], len(d), 0), d
