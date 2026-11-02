"""
Configuration of a study, read from a TOML file (see the README for the
layout). Paths are resolved relative to the configuration file.
"""
from dataclasses import dataclass, field
from pathlib import Path
import tomllib
import numpy as np
from .pupil import SlitPupil
from .tracer import System

__all__ = ["Config", "load_config"]


@dataclass
class Config:
    name: str
    prescription: Path
    first_surface: int
    grating_sign: int
    wavelengths: np.ndarray                 # [um], in the order of the export
    fields: list                            # [(fx, fy)] [mm]
    field_weights: list
    pupil: SlitPupil
    # fibre output model
    f_tel: float                            # telescope focal ratio
    eps_tel: float                          # telescope central obscuration (linear)
    lenslet_d_in: float                     # lenslet input aperture, flat to flat [um]
    lenslet_d_pup: float                    # pupil image on the fibre core [um]
    frd_fwhm_deg: list
    frd_nominal: float
    grid_n: int = 4096
    grid_half: float = 0.40                 # half-width of the angle grid [rad]
    # spectrograph sampling
    dtheta_deg: float = 0.05
    theta_max_deg: float = 13.0
    nphi: int = 180
    uniform_n: int = 181
    hex_samples: int = 160
    mirror_fields: bool = True
    # loss attribution
    stages: dict = field(default_factory=dict)   # pre / obs / cam -> surface ids
    groups: dict = field(default_factory=dict)   # name -> surface ids
    layout: dict = field(default_factory=dict)

    @property
    def th_o(self):
        """Lenslet top-hat half-width (apothem) in fibre-output angle [rad]:
        input half-aperture / micro-lens focal length, f = d_pup * F_tel."""
        return (self.lenslet_d_in / 2) / (self.lenslet_d_pup * self.f_tel)

    def system(self):
        return System(self.prescription, self.first_surface, self.grating_sign)

    def group_of(self):
        return {sid: g for g, ids in self.groups.items() for sid in ids}

    def unique_fields(self):
        """Fields actually traced: with mirror_fields, +x fields are traced as
        their -x mirror (identical transmission for a system symmetric in x)."""
        if not self.mirror_fields:
            return list(self.fields)
        out = []
        for fx, fy in self.fields:
            f = (-abs(fx), fy)
            if f not in out:
                out.append(f)
        return out


def load_config(path):
    path = Path(path)
    with open(path, "rb") as fh:
        c = tomllib.load(fh)
    sp, fb, sm = c["spectrograph"], c["fibre"], c.get("sampling", {})
    return Config(
        name=c.get("name", path.stem),
        prescription=(path.parent / sp["prescription"]).resolve(),
        first_surface=sp.get("first_surface", 3),
        grating_sign=sp.get("grating_sign", 1),
        wavelengths=np.array(sp["wavelengths_um"], float),
        fields=[tuple(f) for f in sp["fields_mm"]],
        field_weights=list(sp["field_weights"]),
        pupil=SlitPupil(**sp["pupil"]),
        f_tel=fb["f_tel"], eps_tel=fb["eps_tel"],
        lenslet_d_in=fb["lenslet_d_in_um"], lenslet_d_pup=fb["lenslet_d_pup_um"],
        frd_fwhm_deg=list(fb["frd_fwhm_deg"]), frd_nominal=fb.get("frd_nominal", 1.0),
        grid_n=fb.get("grid_n", 4096), grid_half=fb.get("grid_half_rad", 0.40),
        dtheta_deg=sm.get("dtheta_deg", 0.05), theta_max_deg=sm.get("theta_max_deg", 13.0),
        nphi=sm.get("nphi", 180), uniform_n=sm.get("uniform_n", 181),
        hex_samples=sm.get("hex_samples", 160), mirror_fields=sm.get("mirror_fields", True),
        stages={k: list(v) for k, v in c["stages"].items()},
        groups={k: list(v) for k, v in c["groups"].items()},
        layout=c.get("layout", {}))
