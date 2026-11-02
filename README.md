# fibrechain

**Lenslet-to-detector throughput of a fibre-fed spectrograph, from the fibre
angular distribution and an independent real ray trace of the spectrograph.**

---

## Why

The geometric vignetting that OpticStudio reports for a spectrograph is computed
with rays drawn **uniformly over the object NA**. A fibre does not deliver such a
beam: its output is the lenslet top hat, blurred by the telescope pupil and by
the fibre FRD, and azimuthally scrambled. When the main loss is a fixed cone of
directions — e.g. the central obscuration of a Schmidt camera — the throughput
depends on that distribution. `fibrechain` computes

```
O  = lenslet top hat (100 %)
I  = O ⊗ K_geo                     telescope pupil seen from the fibre (geometric FRD)
P  = < I ⊗ K_FRD >_φ               fibre FRD + azimuthal scrambling
F  = Σ_rings  ΔEE(θ_i) · t(θ_i)    t = spectrograph transmission vs fibre-output angle
```

where `t(θ)` comes from tracing the real prescription, and every lost ray is
attributed to the surface that stops it.

## Independent of OpticStudio

OpticStudio is used only to **export** the *Prescription Data* report of one
configuration. Surface placement (read from the global vertex frames), ray–surface
intersection, refraction, reflection, VPH diffraction, aperture tests and loss
attribution are done by `fibrechain`. The OpticStudio numbers of the same export
(uniform-ray vignetting, *F/# DATA* table) are used as checks only.

## Install

```bash
pip install -e ".[figures]"            # Python ≥ 3.11, NumPy, SciPy, Matplotlib
```

## Run

1. In OpticStudio: *Analyze → Reports → Prescription Data*, all sections, for the
   configuration of interest; save as text next to your configuration file.
2. Write a `config.toml` (layout below).
3. Run:

```bash
fibrechain run config.toml -o build
```

writes to `build/`:

| File | Content |
|---|---|
| `fibre_output.npz` | encircled energy EE(θ) of the fibre output, with / without `K_geo`, all FRD levels; maps |
| `trans_maps.npz` | id of the stopping surface for every fibre-output direction (field × λ × θ × φ) |
| `stage_transmissions.npz` | azimuthal transmission per ring: before camera / after obscuration / detector |
| `budget.csv`, `budget_summary.csv` | throughput and losses per kernel × FRD × field × λ; on-axis and slit-weighted |
| `uniform_rays.npy` | OpticStudio-like uniform-ray reference, losses by contributor |
| `hexagon_direct.npy` | direct ray-by-ray trace of the scrambled lenslet hexagon |
| `fnumber_check.npy` | working F/#: OpticStudio vs fibrechain |
| `validation.json` | all consistency checks |

Check `validation.json` before using the budget.

## Configuration

Paths are relative to the configuration file. Values in `<...>` are yours.

```toml
name = "<instrument / configuration>"

[spectrograph]
prescription   = "<prescription_export>.txt"   # OpticStudio Prescription Data export
first_surface  = <int>                         # slit surface (fibre exit side)
grating_sign   = 1                             # VPH sign convention (+1 or -1)
wavelengths_um = [<λ1>, <λ2>, ...]             # in the order of the export
fields_mm      = [[<fx>, <fy>], ...]
field_weights  = [<w1>, ...]

[spectrograph.pupil]          # "object NA" aperture, curved slit about the stop
r_obj  = <mm>                 # slit radius of curvature
z_stop = <mm>                 # stop position relative to the slit vertex
r_stop = <mm>                 # stop radius giving the object NA
vdy    = <float>              # pupil decentre (vignetting factor VDY)

[fibre]
f_tel            = <float>    # telescope focal ratio at the lenslet
eps_tel          = <float>    # telescope central obscuration (linear)
lenslet_d_in_um  = <um>       # lenslet input aperture, flat to flat
lenslet_d_pup_um = <um>       # telescope pupil image on the fibre core
frd_fwhm_deg     = [0.0, 0.5, 1.0, 2.0]
frd_nominal      = 1.0

[sampling]                    # optional, defaults shown
dtheta_deg    = 0.05
theta_max_deg = 13.0
nphi          = 180
uniform_n     = 181
hex_samples   = 160
mirror_fields = true          # +x fields traced as their -x mirror

[stages]                      # loss stages, by first stopping surface (-999 = detector)
pre = [<surface ids before the obscuration>]
obs = [<obscuration surface id>]
cam = [<camera surface ids>, -999]

[groups]                      # contributor names used in the reports
"<name>" = [<surface ids>]

[layout]                      # optional: unfolded drawing (UnfoldedLayout)
keep = [<surface ids to draw>]
```

## Use as a library

```python
from fibrechain import load_config, fibre_output, transmission_maps, energy_budget

cfg = load_config("config.toml")
system = cfg.system()                                   # System from the export
p, d = cfg.pupil.fibre_rays(fx, fy, theta, phi)         # rays about the fibre axis
_, _, stopped_by = system.trace(p, d, wavelength_index, wavelength_um)  # -1 = detector
```

## Repository

```
src/fibrechain/
  zemax.py      prescription export reader (+ F/# table)
  optics.py     sag, intersection, apertures
  tracer.py     System: vectorised sequential trace, loss attribution
  pupil.py      SlitPupil: curved slit, stop, VDY; uniform-pupil and fibre-output rays
  fibre.py      fibre output: top hat, K_geo, K_FRD, scrambling, EE
  maps.py       transmission maps in fibre-output-angle space
  budget.py     stage transmissions, ring weights, energy budget
  checks.py     uniform rays, working F/#, direct trace of a distribution
  layout.py     linearised drawing of the traced system
  config.py, pipeline.py, cli.py
```

## Notes

- No prescription or instrument data is distributed with this repository.
- Licence: to be chosen.
- Author: Thomas Pichon.
