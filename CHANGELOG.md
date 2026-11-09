# Changelog

## 1.1.0 — 2026-11-09
- Ray–surface intersection: rays lost by Newton's iteration on strongly curved
  faces (start point outside the domain of the sag) are re-solved by bracketing
  the first crossing along the ray. Steep rays on deep conic faces were
  wrongly counted as stopped by them; rays that converged are unchanged.
- SZERNSAG surfaces traced with their Zernike Standard Sag terms (were traced
  with their base sag only).
- Slit averages: field weights of the configuration (equal weights give the
  plain slit mean); README updated.

## 1.0.0 — 2026-11-02
First release.
- OpticStudio *Prescription Data* reader (surfaces, apertures, indices, global
  frames, F/# table) and vectorised sequential real ray tracer (standard,
  even asphere, SZERNSAG as plane, DGRATING/VPH, mirrors) with attribution of
  every lost ray to the surface that stops it.
- Fibre output model: hexagonal lenslet top hat, geometric FRD kernel of the
  telescope pupil, Gaussian fibre FRD, azimuthal scrambling, encircled energy.
- Transmission maps in fibre-output-angle space, energy budget, OpticStudio-like
  uniform-ray reference, working F/#, direct trace of a distribution.
