"""
Reader for the Zemax OpticStudio "System/Prescription Data" text export
(Analyze > Reports > Prescription Data, all sections, saved as .txt).

Sections used
  * SURFACE DATA SUMMARY      type, radius, glass (MIRROR flag), conic
  * SURFACE DATA DETAIL       aperture (circular / rectangular / rectangular
                              obscuration / floating, with decenters), even-
                              asphere coefficients, grating lines/um and order
  * INDEX OF REFRACTION DATA  index of the medium AFTER each surface, per
                              system wavelength
  * GLOBAL VERTEX COORDINATES rotation matrix and vertex of every surface
                              frame (relative to the global reference surface)

Because the global frames are read directly, coordinate breaks, tilts and
decenters before/after surfaces never have to be interpreted: every surface is
placed exactly where OpticStudio places it. The export must be made in the
configuration of interest (multi-configuration files: one export per
configuration).
"""
import re
import numpy as np

__all__ = ["parse_prescription"]


def _num(s):
    s = s.strip()
    if s in ("Infinity", "-Infinity"):
        return np.inf
    return float(s)


def parse_prescription(path):
    """Parse a prescription export. Returns {surface id: dict}; the image
    surface has id -999, the object 0, the stop 2 when labelled STO.

    Keys per surface: type, R, mirror, conic, ap_type and aperture sizes /
    decenters, asph {power: coefficient}, lpum and order (gratings), n (index
    after the surface at each wavelength), Rg (3x3 rotation, local -> global)
    and og (vertex, global)."""
    txt = open(path, encoding="utf-8", errors="replace").read().replace("\r", "")
    lines = txt.split("\n")

    def block(start, stop):
        i0 = next(i for i, l in enumerate(lines) if l.startswith(start))
        i1 = next(i for i, l in enumerate(lines[i0 + 1:], i0 + 1) if l.startswith(stop))
        return lines[i0 + 1:i1]

    surfs = {}
    # ---- summary
    for l in block("SURFACE DATA SUMMARY", "SURFACE DATA DETAIL"):
        f = [c.strip() for c in l.split("\t")]
        if len(f) < 9 or f[0] in ("Surf", ""):
            continue
        tag = f[0]
        sid = {"OBJ": 0, "STO": 2, "IMA": -999}.get(tag)
        if sid is None:
            if not re.match(r"^\d+$", tag):
                continue
            sid = int(tag)
        typ = f[1]
        s = dict(id=sid, type=typ, comment=f[-1] if len(f) > 9 else "")
        if typ != "COORDBRK":
            s["R"] = _num(f[2])
            s["mirror"] = f[4] == "MIRROR"
            s["conic"] = _num(f[8]) if f[8] not in ("-", "") else 0.0
        surfs[sid] = s
    # ---- detail
    cur = None
    for l in block("SURFACE DATA DETAIL", "EDGE THICKNESS DATA"):
        m = re.match(r"Surface\s+(\S+)\s+(\S+)", l)
        if m:
            tag = m.group(1)
            cur = {"OBJ": 0, "STO": 2, "IMA": -999}.get(tag)
            cur = int(tag) if cur is None else cur
            surfs[cur].setdefault("asph", {})
            continue
        if cur is None or ":" not in l:
            continue
        k, v = [t.strip() for t in l.split(":", 1)]
        s = surfs[cur]
        if k == "Aperture":
            s["ap_type"] = v.split(",")[0]
        elif k in ("Minimum Radius", "Maximum Radius", "X Half Width", "Y Half Width",
                   "X- Decenter", "Y- Decenter"):
            s[k] = _num(v)
        elif k.startswith("Coefficient on r^"):
            p = int(k.split("^")[1])
            s["asph"][p] = _num(v)
        elif k == "Lines / µm":
            s["lpum"] = _num(v)
        elif k == "Diffraction Order":
            s["order"] = _num(v)
    # ---- index of the medium after each surface
    for l in block("INDEX OF REFRACTION DATA", "THERMAL COEFFICIENT"):
        f = [c.strip() for c in l.split("\t") if c.strip()]
        if not f or not re.match(r"^\d+$", f[0]):
            continue
        sid = int(f[0])
        vals = [float(x) for x in f if re.match(r"^\d\.\d{6,}$", x)]
        if sid in surfs:
            surfs[sid]["n"] = np.array(vals[-5:])
        elif sid > max(k for k in surfs if k is not None):
            surfs[-999]["n"] = np.array(vals[-5:])
    # ---- global frames
    gl = block("GLOBAL VERTEX COORDINATES", "GLOBAL SURFACE CENTER")
    i = 0
    nmax = max(surfs)
    while i < len(gl):
        f = [c.strip() for c in gl[i].split("\t") if c.strip()]
        if f and re.match(r"^\d+$", f[0]):
            sid = int(f[0])
            r1 = [float(x) for x in f[1:5]]
            f2 = [float(c) for c in gl[i + 1].split("\t") if c.strip()][:4]
            f3 = [float(c) for c in gl[i + 2].split("\t") if c.strip()][:4]
            Rm = np.array([r1[:3], f2[:3], f3[:3]])
            o = np.array([r1[3], f2[3], f3[3]])
            key = sid if sid in surfs else -999      # last one is the image
            if sid > nmax:
                key = -999
            surfs[key]["Rg"] = Rm
            surfs[key]["og"] = o
            i += 3
        else:
            i += 1
    return surfs




def parse_fnumber_data(path):
    """OpticStudio 'F/# DATA' table of the export: {(fx, fy, wl): (tan, sag)}.
    (Working F/# per field and wavelength, vignetting factors included,
    surface apertures ignored.)"""
    lines = open(path, encoding="utf-8", errors="replace").read().replace("\r", "").split("\n")
    i = next(i for i, l in enumerate(lines) if l.startswith("F/# DATA"))
    out, wls = {}, []
    for l in lines[i + 1:]:
        if l.startswith("CARDINAL POINTS"):
            break
        if "Wavelength:" in l:
            wls = [float(x) for x in re.findall(r"\d+\.\d+", l)]
            continue
        m = re.match(r"\s*\d+\s+(-?[\d.]+),\s*(-?[\d.]+)\s*mm:\s*(.*)", l)
        if m and wls:
            fx, fy = float(m.group(1)), float(m.group(2))
            v = [float(x) for x in m.group(3).split()]
            for k, w in enumerate(wls):
                out[(fx, fy, w)] = (v[2 * k], v[2 * k + 1])
    return out
