"""
fibrechain: lenslet-to-detector throughput of a fibre-fed spectrograph.

The fibre output angular distribution (lenslet top hat, telescope-pupil
kernel, fibre FRD, azimuthal scrambling) is weighted by the transmission of
the spectrograph in fibre-output-angle space, obtained from an independent
real ray trace of the OpticStudio prescription export. Every lost ray is
attributed to the surface that stops it.
"""
from .zemax import parse_prescription, parse_fnumber_data
from .tracer import System
from .pupil import SlitPupil
from .config import Config, load_config
from .fibre import AngleGrid, fibre_output
from .maps import polar_grid, transmission_maps
from .budget import stage_transmissions, ring_weights, energy_budget, summarise
from .checks import uniform_rays, hexagon_samples, direct_trace, working_fnumber
from .layout import UnfoldedLayout
from .pipeline import run, validate

__version__ = "1.1.0"
