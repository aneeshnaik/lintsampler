"""
lintsampler

Efficient random sampling via linear interpolation.

See README file and online documentation (https://lintsampler.readthedocs.io)
for further details and usage instructions. 
"""
from .lintsampler import LintSampler
from .density_structure import DensityStructure
from .density_grid import DensityGrid
from importlib.metadata import version

__version__ = version("lintsampler")

__all__ = ["LintSampler", "DensityStructure", "DensityGrid"]
