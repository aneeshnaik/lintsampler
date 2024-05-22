"""
lintsampler

Efficient random sampling via linear interpolation.

See README file and online documentation (https://lintsampler.readthedocs.io)
for further details and usage instructions. 
"""
from .lintsampler import LintSampler
from .grid import SamplingGrid
from importlib.metadata import version, PackageNotFoundError

__version__ = version("lintsampler")

__all__ = ["LintSampler, SamplingGrid"]
