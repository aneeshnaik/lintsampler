from .lintsampler import LintSampler
from importlib.metadata import version

#try:
__version__ = version("lintsampler")
#except PackageNotFoundError:
#    __version__ = "unknown version"

__all__ = ["LintSampler"]
