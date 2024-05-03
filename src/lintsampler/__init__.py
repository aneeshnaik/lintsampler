from .gridsample import gridsample
from .sample import sample
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("lintsampler")
except PackageNotFoundError:
    __version__ = "unknown version"

__all__ = ["gridsample", "sample"]
