import numpy as np
from lintsampler import DensityStructure


def test_densitystructure():
    """Test abstract density structure returns None for all methods/attributes.
    
    This is a slightly hacky solution to the problem that the abstractmethods
    of abstract base classes don't get covered by pytest tests because the
    class never gets instantiated. The solution is to trick it into thinking
    that there are no abstractmethods, so that the class can be instantiated,
    then running the various (no longer abstract) methods.
    """
    DensityStructure.__abstractmethods__ = set()
    ds = DensityStructure()
    assert ds.total_mass is None
    assert ds.mins is None
    assert ds.maxs is None
    assert ds.dim is None
    assert ds.choose_cells(np.array([0.5, 0.7, 0.9])) is None
