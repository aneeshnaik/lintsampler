import pytest
from scipy.stats import norm
from lintsampler import DensityTree


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_refinement(capfd, verbose):
    """Test print statements with verbose mode on/off"""
    tree = DensityTree(mins=-5, maxs=5, pdf=norm.pdf, min_openings=3)
    tree.refine_by_error(1e-1, verbose=verbose)
    out, err = capfd.readouterr()
    if verbose:
        assert out[:8] == "Pre-loop"
    else:
        assert out == ""