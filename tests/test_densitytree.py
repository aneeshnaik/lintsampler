import pytest
import numpy as np
from scipy.stats import norm, multivariate_normal
from lintsampler import DensityTree


## INPUT CHECKING: DENSITY FN ##################################################


@pytest.mark.parametrize("vectorizedpdf", [True, False])
def test_f_negative(vectorizedpdf):
    """Test error raised if f negative anywhere with tree"""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    neg_pdf = lambda x: -dist.pdf(x)
    with pytest.raises(ValueError):
        DensityTree(np.array([-10, -5]), np.array([10, 5]), pdf=neg_pdf, vectorizedpdf=vectorizedpdf)


@pytest.mark.parametrize("vectorizedpdf", [True, False])
def test_f_nonfinite(vectorizedpdf):
    """Test error raised if f non-finite anywhere with tree"""
    def nonfinite_pdf_vec(x):
        f = np.ones((x.shape[:-1]))
        f[x[..., 0] < 0] = np.nan
        return f

    def nonfinite_pdf_nonvec(x):
        if x[0] < 0:
            return np.nan
        else:
            return 1.0
    
    if vectorizedpdf:
        pdf = nonfinite_pdf_vec
    else:
        pdf = nonfinite_pdf_nonvec
    
    with pytest.raises(ValueError):
        DensityTree(np.array([-10, -5]), np.array([10, 5]), pdf=pdf, vectorizedpdf=vectorizedpdf)


@pytest.mark.parametrize("pdf,vectorizedpdf", [
        (lambda x: np.ones(2), False),
        (lambda x: np.ones((len(x), 2)), True),
])
def test_f_bad_shape(pdf, vectorizedpdf):
    """Test error raised if f returns inappropriate shape"""
    with pytest.raises(ValueError):
        DensityTree(np.array([-10, -5]), np.array([10, 5]), pdf=pdf, vectorizedpdf=vectorizedpdf)


@pytest.mark.parametrize("pdf", [True, 10, np.random.default_rng(42)])
def test_noncallable_pdf(pdf):
    """Test error raised if PDF is not a callable"""
    with pytest.raises(TypeError):
        DensityTree(np.array([-10, -5]), np.array([10, 5]), pdf=pdf)


## INPUT CHECKING: MINS/MAXS ###################################################


def test_minsmaxs_diff_lengths():
    """Test error raised if mins/maxs have different lengths"""
    with pytest.raises(ValueError):
        DensityTree(mins=[1, 2, 3, 4], maxs=[10, 20, 30], pdf=norm.pdf)


def test_minsmaxs_wrong_dim():
    """Test error raised if mins/maxs not 1D"""
    with pytest.raises(ValueError):
        DensityTree(mins=np.zeros((10, 2)), maxs=np.ones((10, 2)), pdf=norm.pdf)


def test_1D_non_monotonic():
    """Test error raised if 1D tree mins/maxs not monotonic"""
    with pytest.raises(ValueError):
        DensityTree(mins=4, maxs=3, pdf=norm.pdf)

        
def test_kD_non_monotonic():
    """Test error raised if kD tree mins/maxs not monotonic"""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    with pytest.raises(ValueError):
        DensityTree(mins=[10, 4], maxs=[20, 3], pdf=dist.pdf)


@pytest.mark.parametrize("maxs", [np.nan, np.inf])
def test_1D_tree_non_finite(maxs):
    """Test error raised if 1D tree mins/maxs not finite"""
    with pytest.raises(ValueError):
        DensityTree(mins=4, maxs=maxs, pdf=norm.pdf)


@pytest.mark.parametrize("maxs", [[20, np.nan], [20, np.inf]])
def test_kD_tree_non_finite(maxs):
    """Test error raised if kD tree mins/maxs not finite"""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    with pytest.raises(ValueError):
        DensityTree(mins=[10, 4], maxs=maxs, pdf=dist.pdf)


## INPUT CHECKING: BATCH/CACHE #################################################


def test_batch_usecache_warning():
    """Test warning if batch=True, usecache=True"""
    with pytest.warns(UserWarning):
        DensityTree(mins=3, maxs=4, pdf=norm.pdf, batch=True, usecache=True)


## TREE CONSTRUCTION ##########################################################


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("batch", [True, False])
def test_num_leaves(dim, batch):
    """Check number of tree leaves makes sense after some full openings"""
    mins = np.zeros(dim)
    maxs = np.ones(dim)
    min_openings = 3
    if dim == 1:
        pdf = norm.pdf
    else:
        pdf = multivariate_normal(np.zeros(dim), np.eye(dim)).pdf
    tree = DensityTree(
        mins=mins, maxs=maxs,
        pdf=pdf, batch=batch, usecache= not batch, min_openings=min_openings
    )
    assert len(tree.leaves) == 2**(dim * min_openings)


## GET LEAF AT POS #############################################################


@pytest.mark.parametrize("dim,pos", [
    (1, [2, 3]),
    (1, -11),
    (2, 1),
    (2, [3, 4, 4]),
    (2, [3, 5.5])
])
def test_get_leaf_bad_pos(dim, pos):
    """Test error raised if get_leaf_at_pos method gets weird or external pos"""
    if dim == 1:
        mins = -10
        maxs = 10
        pdf = norm.pdf
    elif dim == 2:
        mins = [-5, -5]
        maxs = [5, 5]
        pdf = multivariate_normal(mean=np.ones(2), cov=np.eye(2)).pdf
    tree = DensityTree(mins, maxs, pdf=pdf, vectorizedpdf=True, min_openings=1)
    with pytest.raises(ValueError):
        tree.get_leaf_at_pos(pos)


@pytest.mark.parametrize("dim", [1, 2])
def test_get_leaf_gets_correct_leaf(dim):
    """Test leaf returned by get_leaf_at_pos method contains given pos"""
    if dim == 1:
        mins = -10
        maxs = 10
        pdf = norm.pdf
        pos = 9
    elif dim == 2:
        mins = [-5, -5]
        maxs = [5, 5]
        pdf = multivariate_normal(mean=np.ones(2), cov=np.eye(2)).pdf
        pos = np.array([1.02, -3.75])
    tree = DensityTree(mins, maxs, pdf=pdf, vectorizedpdf=True, min_openings=4)
    leaf = tree.get_leaf_at_pos(pos)
    assert np.all((leaf.x <= pos) & (pos <= leaf.x + leaf.dx))


## REFINEMENT ##################################################################


def test_refinement_without_opening():
    """Test refinement fails if min_openings=0"""
    tree = DensityTree(mins=-5, maxs=5, pdf=norm.pdf, min_openings=0)
    with pytest.raises(RuntimeError):
        tree.refine(1e-1)


@pytest.mark.parametrize("verbose", [True, False])
def test_verbose_refinement(capfd, verbose):
    """Test print statements with verbose mode on/off"""
    tree = DensityTree(mins=-5, maxs=5, pdf=norm.pdf, min_openings=1)
    tree.refine(1e-1, verbose=verbose)
    out, err = capfd.readouterr()
    if verbose:
        assert out[:8] == "Pre-loop"
    else:
        assert out == ""