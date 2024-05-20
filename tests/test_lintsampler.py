import pytest
import numpy as np
from scipy.stats import norm, multivariate_normal
import sys
import os
sys.path.insert(0, os.path.abspath("src"))
from lintsampler import LintSampler

X_EDGES = np.linspace(-10, 10, 65)
Y_EDGES = np.linspace(-5, 5, 33)
XY_GRID = np.stack(np.meshgrid(X_EDGES, Y_EDGES, indexing='ij'), axis=-1)
X1_EDGES = np.linspace(-10, 0, 65)
X2_EDGES = np.linspace(0, 10, 65)
Y1_EDGES = np.linspace(-5, 0, 33)
Y2_EDGES = np.linspace(0, 5, 33)
XY1_GRID = np.stack(np.meshgrid(X1_EDGES, Y1_EDGES, indexing='ij'), axis=-1)
XY2_GRID = np.stack(np.meshgrid(X2_EDGES, Y2_EDGES, indexing='ij'), axis=-1)
BAD_CELLS = [
    np.array([1.0, 1.5, 1.5, 2.0]),
    np.ones(10),
    np.array([9.0, 8.0, 10.0, 12.0]),
]
CELLS_1D = [
    X_EDGES, 
    tuple(X_EDGES),
    list(X_EDGES),
    [X1_EDGES, X2_EDGES],
    [tuple(X1_EDGES), tuple(X2_EDGES)],
    [list(X1_EDGES), list(X2_EDGES)],
]
CELLS_2D = [
    (X_EDGES, Y_EDGES),
    (tuple(X_EDGES), tuple(Y_EDGES)),
    (list(X_EDGES), list(Y_EDGES)),
    XY_GRID,
    [(X1_EDGES, Y1_EDGES), (X2_EDGES, Y2_EDGES)],
    [(tuple(X1_EDGES), tuple(Y1_EDGES)), (tuple(X2_EDGES), tuple(Y2_EDGES))],
    [(list(X1_EDGES), list(Y1_EDGES)), (list(X2_EDGES), list(Y2_EDGES))],
    [XY1_GRID, XY2_GRID],
]


def neg_pdf_1D(x):
    return -norm.pdf(x)


def neg_pdf_2D(x):
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    return -dist.pdf(x)


## INPUT CHECKING ##############################################################


def test_bad_seed():
    """Test that providing a nonsensical 'seed' raises an error."""
    with pytest.raises(TypeError):
        LintSampler(norm.pdf, X_EDGES, vectorizedpdf=True, seed=42.5).sample()


def test_nonint_N_samples():
    """Test that providing a non-integer N_samples raises an error."""
    sampler = LintSampler(norm.pdf, X_EDGES, vectorizedpdf=True, seed=42)
    with pytest.raises(TypeError):
        sampler.sample(N_samples=10.0)


@pytest.mark.parametrize("N_samples", [0, -5])
def test_bad_N_samples(N_samples):
    """Test that providing zero, or negative N_samples raises error."""
    sampler = LintSampler(norm.pdf, X_EDGES, vectorizedpdf=True, seed=42)
    with pytest.raises(ValueError):
        sampler.sample(N_samples=N_samples)


@pytest.mark.parametrize("cells", CELLS_1D)
def test_1D_f_negative(cells):
    """Test error raised if f negative anywhere"""
    sampler = LintSampler(neg_pdf_1D, cells, vectorizedpdf=True, seed=42)
    with pytest.raises(ValueError):
        sampler.sample()


@pytest.mark.parametrize("cells", CELLS_2D)
def test_kD_f_negative(cells):
    """Test error raised if f negative anywhere"""
    sampler = LintSampler(neg_pdf_2D, cells, vectorizedpdf=True, seed=42)
    with pytest.raises(ValueError):
        sampler.sample()


@pytest.mark.parametrize("cells", BAD_CELLS)
def test_1D_edges_non_monotonic(cells):
    """Test error raised if f negative anywhere"""
    sampler = LintSampler(norm.pdf, cells, vectorizedpdf=True, seed=42)
    with pytest.raises(ValueError):
        sampler.sample()


## OUTPUT SHAPES ###############################################################


@pytest.mark.parametrize("cells", CELLS_1D)
@pytest.mark.parametrize("N_samples", [None, 16])
@pytest.mark.parametrize("vectorizedpdf", [True, False])
@pytest.mark.parametrize("qmc", [True, False])
def test_1D_output_shapes(cells, N_samples, vectorizedpdf, qmc):
    """Single sample in 1D -> float, multiple samples -> 1D array"""
    sampler = LintSampler(norm.pdf, cells, vectorizedpdf=vectorizedpdf, qmc=qmc, seed=42)
    x = sampler.sample(N_samples=N_samples)
    if N_samples is None:
        assert isinstance(x, float)
    else:
        assert x.shape == (N_samples,)


@pytest.mark.parametrize("cells", CELLS_2D)
@pytest.mark.parametrize("N_samples", [None, 16])
@pytest.mark.parametrize("vectorizedpdf", [True, False])
@pytest.mark.parametrize("qmc", [True, False])
def test_kD_output_shapes(cells, N_samples, vectorizedpdf, qmc):
    """Single sample in kD -> k-vector, multiple samples -> 2D array (N, k)"""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    sampler = LintSampler(dist.pdf, cells, vectorizedpdf=vectorizedpdf, qmc=qmc, seed=42)
    x = sampler.sample(N_samples=N_samples)
    if N_samples is None:
        assert x.shape == (2,)
    else:
        assert x.shape == (N_samples, 2)


## OLD STUFF COPIED FROM TEST_GRIDSAMPLE #######################################

# @pytest.fixture
# def y_edges():
#     return np.linspace(100, 200, 65)

# @pytest.fixture
# def f_1D():
#     rng = np.random.default_rng(42)
#     return rng.uniform(size=33)

# @pytest.fixture
# def f_2D():
#     rng = np.random.default_rng(42)
#     return rng.uniform(size=(33, 65))


# ## INPUT CHECKING ##############################################################


# @pytest.mark.parametrize("bad_edges", [np.array([1.0, 1.5, 1.5, 2.0]), np.ones(10), np.array([9.0, 8.0, 10.0, 12.0])])
# def test_edges_non_monotonic(bad_edges):
#     """Test that edge array not monotonically increasing raises error."""
#     rng = np.random.default_rng(42)
#     f = rng.uniform(size=len(bad_edges))
#     with pytest.raises(ValueError):
#         gridsample(bad_edges, f=f, seed=42)


# def test_edges_2D():
#     """Test that 2D edge array raises error."""
#     rng = np.random.default_rng(42)
#     edges = np.array([[1.0, 1.5, 2.0], [1.0, 1.5, 2.0]])
#     f = rng.uniform(size=len(edges))
#     with pytest.raises(TypeError):
#         gridsample(edges, f=f, seed=42)


# ## DETERMINISM #################################################################


# def test_same_int_seed(x_edges, f_1D):
#     """Test same RNG seed (specified as integer) produces same results."""
#     x1 = gridsample(x_edges, f=f_1D, seed=42)
#     x2 = gridsample(x_edges, f=f_1D, seed=42)
#     assert x1==x2
#     x1 = gridsample(x_edges, f=f_1D, N_samples=10, seed=42)
#     x2 = gridsample(x_edges, f=f_1D, N_samples=10, seed=42)
#     assert np.all(x1==x2)


# def test_same_rng_seed(x_edges, f_1D):
#     """Test same RNG seed (specified as np RNG) produces same results."""
#     x1 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42))
#     x2 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42))
#     assert x1==x2
#     x1 = gridsample(x_edges, f=f_1D, N_samples=10, seed=np.random.default_rng(42))
#     x2 = gridsample(x_edges, f=f_1D, N_samples=10, seed=np.random.default_rng(42))
#     assert np.all(x1==x2)


# ## OUTPUT VALUES ###############################################################


# def test_1D_gaussian():
#     """Test samples from a 1D gaussian have correct mean and width"""
#     mu_true = 30.0
#     sig_true = 1.8
#     x_gaussian = np.linspace(20, 40, 512)
#     f_gaussian = norm.pdf(x_gaussian, loc=mu_true, scale=sig_true)
    
#     x = gridsample(x_gaussian, f=f_gaussian, N_samples=100000, seed=42)
#     mu = np.round(np.mean(x), decimals=0)
#     sig = np.round(np.std(x), decimals=1)
    
#     assert (mu, sig) == (mu_true, sig_true)


# def test_kd_gaussian():
#     """Test samples from a kD gaussian have correct mean and covariances"""
#     mu_true = np.array([3.0, -0.5])
#     cov_true = np.array([
#         [ 1.0,  -0.5],
#         [-0.5,  1.5],
#     ])
#     edges = np.linspace(-10, 10, 513)
#     grid = np.stack(np.meshgrid(edges, edges, indexing='ij'), axis=-1)
#     f = multivariate_normal.pdf(grid, mean=mu_true, cov=cov_true)
#     x = gridsample(edges, edges, f=f, N_samples=1000000)

#     mu = np.round(np.mean(x, axis=0), decimals=1)
#     cov = np.round(np.cov(x.T), decimals=1)
#     assert np.all(mu == mu_true) and np.all(cov == cov_true)










## OLD STUFF COPIED FROM TEST_FREESAMPLE #######################################

# @pytest.fixture
# def x0_single_3d():
#     return np.array([10, 100, 1000])

# @pytest.fixture
# def x1_single_3d():
#     return np.array([20, 200, 2000])

# @pytest.fixture
# def f_single_3d():
#     rng = np.random.default_rng(42)
#     return tuple(rng.uniform(size=8))

# @pytest.fixture
# def x0_batch_1d():
#     return np.arange(0, 10)[:, None]

# @pytest.fixture
# def x1_batch_1d():
#     return np.arange(1, 11)[:, None]

# @pytest.fixture
# def f_batch_1d():
#     rng = np.random.default_rng(42)
#     f = rng.uniform(size=11)
#     return (f[:-1], f[1:])

# @pytest.fixture
# def x0_batch_2d():
#     return np.array([[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]])

# @pytest.fixture
# def x1_batch_2d():
#     return np.array([[20, 200], [30, 300], [40, 400], [50, 500], [60, 600]])

# @pytest.fixture
# def f_batch_2d():
#     rng = np.random.default_rng(42)
#     return tuple(rng.uniform(size=5) for i in range(4))
    

# ## INPUT CHECKING ##############################################################

# def test_f_bad_len(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Test error raised if wrong number of densities provided"""
#     f00, f01, f10, f11 = f_batch_2d
#     with pytest.raises(ValueError):
#         sample(x0_batch_2d, x1_batch_2d, f00, f01, f10, seed=42)


# def test_x0x1_mismatch(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Test error raised if shapes of x0 and x1 are mismatched"""
#     x1 = np.vstack((x1_batch_2d, np.array([70, 700])))
#     with pytest.raises(ValueError):
#         sample(x0_batch_2d, x1, *f_batch_2d, seed=42)


# def test_x0f_mismatch(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Test error raised if batch sizes of x and f are mismatched"""
#     x0 = np.vstack((x0_batch_2d, np.array([60, 600])))
#     x1 = np.vstack((x1_batch_2d, np.array([70, 700])))
#     with pytest.raises(ValueError):
#         sample(x0, x1, *f_batch_2d, seed=42)


# def test_x0_equal_x1(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Test error raised if x0 == x1 anywhere."""
#     x0_batch_2d[2, 0] = x1_batch_2d[2, 0]
#     with pytest.raises(ValueError):
#         sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)


# def test_x0_gtr_x1(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Test error raised if x0 > x1 anywhere."""
#     x0_batch_2d[2, 0] = x1_batch_2d[2, 0] + 10
#     with pytest.raises(ValueError):
#         sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)


# ## DETERMINISM #################################################################


# def test_same_int_seed():
#     """Test same RNG seed (specified as integer) produces same results."""
#     x1 = sample(10, 20, 0.1, 0.3, seed=42)
#     x2 = sample(10, 20, 0.1, 0.3, seed=42)
#     assert x1==x2
#     x1 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
#     x2 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
#     assert np.all(x1==x2)


# def test_same_rng_seed():
#     """Test same RNG seed (specified as np RNG) produces same results."""
#     x1 = sample(10, 20, 0.1, 0.3, seed=np.random.default_rng(42))
#     x2 = sample(10, 20, 0.1, 0.3, seed=np.random.default_rng(42))
#     assert x1==x2
#     x1 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=np.random.default_rng(42))
#     x2 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=np.random.default_rng(42))
#     assert np.all(x1==x2)


# ## OUTPUT SHAPES ###############################################################


# def test_shape_single_sample_single_1D():
#     """Drawing single sample from single 1D cell -> float."""
#     x = sample(10, 20, 0.1, 0.3, seed=42)
#     assert isinstance(x, float)


# def test_shape_multi_sample_single_1D():
#     """Drawing multi samples from single 1D cell -> 1D array (N_samples,)."""
#     x = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
#     assert x.shape == (10,)


# def test_shape_single_sample_single_kD(x0_single_3d, x1_single_3d, f_single_3d):
#     """Drawing single sample from single kD cell -> 1D array (k,)."""
#     x = sample(x0_single_3d, x1_single_3d, *f_single_3d, seed=42)
#     assert x.shape == (3,)


# def test_shape_multi_sample_single_kD(x0_single_3d, x1_single_3d, f_single_3d):
#     """Drawing multi samples from single kD cell -> 2D array (N_samples, k)."""
#     x = sample(x0_single_3d, x1_single_3d, *f_single_3d, N_samples=10, seed=42)
#     assert x.shape == (10, 3)


# def test_shape_single_sample_multi_kD(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Drawing single sample from multi kD cells -> 1D array (k,)."""
#     x = sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)
#     assert x.shape == (2,)
    

# def test_shape_single_sample_multi_1D(x0_batch_1d, x1_batch_1d, f_batch_1d):
#     """Drawing single sample from multi 1D cells -> float."""
#     x = sample(x0_batch_1d, x1_batch_1d, *f_batch_1d, seed=42)
#     assert isinstance(x, float)


# def test_shape_multi_sample_multi_kD(x0_batch_2d, x1_batch_2d, f_batch_2d):
#     """Drawing multi samples from multi kD cells -> 2D array (N_samples, k)."""
#     x = sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, N_samples=10, seed=42)
#     assert x.shape == (10, 2)


# def test_shape_multi_sample_multi_1D(x0_batch_1d, x1_batch_1d, f_batch_1d):
#     """Drawing multi samples from multi 1D cells -> 1D array (N_samples,)."""
#     x = sample(x0_batch_1d, x1_batch_1d, *f_batch_1d, N_samples=10, seed=42)
#     assert x.shape == (10,)


# ## OUTPUT VALUES ###############################################################


# def test_1D_uniform_single():
#     """Test that single uniform sample is in appropriate range."""    
#     x = sample(15, 25, 0.5, 0.5)
#     assert (x < 25) and (x > 15)


# def test_1D_uniform():
#     """Test that batch of uniform samples gives flat histogram."""    
#     x = sample(15, 25, 0.5, 0.5, N_samples=1000000)
#     p = np.histogram(x, np.linspace(15, 25, 11), density=True)[0]
#     assert np.all(np.round(p, decimals=1) == 0.1)
    

# def test_1D_GMM():
#     """Test samples from 1D 2-component GMM have correct means and widths."""
#     # true params
#     mua_true = -10.0
#     mub_true = 10.0
#     siga_true = 1.1
#     sigb_true = 0.9
#     wa_true = 0.4
#     wb_true = 0.6

#     # setup 2 PDFs
#     dista = norm(loc=mua_true, scale=siga_true)
#     distb = norm(loc=mub_true, scale=sigb_true)

#     # hypercells: two separate 1D grids
#     N_grid = 256
#     ea = np.linspace(-16, -4, N_grid + 1)
#     eb = np.linspace(4, 16, N_grid + 1)
#     fa = wa_true * dista.pdf(ea) + wb_true * distb.pdf(ea)
#     fb = wa_true * dista.pdf(eb) + wb_true * distb.pdf(eb)
#     x0 = np.hstack((ea[:-1], eb[:-1]))[:, None]
#     x1 = np.hstack((ea[1:], eb[1:]))[:, None]
#     f0 = np.hstack((fa[:-1], fb[:-1]))
#     f1 = np.hstack((fa[1:], fb[1:]))

#     # draw samples
#     x = sample(x0, x1, f0, f1, N_samples=1000000, seed=42)

#     # find particles corresponding to each Gaussian, get sample stats
#     ma = (x < 0)
#     mb = ~ma    
#     mua = np.round(np.mean(x[ma]), decimals=1)
#     mub = np.round(np.mean(x[mb]), decimals=1)
#     siga = np.round(np.std(x[ma]), decimals=1)
#     sigb = np.round(np.std(x[mb]), decimals=1)
#     wa = np.round(ma.sum() / len(ma), decimals=1)
#     wb = 1 - wa
    
#     # check sample statistics match true params
#     assert mua == mua_true
#     assert mub == mub_true
#     assert siga == siga_true
#     assert sigb == sigb_true
#     assert wa == wa_true
#     assert wb == wb_true


# def test_2D_GMM():
#     """Test samples from 2D 2-component GMM have correct means and covs."""
#     # true params
#     mua_true = np.array([-5, -5])
#     mub_true = np.array([5, 5])
#     cova_true = np.array([
#         [ 1.0,  -0.5],
#         [-0.5,  1.5],
#     ])
#     covb_true = np.array([
#         [ 1.5,  0.5],
#         [ 0.5,  1.0],
#     ])
#     wa_true = 0.3
#     wb_true = 0.7

#     # setup 2 PDFs
#     dista = multivariate_normal(mean=mua_true, cov=cova_true)
#     distb = multivariate_normal(mean=mub_true, cov=covb_true)

#     # hypercells: two separate 256x256 grids
#     N_grid = 256
#     ea = np.linspace(-10, 0, N_grid + 1)
#     ga = np.stack(np.meshgrid(ea, ea, indexing='ij'), axis=-1)
#     eb = np.linspace(0, 10, N_grid + 1)
#     gb = np.stack(np.meshgrid(eb, eb, indexing='ij'), axis=-1)
#     fa = wa_true * dista.pdf(ga) + wb_true * distb.pdf(ga)
#     fb = wa_true * dista.pdf(gb) + wb_true * distb.pdf(gb)
#     x0 = np.vstack([ga[:-1, :-1].reshape((N_grid**2, 2)), gb[:-1, :-1].reshape((N_grid**2, 2))])
#     x1 = np.vstack([ga[1:, 1:].reshape((N_grid**2, 2)), gb[1:, 1:].reshape((N_grid**2, 2))])
#     f00 = np.hstack([fa[:-1, :-1].flatten(), fb[:-1, :-1].flatten()])
#     f01 = np.hstack([fa[:-1, 1:].flatten(), fb[:-1, 1:].flatten()])
#     f10 = np.hstack([fa[1:, :-1].flatten(), fb[1:, :-1].flatten()])
#     f11 = np.hstack([fa[1:, 1:].flatten(), fb[1:, 1:].flatten()])
    
#     # draw samples
#     x = sample(x0, x1, f00, f01, f10, f11, N_samples=2000000, seed=42)
    
#     # find particles corresponding to each Gaussian, get sample stats
#     ma = (x[:, 0] < 0) & (x[:, 1] < 0)
#     mb = ~ma
#     mua = np.round(np.mean(x[ma], axis=0), decimals=1)
#     cova = np.round(np.cov(x[ma].T), decimals=1)
#     mub = np.round(np.mean(x[mb], axis=0), decimals=1)
#     covb = np.round(np.cov(x[mb].T), decimals=1)
#     wa = np.round(ma.sum() / len(ma), decimals=1)
#     wb = 1 - wa
    
#     # check sample statistics match true params
#     assert np.all(mua == mua_true)
#     assert np.all(mub == mub_true)
#     assert np.all(cova == cova_true)
#     assert np.all(covb == covb_true)
#     assert wa == wa_true
#     assert wb == wb_true
