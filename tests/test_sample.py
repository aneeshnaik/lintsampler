import pytest
import numpy as np
import sys
import os
from scipy.stats import norm, multivariate_normal
sys.path.insert(0, os.path.abspath("src"))
from lintsampler import sample

# TODO repeat 1D uniform, 1D gmm and 2D gmm tests w/ default qmc
# TODO check non-power of 2 raises Sobol warning
# TODO check error raised if QMC engine dimension wrong
# TODO test error raised if provided QMC engine wrong
# TODO repeat 1D uniform, 1D gmm and 2D gmm tests w/ Halton engine 
# TODO test determinism with QMC
# TODO test this works with qmc and None N

@pytest.fixture
def x0_single_3d():
    return np.array([10, 100, 1000])

@pytest.fixture
def x1_single_3d():
    return np.array([20, 200, 2000])

@pytest.fixture
def f_single_3d():
    rng = np.random.default_rng(42)
    return tuple(rng.uniform(size=8))

@pytest.fixture
def x0_batch_1d():
    return np.arange(0, 10)[:, None]

@pytest.fixture
def x1_batch_1d():
    return np.arange(1, 11)[:, None]

@pytest.fixture
def f_batch_1d():
    rng = np.random.default_rng(42)
    f = rng.uniform(size=11)
    return (f[:-1], f[1:])

@pytest.fixture
def x0_batch_2d():
    return np.array([[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]])

@pytest.fixture
def x1_batch_2d():
    return np.array([[20, 200], [30, 300], [40, 400], [50, 500], [60, 600]])

@pytest.fixture
def f_batch_2d():
    rng = np.random.default_rng(42)
    return tuple(rng.uniform(size=5) for i in range(4))
    

## INPUT CHECKING ##############################################################


def test_bad_seed():
    """Test that providing a nonsensical 'seed' raises an error."""
    with pytest.raises(TypeError):
        sample(10, 20, 0.1, 0.3, seed=43.0)


def test_nonint_N_samples():
    """Test that providing a non-integer N_samples raises an error."""
    with pytest.raises(TypeError):
        sample(10, 20, 0.1, 0.3, N_samples=10.0, seed=42)


@pytest.mark.parametrize("N_samples", [0, -5])
def test_bad_N_samples(N_samples):
    """Test that providing a non-integer N_samples raises an error."""
    with pytest.raises(ValueError):
        sample(10, 20, 0.1, 0.3, N_samples=N_samples, seed=42)


def test_f_negative(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if f negative anywhere"""
    f00, f01, f10, f11 = f_batch_2d
    f01[3] *= -1
    with pytest.raises(ValueError):
        sample(x0_batch_2d, x1_batch_2d, f00, f01, f10, f11, seed=42)


def test_f_bad_len(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if wrong number of densities provided"""
    f00, f01, f10, f11 = f_batch_2d
    with pytest.raises(ValueError):
        sample(x0_batch_2d, x1_batch_2d, f00, f01, f10, seed=42)


def test_x0x1_mismatch(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if shapes of x0 and x1 are mismatched"""
    x1 = np.vstack((x1_batch_2d, np.array([70, 700])))
    with pytest.raises(ValueError):
        sample(x0_batch_2d, x1, *f_batch_2d, seed=42)


def test_x0f_mismatch(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if batch sizes of x and f are mismatched"""
    x0 = np.vstack((x0_batch_2d, np.array([60, 600])))
    x1 = np.vstack((x1_batch_2d, np.array([70, 700])))
    with pytest.raises(ValueError):
        sample(x0, x1, *f_batch_2d, seed=42)


def test_x0_equal_x1(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if x0 == x1 anywhere."""
    x0_batch_2d[2, 0] = x1_batch_2d[2, 0]
    with pytest.raises(ValueError):
        sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)


def test_x0_gtr_x1(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Test error raised if x0 > x1 anywhere."""
    x0_batch_2d[2, 0] = x1_batch_2d[2, 0] + 10
    with pytest.raises(ValueError):
        sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)


## DETERMINISM #################################################################


def test_same_int_seed():
    """Test same RNG seed (specified as integer) produces same results."""
    x1 = sample(10, 20, 0.1, 0.3, seed=42)
    x2 = sample(10, 20, 0.1, 0.3, seed=42)
    assert x1==x2
    x1 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
    x2 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
    assert np.all(x1==x2)


def test_same_rng_seed():
    """Test same RNG seed (specified as np RNG) produces same results."""
    x1 = sample(10, 20, 0.1, 0.3, seed=np.random.default_rng(42))
    x2 = sample(10, 20, 0.1, 0.3, seed=np.random.default_rng(42))
    assert x1==x2
    x1 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=np.random.default_rng(42))
    x2 = sample(10, 20, 0.1, 0.3, N_samples=10, seed=np.random.default_rng(42))
    assert np.all(x1==x2)


## OUTPUT SHAPES ###############################################################


def test_shape_single_sample_single_1D():
    """Drawing single sample from single 1D cell -> float."""
    x = sample(10, 20, 0.1, 0.3, seed=42)
    assert isinstance(x, float)


def test_shape_multi_sample_single_1D():
    """Drawing multi samples from single 1D cell -> 1D array (N_samples,)."""
    x = sample(10, 20, 0.1, 0.3, N_samples=10, seed=42)
    assert x.shape == (10,)


def test_shape_single_sample_single_kD(x0_single_3d, x1_single_3d, f_single_3d):
    """Drawing single sample from single kD cell -> 1D array (k,)."""
    x = sample(x0_single_3d, x1_single_3d, *f_single_3d, seed=42)
    assert x.shape == (3,)


def test_shape_multi_sample_single_kD(x0_single_3d, x1_single_3d, f_single_3d):
    """Drawing multi samples from single kD cell -> 2D array (N_samples, k)."""
    x = sample(x0_single_3d, x1_single_3d, *f_single_3d, N_samples=10, seed=42)
    assert x.shape == (10, 3)


def test_shape_single_sample_multi_kD(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Drawing single sample from multi kD cells -> 1D array (k,)."""
    x = sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, seed=42)
    assert x.shape == (2,)
    

def test_shape_single_sample_multi_1D(x0_batch_1d, x1_batch_1d, f_batch_1d):
    """Drawing single sample from multi 1D cells -> float."""
    x = sample(x0_batch_1d, x1_batch_1d, *f_batch_1d, seed=42)
    assert isinstance(x, float)


def test_shape_multi_sample_multi_kD(x0_batch_2d, x1_batch_2d, f_batch_2d):
    """Drawing multi samples from multi kD cells -> 2D array (N_samples, k)."""
    x = sample(x0_batch_2d, x1_batch_2d, *f_batch_2d, N_samples=10, seed=42)
    assert x.shape == (10, 2)


def test_shape_multi_sample_multi_1D(x0_batch_1d, x1_batch_1d, f_batch_1d):
    """Drawing multi samples from multi 1D cells -> 1D array (N_samples,)."""
    x = sample(x0_batch_1d, x1_batch_1d, *f_batch_1d, N_samples=10, seed=42)
    assert x.shape == (10,)


## OUTPUT VALUES ###############################################################


def test_1D_uniform_single():
    """Test that single uniform sample is in appropriate range."""    
    x = sample(15, 25, 0.5, 0.5)
    assert (x < 25) and (x > 15)


def test_1D_uniform():
    """Test that batch of uniform samples gives flat histogram."""    
    x = sample(15, 25, 0.5, 0.5, N_samples=1000000)
    p = np.histogram(x, np.linspace(15, 25, 11), density=True)[0]
    assert np.all(np.round(p, decimals=1) == 0.1)
    

def test_1D_GMM():
    """Test samples from 1D 2-component GMM have correct means and widths."""
    # true params
    mua_true = -10.0
    mub_true = 10.0
    siga_true = 1.1
    sigb_true = 0.9
    wa_true = 0.4
    wb_true = 0.6

    # setup 2 PDFs
    dista = norm(loc=mua_true, scale=siga_true)
    distb = norm(loc=mub_true, scale=sigb_true)

    # hypercells: two separate 1D grids
    N_grid = 256
    ea = np.linspace(-16, -4, N_grid + 1)
    eb = np.linspace(4, 16, N_grid + 1)
    fa = wa_true * dista.pdf(ea) + wb_true * distb.pdf(ea)
    fb = wa_true * dista.pdf(eb) + wb_true * distb.pdf(eb)
    x0 = np.hstack((ea[:-1], eb[:-1]))[:, None]
    x1 = np.hstack((ea[1:], eb[1:]))[:, None]
    f0 = np.hstack((fa[:-1], fb[:-1]))
    f1 = np.hstack((fa[1:], fb[1:]))

    # draw samples
    x = sample(x0, x1, f0, f1, N_samples=1000000, seed=42)

    # find particles corresponding to each Gaussian, get sample stats
    ma = (x < 0)
    mb = ~ma    
    mua = np.round(np.mean(x[ma]), decimals=1)
    mub = np.round(np.mean(x[mb]), decimals=1)
    siga = np.round(np.std(x[ma]), decimals=1)
    sigb = np.round(np.std(x[mb]), decimals=1)
    wa = np.round(ma.sum() / len(ma), decimals=1)
    wb = 1 - wa
    
    # check sample statistics match true params
    assert mua == mua_true
    assert mub == mub_true
    assert siga == siga_true
    assert sigb == sigb_true
    assert wa == wa_true
    assert wb == wb_true


def test_2D_GMM():
    """Test samples from 2D 2-component GMM have correct means and covs."""
    # true params
    mua_true = np.array([-5, -5])
    mub_true = np.array([5, 5])
    cova_true = np.array([
        [ 1.0,  -0.5],
        [-0.5,  1.5],
    ])
    covb_true = np.array([
        [ 1.5,  0.5],
        [ 0.5,  1.0],
    ])
    wa_true = 0.3
    wb_true = 0.7

    # setup 2 PDFs
    dista = multivariate_normal(mean=mua_true, cov=cova_true)
    distb = multivariate_normal(mean=mub_true, cov=covb_true)

    # hypercells: two separate 256x256 grids
    N_grid = 256
    ea = np.linspace(-10, 0, N_grid + 1)
    ga = np.stack(np.meshgrid(ea, ea, indexing='ij'), axis=-1)
    eb = np.linspace(0, 10, N_grid + 1)
    gb = np.stack(np.meshgrid(eb, eb, indexing='ij'), axis=-1)
    fa = wa_true * dista.pdf(ga) + wb_true * distb.pdf(ga)
    fb = wa_true * dista.pdf(gb) + wb_true * distb.pdf(gb)
    x0 = np.vstack([ga[:-1, :-1].reshape((N_grid**2, 2)), gb[:-1, :-1].reshape((N_grid**2, 2))])
    x1 = np.vstack([ga[1:, 1:].reshape((N_grid**2, 2)), gb[1:, 1:].reshape((N_grid**2, 2))])
    f00 = np.hstack([fa[:-1, :-1].flatten(), fb[:-1, :-1].flatten()])
    f01 = np.hstack([fa[:-1, 1:].flatten(), fb[:-1, 1:].flatten()])
    f10 = np.hstack([fa[1:, :-1].flatten(), fb[1:, :-1].flatten()])
    f11 = np.hstack([fa[1:, 1:].flatten(), fb[1:, 1:].flatten()])
    
    # draw samples
    x = sample(x0, x1, f00, f01, f10, f11, N_samples=2000000, seed=42)
    
    # find particles corresponding to each Gaussian, get sample stats
    ma = (x[:, 0] < 0) & (x[:, 1] < 0)
    mb = ~ma
    mua = np.round(np.mean(x[ma], axis=0), decimals=1)
    cova = np.round(np.cov(x[ma].T), decimals=1)
    mub = np.round(np.mean(x[mb], axis=0), decimals=1)
    covb = np.round(np.cov(x[mb].T), decimals=1)
    wa = np.round(ma.sum() / len(ma), decimals=1)
    wb = 1 - wa
    
    # check sample statistics match true params
    assert np.all(mua == mua_true)
    assert np.all(mub == mub_true)
    assert np.all(cova == cova_true)
    assert np.all(covb == covb_true)
    assert wa == wa_true
    assert wb == wb_true
