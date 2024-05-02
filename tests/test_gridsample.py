import pytest
import numpy as np
import sys
import os
from scipy.stats import norm, multivariate_normal
from scipy.stats.qmc import Sobol, Halton
sys.path.insert(0, os.path.abspath("src"))
from lintsampler import gridsample


@pytest.fixture
def x_edges():
    return np.linspace(0, 10, 33)

@pytest.fixture
def y_edges():
    return np.linspace(100, 200, 65)

@pytest.fixture
def f_1D():
    rng = np.random.default_rng(42)
    return rng.uniform(size=33)

@pytest.fixture
def f_2D():
    rng = np.random.default_rng(42)
    return rng.uniform(size=(33, 65))


## INPUT CHECKING ##############################################################


def test_bad_seed(x_edges, f_1D):
    """Test that providing a nonsensical 'seed' raises an error."""
    with pytest.raises(TypeError):
        gridsample(x_edges, f=f_1D, seed=42.0)


def test_nonint_N_samples(x_edges, f_1D):
    """Test that providing a non-integer N_samples raises an error."""
    with pytest.raises(TypeError):
        gridsample(x_edges, f=f_1D, N_samples=10.0, seed=42)


@pytest.mark.parametrize("N_samples", [0, -5])
def test_bad_N_samples(x_edges, f_1D, N_samples):
    """Test that providing a non-integer N_samples raises an error."""
    with pytest.raises(ValueError):
        gridsample(x_edges, f=f_1D, N_samples=N_samples, seed=42)


@pytest.mark.parametrize("bad_edges", [np.array([1.0, 1.5, 1.5, 2.0]), np.ones(10), np.array([9.0, 8.0, 10.0, 12.0])])
def test_edges_non_monotonic(bad_edges):
    """Test that edge array not monotonically increasing raises error."""
    rng = np.random.default_rng(42)
    f = rng.uniform(size=len(bad_edges))
    with pytest.raises(ValueError):
        gridsample(bad_edges, f=f, seed=42)


def test_edges_2D():
    """Test that 2D edge array raises error."""
    rng = np.random.default_rng(42)
    edges = np.array([[1.0, 1.5, 2.0], [1.0, 1.5, 2.0]])
    f = rng.uniform(size=len(edges))
    with pytest.raises(TypeError):
        gridsample(edges, f=f, seed=42)


def test_bad_f_shape(x_edges, y_edges):
    """Test that error is raised if shape of f doesn't match edge arrays."""
    rng = np.random.default_rng(42)
    f_bad = rng.uniform(size=(len(x_edges) + 1, len(y_edges) + 1))
    with pytest.raises(ValueError):
        gridsample(x_edges, y_edges, f=f_bad, seed=42)


def test_bad_f_length(x_edges, y_edges):
    """Test that error is raised if f doesn't have right number of dims."""
    rng = np.random.default_rng(42)
    f_bad = rng.uniform(size=(len(x_edges), len(y_edges), 10))
    with pytest.raises(ValueError):
        gridsample(x_edges, y_edges, f=f_bad, seed=42)


def test_f_negative(x_edges, f_1D):
    """Test error raised if f negative anywhere"""
    f_bad = np.copy(f_1D)
    f_bad[2] *= -1
    with pytest.raises(ValueError):
        gridsample(x_edges, f=f_bad, seed=42)


def test_wrong_qmc_dimension(x_edges, y_edges, f_2D):
    """Test error raised if dimension of user-provided QMC engine is wrong."""
    engine = Sobol(d=2, scramble=True, seed=42)
    with pytest.raises(ValueError):
        gridsample(x_edges, y_edges, f=f_2D, qmc=True, qmc_engine=engine)


def test_wrong_qmc(x_edges, y_edges, f_2D):
    """Test error raised if user-provided QMC engine is not scipy QMC engine."""
    engine = np.random.default_rng(42)
    with pytest.raises(TypeError):
        gridsample(x_edges, y_edges, f=f_2D, qmc=True, qmc_engine=engine)


def test_non_power2_sobol_warning(x_edges, y_edges, f_2D):
    """Test warning raised if using Sobol sampler with non-power of 2."""
    with pytest.warns(UserWarning):
        gridsample(x_edges, y_edges, f=f_2D, N_samples=20, qmc=True)


def test_qmc_flag_engine_warning(x_edges, y_edges, f_2D):
    """Test warning raised if user-provided qmc engine while qmc flag False"""
    engine = Sobol(d=3, scramble=True, seed=42)
    with pytest.warns(UserWarning):
        gridsample(x_edges, y_edges, f=f_2D, qmc=False, qmc_engine=engine)


def test_qmc_seed_warning(x_edges, y_edges, f_2D):
    """Test warning raised if user-provided qmc engine while seed also given"""
    engine = Sobol(d=3, scramble=True, seed=42)
    with pytest.warns(UserWarning):
        gridsample(x_edges, y_edges, f=f_2D, qmc=True, qmc_engine=engine, seed=42)


## DETERMINISM #################################################################


def test_same_int_seed(x_edges, f_1D):
    """Test same RNG seed (specified as integer) produces same results."""
    x1 = gridsample(x_edges, f=f_1D, seed=42)
    x2 = gridsample(x_edges, f=f_1D, seed=42)
    assert x1==x2
    x1 = gridsample(x_edges, f=f_1D, N_samples=10, seed=42)
    x2 = gridsample(x_edges, f=f_1D, N_samples=10, seed=42)
    assert np.all(x1==x2)


def test_same_int_seed_qmc(x_edges, f_1D):
    """Test same RNG seed (specified as integer) produces same results."""
    x1 = gridsample(x_edges, f=f_1D, seed=42, qmc=True)
    x2 = gridsample(x_edges, f=f_1D, seed=42, qmc=True)
    assert x1==x2
    x1 = gridsample(x_edges, f=f_1D, N_samples=16, seed=42, qmc=True)
    x2 = gridsample(x_edges, f=f_1D, N_samples=16, seed=42, qmc=True)
    assert np.all(x1==x2)


def test_same_rng_seed(x_edges, f_1D):
    """Test same RNG seed (specified as np RNG) produces same results."""
    x1 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42))
    x2 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42))
    assert x1==x2
    x1 = gridsample(x_edges, f=f_1D, N_samples=10, seed=np.random.default_rng(42))
    x2 = gridsample(x_edges, f=f_1D, N_samples=10, seed=np.random.default_rng(42))
    assert np.all(x1==x2)


def test_same_rng_seed_qmc(x_edges, f_1D):
    """Test same RNG seed (specified as np RNG) produces same results."""
    x1 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42), qmc=True)
    x2 = gridsample(x_edges, f=f_1D, seed=np.random.default_rng(42), qmc=True)
    assert x1==x2
    x1 = gridsample(x_edges, f=f_1D, N_samples=16, seed=np.random.default_rng(42), qmc=True)
    x2 = gridsample(x_edges, f=f_1D, N_samples=16, seed=np.random.default_rng(42), qmc=True)
    assert np.all(x1==x2)


def test_qmc_engine_reset(x_edges, y_edges, f_2D):
    """Test warning raised if user-provided qmc engine while seed also given"""
    engine = Sobol(d=3, scramble=True, seed=42)
    x1 = gridsample(x_edges, y_edges, f=f_2D, N_samples=16, qmc=True, qmc_engine=engine)
    engine.reset()
    x2 = gridsample(x_edges, y_edges, f=f_2D, N_samples=16, qmc=True, qmc_engine=engine)
    assert np.all(x1==x2)


## OUTPUT SHAPES ###############################################################


def test_float_single_sample_1D_float(x_edges, f_1D):
    """Test that drawing a single sample in 1D returns a float"""
    x = gridsample(x_edges, f=f_1D, seed=42)
    assert isinstance(x, float)


def test_float_single_sample_1D_float_qmc(x_edges, f_1D):
    """Test that drawing a single sample in 1D returns a float"""
    x = gridsample(x_edges, f=f_1D, seed=42, qmc=True)
    assert isinstance(x, float)


def test_shape_multiple_samples_1D(x_edges, f_1D):
    """Test that N samples in 1D returns 1D array shape (N,)"""
    x = gridsample(x_edges, f=f_1D, N_samples=10, seed=42)
    assert x.shape == (10,)


def test_shape_multiple_samples_1D_qmc(x_edges, f_1D):
    """Test that N samples in 1D returns 1D array shape (N,)"""
    x = gridsample(x_edges, f=f_1D, N_samples=16, seed=42, qmc=True)
    assert x.shape == (16,)


def test_shape_single_sample_kD(x_edges, y_edges, f_2D):
    """Test that 1 sample in kD returns 1D array shape (k,)"""
    x = gridsample(x_edges, y_edges, f=f_2D, seed=42)
    assert x.shape == (2,)


def test_shape_single_sample_kD_qmc(x_edges, y_edges, f_2D):
    """Test that 1 sample in kD returns 1D array shape (k,)"""
    x = gridsample(x_edges, y_edges, f=f_2D, seed=42, qmc=True)
    assert x.shape == (2,)



def test_shape_multiple_samples_kD(x_edges, y_edges, f_2D):
    """Test that N samples in kD returns 2D array shape (N, k)"""
    x = gridsample(x_edges, y_edges, f=f_2D, N_samples=10, seed=42)
    assert x.shape == (10, 2)


def test_shape_multiple_samples_kD_qmc(x_edges, y_edges, f_2D):
    """Test that N samples in kD returns 2D array shape (N, k)"""
    x = gridsample(x_edges, y_edges, f=f_2D, N_samples=16, seed=42, qmc=True)
    assert x.shape == (16, 2)



## OUTPUT VALUES ###############################################################


def test_1D_gaussian():
    """Test samples from a 1D gaussian have correct mean and width"""
    mu_true = 30.0
    sig_true = 1.8
    x_gaussian = np.linspace(20, 40, 512)
    f_gaussian = norm.pdf(x_gaussian, loc=mu_true, scale=sig_true)
    
    x = gridsample(x_gaussian, f=f_gaussian, N_samples=100000, seed=42)
    mu = np.round(np.mean(x), decimals=0)
    sig = np.round(np.std(x), decimals=1)
    
    assert (mu, sig) == (mu_true, sig_true)


def test_1D_gaussian_qmc():
    """Test samples from a 1D gaussian have correct mean and width"""
    mu_true = 30.0
    sig_true = 1.8
    x_gaussian = np.linspace(20, 40, 512)
    f_gaussian = norm.pdf(x_gaussian, loc=mu_true, scale=sig_true)
    
    x = gridsample(x_gaussian, f=f_gaussian, N_samples=2**20, seed=42, qmc=True)
    mu = np.round(np.mean(x), decimals=0)
    sig = np.round(np.std(x), decimals=1)
    
    assert (mu, sig) == (mu_true, sig_true)


def test_1D_gaussian_qmc_halton():
    """1D Gaussian test with Halton engine."""
    mu_true = 30.0
    sig_true = 1.8
    x_gaussian = np.linspace(20, 40, 512)
    f_gaussian = norm.pdf(x_gaussian, loc=mu_true, scale=sig_true)
    
    engine = Halton(d=2)
    x = gridsample(x_gaussian, f=f_gaussian, N_samples=2**20, qmc=True, qmc_engine=engine)
    mu = np.round(np.mean(x), decimals=0)
    sig = np.round(np.std(x), decimals=1)
    
    assert (mu, sig) == (mu_true, sig_true)


def test_kd_gaussian():
    """Test samples from a kD gaussian have correct mean and covariances"""
    mu_true = np.array([3.0, -0.5])
    cov_true = np.array([
        [ 1.0,  -0.5],
        [-0.5,  1.5],
    ])
    edges = np.linspace(-10, 10, 513)
    grid = np.stack(np.meshgrid(edges, edges, indexing='ij'), axis=-1)
    f = multivariate_normal.pdf(grid, mean=mu_true, cov=cov_true)
    x = gridsample(edges, edges, f=f, N_samples=1000000)

    mu = np.round(np.mean(x, axis=0), decimals=1)
    cov = np.round(np.cov(x.T), decimals=1)
    assert np.all(mu == mu_true) and np.all(cov == cov_true)


def test_kd_gaussian_qmc():
    """Test samples from a kD gaussian have correct mean and covariances"""
    mu_true = np.array([3.0, -0.5])
    cov_true = np.array([
        [ 1.0,  -0.5],
        [-0.5,  1.5],
    ])
    edges = np.linspace(-10, 10, 513)
    grid = np.stack(np.meshgrid(edges, edges, indexing='ij'), axis=-1)
    f = multivariate_normal.pdf(grid, mean=mu_true, cov=cov_true)
    x = gridsample(edges, edges, f=f, N_samples=2**20, qmc=True)

    mu = np.round(np.mean(x, axis=0), decimals=1)
    cov = np.round(np.cov(x.T), decimals=1)
    assert np.all(mu == mu_true) and np.all(cov == cov_true)


def test_kd_gaussian_qmc_halton():
    """kD Gaussian test with Halton engine."""
    mu_true = np.array([3.0, -0.5])
    cov_true = np.array([
        [ 1.0,  -0.5],
        [-0.5,  1.5],
    ])
    edges = np.linspace(-10, 10, 513)
    grid = np.stack(np.meshgrid(edges, edges, indexing='ij'), axis=-1)
    f = multivariate_normal.pdf(grid, mean=mu_true, cov=cov_true)
    
    engine = Halton(d=3)
    x = gridsample(edges, edges, f=f, N_samples=2**20, qmc=True, qmc_engine=engine)

    mu = np.round(np.mean(x, axis=0), decimals=1)
    cov = np.round(np.cov(x.T), decimals=1)
    assert np.all(mu == mu_true) and np.all(cov == cov_true)
