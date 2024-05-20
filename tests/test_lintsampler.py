import pytest
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.stats.qmc import Sobol, Halton
import sys
import os
sys.path.insert(0, os.path.abspath("src"))
from lintsampler import LintSampler

X_EDGES = np.linspace(-10, 10, 65)
Y_EDGES = np.linspace(-5, 5, 33)
XY_GRID = np.stack(np.meshgrid(X_EDGES, Y_EDGES, indexing='ij'), axis=-1)
X0_EDGES = np.linspace(-10, 0, 65)
X1_EDGES = np.linspace(0, 10, 65)
Y0_EDGES = np.linspace(-5, 0, 33)
Y1_EDGES = np.linspace(0, 5, 33)
XY00_GRID = np.stack(np.meshgrid(X0_EDGES, Y0_EDGES, indexing='ij'), axis=-1)
XY01_GRID = np.stack(np.meshgrid(X0_EDGES, Y1_EDGES, indexing='ij'), axis=-1)
XY10_GRID = np.stack(np.meshgrid(X1_EDGES, Y0_EDGES, indexing='ij'), axis=-1)
XY11_GRID = np.stack(np.meshgrid(X1_EDGES, Y1_EDGES, indexing='ij'), axis=-1)
BAD_CELLS = [
    np.array([1.0, 1.5, 1.5, 2.0]),
    np.ones(10),
    np.array([9.0, 8.0, 10.0, 12.0]),
]
CELLS_1D = [
    X_EDGES, 
    tuple(X_EDGES),
    list(X_EDGES),
    [X0_EDGES, X1_EDGES],
    [tuple(X0_EDGES), tuple(X1_EDGES)],
    [list(X0_EDGES), list(X1_EDGES)],
]
CELLS_2D = [
    (X_EDGES, Y_EDGES),
    (tuple(X_EDGES), tuple(Y_EDGES)),
    (list(X_EDGES), list(Y_EDGES)),
    XY_GRID,
    [(X0_EDGES, Y0_EDGES), (X0_EDGES, Y1_EDGES), (X1_EDGES, Y0_EDGES), (X1_EDGES, Y1_EDGES)],
    [(tuple(X0_EDGES), tuple(Y0_EDGES)), (tuple(X0_EDGES), tuple(Y1_EDGES)), (tuple(X1_EDGES), tuple(Y0_EDGES)), (tuple(X1_EDGES), tuple(Y1_EDGES))],
    [(list(X0_EDGES), list(Y0_EDGES)), (list(X0_EDGES), list(Y1_EDGES)), (list(X1_EDGES), list(Y0_EDGES)), (list(X1_EDGES), list(Y1_EDGES))],
    [XY00_GRID, XY01_GRID, XY10_GRID, XY11_GRID],
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
    """Test error raised if 1D edges not monotonic"""
    sampler = LintSampler(norm.pdf, cells, vectorizedpdf=True, seed=42)
    with pytest.raises(ValueError):
        sampler.sample()


@pytest.mark.parametrize("qmc_engine", [Sobol(d=1, scramble=True, seed=42), Halton(d=1, seed=42), Sobol(d=3, scramble=True, seed=42), Halton(d=3, seed=42)])
def test_wrong_qmc_dimension(qmc_engine):
    """Test error raised if dimension of user-provided QMC engine is wrong."""
    with pytest.raises(ValueError):
        sampler = LintSampler(norm.pdf, cells=X_EDGES, vectorizedpdf=True, qmc=True, qmc_engine=qmc_engine)
        sampler.sample()


@pytest.mark.parametrize("qmc_engine", [123.45, np.random.default_rng()])
def test_wrong_qmc(qmc_engine):
    """Test error raised if user-provided QMC engine is not scipy QMC engine."""
    with pytest.raises(TypeError):
        sampler = LintSampler(norm.pdf, cells=X_EDGES, vectorizedpdf=True, qmc=True, qmc_engine=qmc_engine)
        sampler.sample()


def test_non_power2_sobol_warning():
    """Test warning raised if using Sobol sampler with non-power of 2."""
    with pytest.warns(UserWarning):
        sampler = LintSampler(norm.pdf, cells=X_EDGES, vectorizedpdf=True, qmc=True)
        sampler.sample(N_samples=10)


def test_qmc_flag_engine_warning():
    """Test warning raised if user-provided qmc engine while qmc flag False"""
    engine = Sobol(d=2, scramble=True, seed=42)
    with pytest.warns(UserWarning):
        sampler = LintSampler(norm.pdf, cells=X_EDGES, vectorizedpdf=True, qmc=False, qmc_engine=engine)
        sampler.sample()


def test_qmc_seed_warning():
    """Test warning raised if user-provided qmc engine while seed also given"""
    engine = Sobol(d=2, scramble=True, seed=42)
    with pytest.warns(UserWarning):
        sampler = LintSampler(norm.pdf, cells=X_EDGES, vectorizedpdf=True, qmc=True, qmc_engine=engine, seed=42)
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


## DETERMINISM #################################################################


@pytest.mark.parametrize("N_samples", [None, 16])
@pytest.mark.parametrize("qmc", [True, False])
def test_same_int_seed(N_samples, qmc):
    """Test same RNG seed (specified as int) produces same results."""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    sampler1 = LintSampler(dist.pdf, (X_EDGES, Y_EDGES), vectorizedpdf=True, qmc=qmc, seed=42)
    sampler2 = LintSampler(dist.pdf, (X_EDGES, Y_EDGES), vectorizedpdf=True, qmc=qmc, seed=42)
    x1 = sampler1.sample(N_samples=N_samples)
    x2 = sampler2.sample(N_samples=N_samples)
    assert np.all(x1==x2)


@pytest.mark.parametrize("N_samples", [None, 16])
@pytest.mark.parametrize("qmc", [True, False])
def test_same_rng_seed(N_samples, qmc):
    """Test same RNG seed (specified as numpy rng) produces same results."""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    sampler1 = LintSampler(dist.pdf, (X_EDGES, Y_EDGES), vectorizedpdf=True, qmc=qmc, seed=np.random.default_rng(42))
    sampler2 = LintSampler(dist.pdf, (X_EDGES, Y_EDGES), vectorizedpdf=True, qmc=qmc, seed=np.random.default_rng(42))
    x1 = sampler1.sample(N_samples=N_samples)
    x2 = sampler2.sample(N_samples=N_samples)
    assert np.all(x1==x2)


@pytest.mark.parametrize("qmc_engine", [Sobol(d=3, scramble=True, seed=42), Halton(d=3, seed=42)])
def test_qmc_engine_reset(qmc_engine):
    """Test same same results after resetting QMC engine."""
    dist = multivariate_normal(mean=np.ones(2), cov=np.eye(2))
    sampler = LintSampler(dist.pdf, (X_EDGES, Y_EDGES), vectorizedpdf=True, qmc=True, qmc_engine=qmc_engine)
    x1 = sampler.sample()
    qmc_engine.reset()
    x2 = sampler.sample()
    assert np.all(x1==x2)


## OUTPUT VALUES ###############################################################


@pytest.mark.parametrize("qmc", [True, False])
def test_1D_uniform(qmc):
    """Test that batch of uniform samples gives flat histogram."""
    cells = np.array([15.0, 25.0])
    uniform_pdf = lambda x: np.ones_like(x)
    sampler = LintSampler(uniform_pdf, cells=cells, vectorizedpdf=True, qmc=qmc)
    x = sampler.sample(N_samples=2**17)
    p = np.histogram(x, np.linspace(15, 25, 11), density=True)[0]
    assert np.all(np.round(p, decimals=1) == 0.1)


@pytest.mark.parametrize("cells", CELLS_1D)
@pytest.mark.parametrize("vectorizedpdf", [True, False])
@pytest.mark.parametrize("qmc,qmc_engine", [(False, None), (True, None), (True, Halton(d=2))])
def test_1D_gaussian(cells, vectorizedpdf, qmc, qmc_engine):
    """Test samples from a 1D gaussian have correct mean and variance"""
    mu_true = -2.0
    sig_true = 1.8
    d = norm(loc=mu_true, scale=sig_true)
    sampler = LintSampler(d.pdf, cells, vectorizedpdf=vectorizedpdf, qmc=qmc, qmc_engine=qmc_engine)
    x = sampler.sample(N_samples=2**17)
    mu = np.round(np.mean(x), decimals=0)
    sig = np.round(np.std(x), decimals=1)
    assert (mu, sig) == (mu_true, sig_true)


@pytest.mark.parametrize("cells", CELLS_2D)
@pytest.mark.parametrize("vectorizedpdf", [True, False])
@pytest.mark.parametrize("qmc,qmc_engine", [(False, None), (True, None), (True, Halton(d=3))])
def test_kD_gaussian(cells, vectorizedpdf, qmc, qmc_engine):
    mu_true = np.array([1.5, -0.5])
    cov_true = np.array([
        [ 1.0,  -0.5],
        [-0.5,  1.5],
    ])
    dist = multivariate_normal(mean=mu_true, cov=cov_true)

    sampler = LintSampler(dist.pdf, cells=cells, vectorizedpdf=vectorizedpdf, qmc=qmc, qmc_engine=qmc_engine)
    x = sampler.sample(N_samples=2**18)

    mu = np.round(np.mean(x, axis=0), decimals=1)
    cov = np.round(np.cov(x.T), decimals=1)
    assert np.all(mu == mu_true) and np.all(cov == cov_true)


@pytest.mark.parametrize("qmc", [True, False])
def test_1D_GMM(qmc):
    """Test samples from 1D 2-component 1D GMM have correct means and widths."""
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
    pdf = lambda x: wa_true * dista.pdf(x) + wb_true * distb.pdf(x)

    # cells: two separate 1D grids
    N_grid = 256
    ga = np.linspace(-16, -4, N_grid + 1)
    gb = np.linspace(4, 16, N_grid + 1)
    cells = [ga, gb]
    
    # draw samples
    sampler = LintSampler(pdf=pdf, cells=cells, vectorizedpdf=True, qmc=qmc)
    x = sampler.sample(2**17)

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


@pytest.mark.parametrize("qmc", [True, False])
def test_2D_GMM(qmc):
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
    pdf = lambda x: wa_true * dista.pdf(x) + wb_true * distb.pdf(x)

    # cells: two separate 256x256 grids
    N_grid = 256
    ea = np.linspace(-10, 0, N_grid + 1)
    eb = np.linspace(0, 10, N_grid + 1)
    cells = [(ea, ea), (eb, eb)]
    
    # draw samples
    sampler = LintSampler(pdf, cells=cells, vectorizedpdf=True, qmc=qmc)
    x = sampler.sample(N_samples=2**18)
    
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
