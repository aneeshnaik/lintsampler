import numpy as np
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib.colors import LogNorm
from scipy.stats import norm
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter
from lintsampler import LintSampler
plt.style.use('figstyle.mplstyle')


def gmm_pdf(x):
    mu = np.array([-3.0, 0.5, 2.5])
    sig = np.array([1.0, 0.25, 0.75])
    w = np.array([0.4, 0.25, 0.35])
    return np.sum([w[i] * norm.pdf(x, mu[i], sig[i]) for i in range(3)], axis=0)

def create_plot_data(plot_data_file):

    # some parameters
    N_calc = 20
    Ng_arr = np.logspace(4, 16, 64, base=2, dtype=int)
    Ns_arr = np.logspace(4, 16, 64, base=2, dtype=int)
    
    # RNG
    rng = np.random.default_rng(42)
    
    # calculate reference value for error
    ref, _ = quad(lambda x: np.log(gmm_pdf(x)) * gmm_pdf(x), -12, 12, epsabs=1e-10, epsrel=1e-10)

    # double loop over grid
    err_grid = np.zeros((len(Ng_arr), len(Ns_arr)))
    for i, Ng in enumerate(Ng_arr):
        print(i, Ng)
        sampler = LintSampler(np.linspace(-12, 12, Ng + 1), pdf=gmm_pdf, vectorizedpdf=True, seed=rng)        
        for j, Ns in enumerate(Ns_arr):
            err_grid[i, j] = np.median([np.abs(np.log(gmm_pdf(sampler.sample(int(Ns)))).mean() - ref) for _ in range(N_calc)])

    # save    
    np.savez(plot_data_file, Ng_arr=Ng_arr, Ns_arr=Ns_arr, err_grid=err_grid)


if __name__ == "__main__":
    
    # load saved plot data (create if not present)
    plot_data_file = 'error_scaling_2D_data.npz'
    if not exists(plot_data_file):
        create_plot_data(plot_data_file)
    data = np.load(plot_data_file)
    Ng_arr = data['Ng_arr']
    Ns_arr = data['Ns_arr']
    err_grid = data['err_grid']

    # fig setup
    fig = plt.figure(figsize=(6, 6))
    X0 = 0.14
    X1 = 0.82
    X2 = 0.87
    dX = X1 - X0
    cdX = 0.035
    dY = dX
    Y0 = 0.15
    ax = fig.add_axes([X0, Y0, dX, dY])
    cax = fig.add_axes([X2, Y0, cdX, dY])
    
    # plot contour map
    fc = ax.contourf(
        Ng_arr, Ns_arr, gaussian_filter(err_grid, sigma=3),
        norm=LogNorm(), levels=np.logspace(-3, 0, 13, base=10), cmap='magma_r'
    )
    
    # scales/labels etc
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xlabel(r'$N_\mathrm{samples}$')
    ax.set_ylabel(r'$N_\mathrm{grid}$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines[['left', 'bottom']].set_position(('data', 10))
    
    # colorbar
    plt.colorbar(fc, cax=cax)
    cax.set_ylabel('Error')
    
    # save
    fig.savefig("../source/assets/error_scaling_2D.png", bbox_inches='tight')
    
    # show
    plt.show()
