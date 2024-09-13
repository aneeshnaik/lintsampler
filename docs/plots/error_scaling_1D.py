import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from lintsampler import LintSampler
plt.style.use('figstyle.mplstyle')


def gmm_pdf(x):
    mu = np.array([-3.0, 0.5, 2.5])
    sig = np.array([1.0, 0.25, 0.75])
    w = np.array([0.4, 0.25, 0.35])
    return np.sum([w[i] * norm.pdf(x, mu[i], sig[i]) for i in range(3)], axis=0)


if __name__ == "__main__":
    
    # plot parameters
    N_calc = 20
    Ng_arr = np.logspace(4, 15, 12, base=2, dtype=int)
    
    # RNG
    rng = np.random.default_rng(42)
    
    # calculate reference value for error
    ref, _ = quad(lambda x: np.log(gmm_pdf(x)) * gmm_pdf(x), -12, 12, epsabs=1e-10, epsrel=1e-10)
    
    # loop over N_grid, calculate error
    err_arr1 = np.zeros(len(Ng_arr))
    err_arr2 = np.zeros(len(Ng_arr))
    for i, Ng in enumerate(Ng_arr):

        # set up samplers
        grid = np.linspace(-12, 12, Ng + 1)
        sampler1 = LintSampler(grid, pdf=gmm_pdf, vectorizedpdf=True, seed=rng)
        sampler2 = LintSampler(grid, pdf=gmm_pdf, vectorizedpdf=True, seed=rng, qmc=True)

        # calculate error
        err_arr1[i] = np.median([np.abs(np.log(gmm_pdf(sampler1.sample(int(Ng)))).mean() - ref) for _ in range(N_calc)])
        err_arr2[i] = np.median([np.abs(np.log(gmm_pdf(sampler2.sample(int(Ng)))).mean() - ref) for _ in range(N_calc)])
        
    # set up figure
    fig = plt.figure(figsize=(6, 4.5))
    ax = fig.add_axes([0.15, 0.17, 0.8, 0.8])
 
    # plot trends
    ax.plot(Ng_arr, err_arr1, c='teal', lw=2, label='Pseudo-random')
    ax.plot(Ng_arr, err_arr2, c='goldenrod', lw=2, label='QMC')
    plt.legend(frameon=False)

    # power law guides
    ax.plot([32, 128], [1e-2, 1e-2/4], c='lightgrey', ls='dashed')
    ax.plot([128, 512], [1e-1, 1e-1/2], c='lightgrey', ls='dashed')
    ax.text(38, 10**(-2.6), r'$N^{-1}$', usetex=True)
    ax.text(180, 1e-1, r'$N^{-1/2}$', usetex=True)

    # plot settings
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines[['bottom']].set_position(('data', 1e-5))
    ax.spines[['left']].set_position(('data', 2**3))
    
    plt.xlabel(r'$N_\mathrm{grid}\ (=N_\mathrm{samples})$')
    plt.ylabel('Error')
    
    plt.savefig("../source/assets/error_scaling_1D.png", bbox_inches='tight')
    
    plt.show()