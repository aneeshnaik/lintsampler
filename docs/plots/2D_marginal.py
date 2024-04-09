import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('figstyle.mplstyle')


def sgnf_grid(x, mean, cov):
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    
    # evaluate significance on grid
    icov = np.linalg.inv(cov)
    sgnf = np.sqrt(np.sum((grid - mean) * np.sum((grid[:, :, None] - mean) * icov, axis=-1), axis=-1))
    
    return grid[..., 0], grid[..., 1], sgnf


if __name__ == "__main__":
    
    # PDF
    mean = np.array([1.5, -0.5])
    cov = np.array([
        [1.0, 1.2],
        [1.2, 1.8]
    ])
    
    # fig setup
    asp = 6 / 8
    fig = plt.figure(figsize=(6, 6 / asp))
    bottom = 0.08
    left = 0.1
    right = 0.9
    dX = right - left
    dY = asp * dX
    axb = fig.add_subplot([left, bottom, dX, dY])
    axt = fig.add_subplot([left, bottom + dY, dX, 0.5 * dY])

    # coord grid
    x = np.linspace(-6, 6, 512)
    
    # get signifance on grid
    xgr, ygr, sgnf = sgnf_grid(x, mean, cov)

    # plot contours
    axb.contour(xgr, ygr, sgnf, levels=[1, 2, 3, 4], colors='teal')

    # plot marginal pdf
    y = norm.pdf(x, loc=mean[0], scale=np.sqrt(cov[0, 0]))
    axt.plot(x, y, c='teal', lw=2)

    # make x lims match, remove extraneous spines and y ticks/labels
    for ax in [axb, axt]:
        ax.set_xlim(-6.5, 6.5)
        ax.tick_params(left=False, labelleft=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_bounds(-6, 6)
    
    # bottom panel: make y lims match x lims
    axb.set_ylim(-6.5, 6.5)
    
    # top panel: remove x labels and make ticks inout
    axt.tick_params(labelbottom=False, direction='inout')
    
    # axis label
    axb.set_xlabel(r'$x$', usetex=True)
    
    # equations
    axb.text(0.5, 0.6, r'$p(x, y)$', ha='right', va='top', transform=axb.transAxes, usetex=True)
    axt.text(0.5, 0.6, r'$p(x)=\displaystyle\int dy\ p(x, y)$', ha='right', va='top', transform=axt.transAxes, usetex=True)
    
    # save
    fig.savefig("../source/assets/2D_marginal.png", bbox_inches='tight')
    
    plt.show()