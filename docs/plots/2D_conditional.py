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
    xslice = 3.5
    muy = mean[1] + (cov[0, 1] / cov[0, 0]) * (xslice - mean[0])
    sigy = np.sqrt(cov[1, 1] - cov[0, 1]**2 / cov[0, 0])
    
    # fig setup
    asp = 6 / 10
    fig = plt.figure(figsize=(6 / asp, 6))
    bottom = 0.04
    top = 0.9
    left = 0.1
    dY = top - bottom
    dX = asp * dY
    axl = fig.add_subplot([left, bottom, dX, dY])
    axr = fig.add_subplot([left + dX, bottom, 0.5 * dX, dY])

    # coord grid
    x = np.linspace(-6, 6, 512)
    
    # get signifance on grid
    xgr, ygr, sgnf = sgnf_grid(x, mean, cov)

    # plot contours
    axl.contour(xgr, ygr, sgnf, levels=[1, 2, 3, 4], colors='teal')
    
    # plot slice line
    axl.plot([xslice, xslice], [-6, 6], c='goldenrod', lw=2)
    
    # plot conditional pdf
    y = norm.pdf(x, loc=muy, scale=sigy)
    axr.plot(y, x, c='teal', lw=2)
    
    # make y lims match, remove extraneous spines and x ticks/labels
    for ax in [axl, axr]:
        ax.set_ylim(-6.5, 6.5)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_bounds(-6, 6)
    
    # left panel: make y lims match x lims
    axl.set_ylim(-6.5, 6.5)
    
    # right panel: remove y labels and make ticks inout
    axr.tick_params(labelleft=False, direction='inout')
    
    # axis label
    axl.set_ylabel(r'$y$', usetex=True)
    
    # equations
    axl.text(0.5, 0.6, r'$p(x, y)$', ha='right', va='top', transform=axl.transAxes, usetex=True)
    axl.text(0.8, 0.2, r'$x=3.5$', ha='left', va='top', transform=axl.transAxes, usetex=True, rotation=270)
    axr.text(0.8, 0.75, r'$p(y | x=3.5)$', ha='right', va='top', transform=axr.transAxes, usetex=True)
    
    # save
    fig.savefig("../source/assets/2D_conditional.png", bbox_inches='tight')
    
    plt.show()