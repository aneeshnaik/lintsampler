import numpy as np
from scipy.stats import multivariate_normal

from lintsampler.gridsample import _gridcell_faverages, _gridcell_volumes

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

plt.style.use('figstyle.mplstyle')


if __name__ == "__main__":
    
    COV = np.array([
        [1.0, 0.8],
        [0.8, 1.6],
    ])
    dist = multivariate_normal(mean=np.zeros(2), cov=COV) 

    chosen_cell = (5, 3)

    # fig setup
    fig = plt.figure(figsize=(6, 6))
    left = 0.11
    right = 0.9
    cf = 0.06
    cgap = 0.05
    dX = (right - left - cgap) / (1 + cf)
    cdX = cf * dX
    dY = dX
    bottom = 0.1
    ax = fig.add_axes([left, bottom, dX, dY])
    cax = fig.add_axes([left + dX + cgap, bottom, cdX, dY])

    X_EDGES = np.array([-5, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5])
    Y_EDGES = np.linspace(-5, 5, 11)
    Nx = len(X_EDGES) - 1
    Ny = len(Y_EDGES) - 1

    X_GRID, Y_GRID = np.meshgrid(X_EDGES, Y_EDGES, indexing='ij')
    p = 100*dist.pdf(np.stack((X_GRID, Y_GRID), axis=-1))
    
    # get cell masses    
    V = _gridcell_volumes(X_EDGES, Y_EDGES)
    f_avg = _gridcell_faverages(p)
    m = f_avg * V
    m /= m.sum()
    
    norm = Normalize(vmin=0, vmax=0.05)
    cmappable = ScalarMappable(cmap='magma', norm=norm)
    
    for i in range(Nx):
        for j in range(Ny):
            x = X_EDGES[i]
            y = Y_EDGES[j]
            dx = np.diff(X_EDGES)[i]
            dy = np.diff(Y_EDGES)[j]
            if (i == chosen_cell[0]) and (j == chosen_cell[1]):
                ec = 'skyblue'
                ls = 'solid'
                lw = 1.5
                zo = 1
            else:
                ec = 'lightgrey'
                ls = 'dotted'
                lw = 0.5
                zo = 0
            r = Rect((x, y), dx, dy, linewidth=lw, edgecolor=ec, linestyle=ls, facecolor=cmappable.to_rgba(m[i, j]), zorder=zo)
            ax.add_patch(r)
    #s = ax.scatter(X_GRID.flatten(), Y_GRID.flatten(), c=p.flatten(), cmap='inferno', s=30, vmin=0, vmax=16)
    plt.colorbar(cmappable, cax=cax)

    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set x spine extent
    ax.spines['bottom'].set_bounds(-5, 5)
    ax.spines['left'].set_bounds(-5, 5)
    ax.spines[['left', 'bottom']].set_position(('data', -5.45))
    
    # axis labels
    ax.set_xlabel(r'$x$', usetex=True)
    ax.set_ylabel(r'$y$', usetex=True)
    
    ax.set_xlim(-5.02, 5.02)
    ax.set_ylim(-5.02, 5.02)
    
    # tick labels
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    
    cax.set_yticks([0, 0.05])
    cax.set_ylabel("Probability")
    
    # save
    fig.savefig("../source/assets/worked_example_cellprobs.png", bbox_inches='tight')

    plt.show()