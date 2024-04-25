import numpy as np
from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as Rect
plt.style.use('figstyle.mplstyle')


if __name__ == "__main__":
    
    COV = np.array([
        [1.0, 0.8],
        [0.8, 1.6],
    ])
    dist = multivariate_normal(mean=np.zeros(2), cov=COV) 

    # fig setup
    fig = plt.figure(figsize=(6, 6))
    left = 0.1
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
    for i in range(Nx):
        for j in range(Ny):
            x = X_EDGES[i]
            y = Y_EDGES[j]
            dx = np.diff(X_EDGES)[i]
            dy = np.diff(Y_EDGES)[j]
            r1 = Rect((x, y), dx, dy, linewidth=0.5, edgecolor='grey', linestyle='dotted', facecolor='none')
            ax.add_patch(r1)
    s = ax.scatter(X_GRID.flatten(), Y_GRID.flatten(), c=p.flatten(), cmap='magma', s=30, vmin=0, vmax=16)
    plt.colorbar(s, cax=cax)

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
    
    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(-5.1, 5.1)
    
    # tick labels
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    
    cax.set_yticks([0, 16])
    cax.set_ylabel("Density")
    
    # save
    fig.savefig("../source/assets/worked_example_grid.png", bbox_inches='tight')

    plt.show()