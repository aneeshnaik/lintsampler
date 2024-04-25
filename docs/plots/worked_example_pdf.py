import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

def sgnf_grid(x, mean, cov):
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    
    # evaluate significance on grid
    icov = np.linalg.inv(cov)
    sgnf = np.sqrt(np.sum((grid - mean) * np.sum((grid[:, :, None] - mean) * icov, axis=-1), axis=-1))
    
    return grid[..., 0], grid[..., 1], sgnf


if __name__ == "__main__":
    
    # PDF
    mean = np.array([0, 0])
    cov = np.array([
        [1.0, 0.8],
        [0.8, 1.6]
    ])
    
    # coord array
    x = np.linspace(-5.5, 5.5, 512)

    # get signifance on grid
    xgr, ygr, sgnf = sgnf_grid(x, mean, cov)
    
    # fig setup
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    # plot contours
    ax.contour(xgr, ygr, sgnf, levels=[1, 2, 3, 4], colors='teal')
    
    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set x spine extent
    ax.spines['bottom'].set_bounds(-5, 5)
    ax.spines['left'].set_bounds(-5, 5)
    
    # axis labels
    ax.set_xlabel(r'$x$', usetex=True)
    ax.set_ylabel(r'$y$', usetex=True)
    
    # tick labels
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    
    # save
    fig.savefig("../source/assets/worked_example_pdf.png", bbox_inches='tight')
    
    plt.show()