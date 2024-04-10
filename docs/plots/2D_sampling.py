import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
plt.style.use('figstyle.mplstyle')


def sgnf_grid(x, mean, cov):
    grid = np.stack(np.meshgrid(x, x, indexing='ij'), axis=-1)
    
    # evaluate significance on grid
    icov = np.linalg.inv(cov)
    sgnf = np.sqrt(np.sum((grid - mean) * np.sum((grid[:, :, None] - mean) * icov, axis=-1), axis=-1))
    
    return grid[..., 0], grid[..., 1], sgnf


def generate_sample(mean, cov):
    
    rng = np.random.default_rng(42)
    ux = rng.uniform()
    uy = rng.uniform()
    
    xs = norm.ppf(ux, loc=mean[0], scale=cov[0, 0])
    muy = mean[1] + (cov[0, 1] / cov[0, 0]) * (xs - mean[0])
    sigy = np.sqrt(cov[1, 1] - cov[0, 1]**2 / cov[0, 0])
    ys = norm.ppf(uy, loc=muy, scale=sigy)

    return ux, uy, xs, ys


if __name__ == "__main__":

    # PDF
    mean = np.array([1.5, -0.5])
    cov = np.array([
        [1.0, 1.2],
        [1.2, 1.8]
    ])

    # generate samples    
    N_SAMPLES = 1000
    ux, uy, xs, ys = generate_sample(mean, cov)
    
    # conditional PDF params
    muy = mean[1] + (cov[0, 1] / cov[0, 0]) * (xs - mean[0])
    sigy = np.sqrt(cov[1, 1] - cov[0, 1]**2 / cov[0, 0])
    
    # figure
    fig = plt.figure(figsize=(10, 10))
    left = 0.1
    bottom = 0.1
    dX = 0.4
    dY = 0.4
    ax0 = fig.add_axes([left, bottom, dX, dY])
    axxp = fig.add_axes([left, bottom + dY, dX, 0.5 * dY])
    axxc = fig.add_axes([left, bottom + 1.5 * dY, dX, 0.5 * dY])
    axyp = fig.add_axes([left + dX, bottom, 0.5 * dX, dY])
    axyc = fig.add_axes([left + 1.5 * dX, bottom, 0.5 * dX, dY])

    # coord grid
    x = np.linspace(-6, 6, 512)

    # get signifance on grid
    xgr, ygr, sgnf = sgnf_grid(x, mean, cov)

    # plot contours
    ax0.contour(xgr, ygr, sgnf, levels=[1, 2, 3, 4], colors='teal')
    
    # plot x marginal pdf and CDF
    y = norm.pdf(x, loc=mean[0], scale=np.sqrt(cov[0, 0]))
    F = norm.cdf(x, loc=mean[0], scale=np.sqrt(cov[0, 0]))
    axxp.plot(x, y, c='teal', lw=2)
    axxc.plot(x, F, c='teal', lw=2)
    
    # plot y conditional pdf and CDF
    y = norm.pdf(x, loc=muy, scale=sigy)
    F = norm.cdf(x, loc=muy, scale=sigy)
    axyp.plot(y, x, c='teal', lw=2)
    axyc.plot(F, x, c='teal', lw=2)

    # plot various guide lines
    axxc.plot([-6.5, xs], [ux, ux], lw=1, c='grey')
    axxc.plot([xs, xs], [-0.05, ux], lw=1, c='grey')
    axxp.plot([xs, xs], [-0.05, 0.5], lw=1, c='grey')
    ax0.plot([xs, xs], [-6.5, 6.5], lw=1, c='grey')
    axyc.plot([uy, uy], [-6.5, ys], lw=1, c='grey')
    axyc.plot([-0.05, uy], [ys, ys], lw=1, c='grey')
    axyp.plot([-0.05, 0.7], [ys, ys], lw=1, c='grey')
    ax0.plot([-6.5, 6.5], [ys, ys], lw=1, c='grey')
    
    # plot circle at sampling point
    circ0 = Circle((xs, ys), 0.25, fc='goldenrod', zorder=10)
    ax0.add_patch(circ0)
    
    # annotate steps
    axxc.annotate('Step 1', (-6.5, ux), (-5.5, ux + 0.15), arrowprops={'arrowstyle': '->'}, usetex=True)
    axxc.annotate('Step 2', (xs, ux), (xs + 2, ux), arrowprops={'arrowstyle': '->'}, usetex=True)
    axyp.annotate('Step 3', (0.2, 1.3), (0.2, 4), arrowprops={'arrowstyle': '->'}, usetex=True)
    axyc.annotate('Step 4', (uy, -6.5), (uy + 0.1, -5.5), arrowprops={'arrowstyle': '->'}, usetex=True)
    axyc.annotate('Step 5', (uy, ys), (uy + 0.1, ys - 2), arrowprops={'arrowstyle': '->'}, usetex=True)
    
    # equations
    ax0.text(0.3, 0.4, r'$p(x, y)$', transform=ax0.transAxes, usetex=True)
    axxp.text(0.95, 0.5, r'$p(x)=\displaystyle\int dy\ p(x, y)$', va='center', transform=axxp.transAxes, usetex=True)
    axxc.text(0.95, 0.5, r"$F(x)=\displaystyle\int_{-\infty}^{y} dx'\ p(x')$", va='center', transform=axxc.transAxes, usetex=True)
    axyp.text(0.5, 1.01, r'$p(y | x=X)$', ha='center', transform=axyp.transAxes, usetex=True)
    axyc.text(0.5, 1.01, r"$F(y)=\displaystyle\int_{-\infty}^{y} dy'\ p(y'|x=X)$", ha='center', transform=axyc.transAxes, usetex=True)

    # axis limits
    for ax in [ax0, axxp, axxc]:
        ax.set_xlim(-6.5, 6.5)
    for ax in [ax0, axyp, axyc]:
        ax.set_ylim(-6.5, 6.5)
    axxp.set_ylim(-0.05, 0.5)
    axxc.set_ylim(-0.05, 1.05)
    axyp.set_xlim(-0.05, 0.7)
    axyc.set_xlim(-0.05, 1.05)
    
    # axis labels
    ax0.set_xlabel(r'$x$', usetex=True)
    ax0.set_ylabel(r'$y$', usetex=True)
    
    # remove extraneous spines and ticks/labels
    for ax in [ax0, axxp, axxc]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_bounds(-6, 6)
    for ax in [ax0, axyp, axyc]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(-6, 6)
    axxp.spines['left'].set_visible(False)
    axyp.spines['bottom'].set_visible(False)
    axxp.tick_params(left=False, labelleft=False, labelbottom=False, direction='inout')
    axyp.tick_params(bottom=False, labelbottom=False, labelleft=False, direction='inout')
    axxc.set_yticks([0, 0.5, 1])
    axxc.spines['left'].set_bounds(0, 1)
    axyc.spines['bottom'].set_bounds(0, 1)
    axxc.tick_params(axis='x', labelbottom=False, direction='inout')
    axyc.tick_params(axis='y', labelleft=False, direction='inout')

    # save
    fig.savefig("../source/assets/2D_sampling.png", bbox_inches='tight')

    plt.show()