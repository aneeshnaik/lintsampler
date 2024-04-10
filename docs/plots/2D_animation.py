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


def generate_samples(mean, cov, N):
    
    rng = np.random.default_rng(42)
    ux = rng.uniform(size=N)
    uy = rng.uniform(size=N)
    
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
    N_SAMPLES = 100
    ux, uy, xs, ys = generate_samples(mean, cov, N_SAMPLES)
    
    # conditional PDF params
    muy = mean[1] + (cov[0, 1] / cov[0, 0]) * (xs - mean[0])
    sigy = np.sqrt(cov[1, 1] - cov[0, 1]**2 / cov[0, 0])
    
    # figure
    fig = plt.figure(figsize=(10, 10))
    left = 0.05
    bottom = 0.05
    right = 0.99
    dX = (right - left) / 2
    dY = dX
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

    # initial artists
    pcoll = ax0.scatter([], [], s=2, c='goldenrod')
    line1, = axxc.plot([], [], lw=1, c='grey')
    line2, = axxc.plot([], [], lw=1, c='grey')
    line3, = axxp.plot([], [], lw=1, c='grey')
    line4, = ax0.plot([], [], lw=1, c='grey')
    ypdfline, = axyp.plot([], [], c='teal', lw=2)
    ycdfline, = axyc.plot([], [], c='teal', lw=2)
    line5, = axyc.plot([], [], lw=1, c='grey')
    line6, = axyc.plot([], [], lw=1, c='grey')
    line7, = axyp.plot([], [], lw=1, c='grey')
    line8, = ax0.plot([], [], lw=1, c='grey')
    circ = Circle((0, 0), 0.25, fc='none', zorder=10)
    ax0.add_patch(circ)
    
    # axis limits
    for ax in [ax0, axxp, axxc]:
        ax.set_xlim(-6.5, 6.5)
    for ax in [ax0, axyp, axyc]:
        ax.set_ylim(-6.5, 6.5)
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
    axyc.set_xticks([0, 0.5, 1])
    axxc.spines['left'].set_bounds(0, 1)
    axyc.spines['bottom'].set_bounds(0, 1)
    axxc.tick_params(axis='x', labelbottom=False, direction='inout')
    axyc.tick_params(axis='y', labelleft=False, direction='inout')

    def animate(i):
        si = i//5
        if i % 5 == 0:
            circ.set_facecolor('none')
            line1.set_data([-6.5, xs[si]], [ux[si], ux[si]])
            line2.set_data([], [])
            line3.set_data([], [])
            line4.set_data([], [])
            ypdfline.set_data([], [])
            ycdfline.set_data([], [])
            line5.set_data([], [])
            line6.set_data([], [])
            line7.set_data([], [])
            line8.set_data([], [])
        elif i % 5 == 1:
            line1.set_data([-6.5, xs[si]], [ux[si], ux[si]])
            line2.set_data([xs[si], xs[si]], [-0.05, ux[si]])
            line3.set_data([xs[si], xs[si]], axxp.get_ylim())
            line4.set_data([xs[si], xs[si]], [-6.5, 6.5])
        elif i % 5 == 2:
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            ypdfline.set_data(norm.pdf(x, loc=muy[si], scale=sigy), x)
            ycdfline.set_data(norm.cdf(x, loc=muy[si], scale=sigy), x)
        elif i % 5 == 3:
            line4.set_data([xs[si], xs[si]], [-6.5, 6.5])
            line5.set_data([uy[si], uy[si]], [-6.5, ys[si]])
        elif i % 5 == 4:
            line1.set_data([], [])
            line2.set_data([], [])
            line3.set_data([], [])
            line4.set_data([xs[si], xs[si]], [-6.5, 6.5])
            line6.set_data([-0.05, uy[si]], [ys[si], ys[si]])
            line7.set_data(axyp.get_xlim(), [ys[si], ys[si]])
            line8.set_data([-6.5, 6.5], [ys[si], ys[si]])
            circ.set_center((xs[si], ys[si]))
            circ.set_facecolor('goldenrod')
        pcoll.set_offsets(np.stack((xs[:si], ys[:si]), axis=-1))
        return line1, line2, line3, line4, line5, line6, line7, line8, circ, ypdfline, ycdfline, pcoll

    fps = 3
    ani = FuncAnimation(
        fig, animate, frames=5*N_SAMPLES, blit=True, interval=1000/fps
    )
    
    # save
    ani.save('../source/assets/2D_animation.gif', writer='pillow', fps=fps)
    
    plt.show()
