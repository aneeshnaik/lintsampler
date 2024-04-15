import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('figstyle.mplstyle')


if __name__ == "__main__":
    
    x0 = np.linspace(-0.2, 1.2, 500)
    y0 = x0**3 - 4 * x0**2 + 6
    x1 = np.linspace(0, 1, 500)
    y1 = -x1**2 / 3 + 4 * x1 / 3
    
    HIST_BINS = np.linspace(0, 1, 11)
    
    N_SAMPLES = 1000
    rng = np.random.default_rng(42)
    u = rng.uniform(size=N_SAMPLES)
    xs = 2 - np.sqrt(4 - 3 * u)
    
    # fig setup
    fig = plt.figure(figsize=(6, 6.5))
    left = 0.08
    right = 0.99
    bottom = 0.08
    top = 0.99
    dX = right - left
    dY = (top - bottom) / 2
    ax0 = fig.add_axes([left, bottom, dX, dY])
    ax1 = fig.add_axes([left, bottom + dY, dX, dY])
    
    # plot PDF and interpolant and CDF
    ax0.plot(x0, y0, c='lightgrey', lw=2)
    ax0.plot([0, 1], [6, 3], c='teal', lw=2)
    ax1.plot(x1, y1, c='teal', lw=2)

    # edges = np.linspace(0, 1, 81)
    # cens = 0.5 * (edges[1:] + edges[:-1])
    # hist, edges = np.histogram(xs, density=True, bins=edges)
    # ax0.bar(cens, hist * 4.5, width=1/80)
    
    # initialise lines and bars
    line1, = ax1.plot([], [], lw=1, c='grey')
    line2, = ax1.plot([], [], lw=1, c='grey')
    line3, = ax0.plot([], [], lw=1, c='grey')
    _, _, bars = ax0.hist(np.array([]), HIST_BINS, lw=1, ec="none", fc="goldenrod", alpha=0.5)

    # axis labels
    ax0.set_xlabel(r'$x$', usetex=True)
    ax1.set_ylabel(r'CDF, $F(x)$', usetex=True)

    # axis lims
    ax0.set_xlim(-0.25, 1.25)
    ax1.set_xlim(-0.25, 1.25)
    ax0.set_ylim(0, 7)
    ax1.set_ylim(-0.05, 1.05)

    # ticks
    for ax in [ax0, ax1]:
        ax.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax0.tick_params(left=False, labelleft=False)
    ax1.tick_params(axis='x', direction='inout', labelbottom=False)
    
    # remove top/right spines
    for ax in [ax0, ax1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    
    # set x spine extent
    ax0.spines['bottom'].set_bounds(0, 1)
    ax1.spines['bottom'].set_bounds(0, 1)
    ax1.spines['left'].set_bounds(0, 1)
    
    def animate(i):
        n, _ = np.histogram(xs[:i], HIST_BINS)
        for count, rect in zip(n, bars.patches):
            rect.set_height(4.5 * count / N_SAMPLES / np.diff(HIST_BINS)[0])
        line1.set_data([-0.25, xs[i]], [u[i], u[i]])
        line2.set_data([xs[i], xs[i]], [-0.05, u[i]])
        line3.set_data([xs[i], xs[i]], [0, 7])
        return line1, line2, line3, *bars
    
    # create the animation
    fps = 15
    ani = FuncAnimation(fig, animate, frames=N_SAMPLES, blit=True, interval=1000/fps)

    # save
    ani.save('../source/assets/lint1Dunit_animation.gif', writer='pillow', fps=fps)
    
    plt.show()