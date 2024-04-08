import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
plt.style.use('figstyle.mplstyle')


if __name__ == "__main__":
    
    N_SAMPLES = 1000
    HIST_BINS = np.linspace(-6, 6, 65)

    rng = np.random.default_rng(42)
    u = rng.uniform(size=N_SAMPLES)
    xs = norm.ppf(u)
    
    
    x_pdf = np.linspace(-6, 6, 500)
    y_pdf = norm.pdf(np.linspace(-6, 6, 500))
    y_cdf = norm.cdf(x_pdf)
    
    fig = plt.figure(figsize=(6, 7))
    axb = fig.add_subplot([0.1, 0.1, 0.8, 0.4])
    axt = fig.add_subplot([0.1, 0.5, 0.8, 0.4])

    axb.plot(x_pdf, y_pdf, c='teal', lw=2)
    axt.plot(x_pdf, y_cdf, c='teal', lw=2)
    
    # initialise lines
    line1, = axt.plot([], [], lw=1, c='grey')
    line2, = axt.plot([], [], lw=1, c='grey')
    line3, = axb.plot([], [], lw=1, c='grey')
    
    _, _, bars = axb.hist(np.array([]), HIST_BINS, lw=1, ec="none", fc="goldenrod", alpha=0.5)
    axb.set_xlim(-6.6, 6.6)
    axt.set_xlim(-6.6, 6.6)
    axb.set_ylim(-0.02, 0.42)
    axt.set_ylim(-0.05, 1.05)

    # set x/y ticks
    axt.set_xticks(np.arange(-6, 6+2, 2))
    axb.set_xticks(np.arange(-6, 6+2, 2))
    axt.set_yticks([0, 0.5, 1])
    
    # remove y axis labels/ticks
    axb.tick_params(left=False, labelleft=False)
    axt.tick_params(labelbottom=False, axis='x', direction='inout')
    
    # remove extraneous spines
    axt.spines['top'].set_visible(False)
    axt.spines['right'].set_visible(False)
    axb.spines['top'].set_visible(False)
    axb.spines['left'].set_visible(False)
    axb.spines['right'].set_visible(False)
    
    # set spine extents
    axt.spines['bottom'].set_bounds(-6, 6)
    axb.spines['bottom'].set_bounds(-6, 6)
    axt.spines['left'].set_bounds(0, 1)
    
    def animate(i):
        n, _ = np.histogram(xs[:i], HIST_BINS)
        for count, rect in zip(n, bars.patches):
            rect.set_height(count / N_SAMPLES / np.diff(HIST_BINS)[0])
        line1.set_data([-6.6, xs[i]], [u[i], u[i]])
        line2.set_data([xs[i], xs[i]], [-0.05, u[i]])
        line3.set_data([xs[i], xs[i]], [-0.02, 0.42])
        return line1, line2, line3, *bars.patches

    # Create the animation
    fps = 10
    ani = FuncAnimation(fig, animate, frames=N_SAMPLES, blit=True, interval=1000/fps)

    ani.save('../source/assets/1D_animation.gif', writer='pillow', fps=fps)

    # Show the animation
    plt.show()
