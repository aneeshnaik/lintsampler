import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

if __name__ == "__main__":
    
    x = np.linspace(-0.2, 1.2, 500)
    y = x**3 - 4 * x**2 + 6

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # plot PDF and interpolant
    ax.plot(x, y, c='lightgrey', lw=2)
    ax.plot([0, 1], [6, 3], c='teal', lw=2)
    
    # line annotations
    ax.annotate("True density $f(x)$", (0.65, 4.7), (0.7, 5.3), arrowprops={'arrowstyle': '->'}, usetex=True)
    ax.annotate(r"Linear interpolant $\hat{f}(x)$", (0.5, 4.4), (0.15, 3), arrowprops={'arrowstyle': '->'}, usetex=True)

    # axis labels
    ax.set_xlabel(r'$x$', usetex=True)
    ax.set_ylabel('(Unnormalised) Density')

    # axis lims
    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0, 7)

    # ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 3, 6])
    
    # remove top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set x spine extent
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(0, 6)
    
    # save
    fig.savefig("../source/assets/lint1Dunit_pdf.png", bbox_inches='tight')

    plt.show()