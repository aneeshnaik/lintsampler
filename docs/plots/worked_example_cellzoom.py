import numpy as np

import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

F00 = 4.653893201041552
F01 = 9.649159214322465
F10 = 3.7786650410289577
F11 = 11.884132617130431


def get_plot_data(N):
    
    X0, Y0 = (-1, -2)
    X1, Y1 = (-0.5, -1)
    
    X_edges = np.linspace(X0, X1, N + 1)
    X_cens = 0.5 * (X_edges[1:] + X_edges[:-1])
    Y_edges = np.linspace(Y0, Y1, N + 1)
    Y_cens = 0.5 * (Y_edges[1:] + Y_edges[:-1])
    X, Y = np.meshgrid(X_cens, Y_cens, indexing='ij')
    finterp = (F00*(X1-X)*(Y1-Y) + F01*(X1-X)*(Y-Y0) + F10*(X-X0)*(Y1-Y) + F11*(X-X0)*(Y-Y0)) / ((X1-X0) * (Y1-Y0))
    return X, Y, finterp


if __name__=="__main__":
    
    N_plot = 512
    X0, Y0, finterp0 = get_plot_data(N_plot)
    
    fig =  plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.3, 0.1, 0.4, 0.8])    

    # plot contour map
    pcm = ax.contourf(X0, Y0, finterp0, levels=20, cmap='magma')

    # label corner densities
    propw = dict(arrowstyle="-|>,head_width=0.2,head_length=0.4",
            shrinkA=0,shrinkB=0, fc='w', ec='w')
    ax.annotate(
        r"$f_{00}=$ "+f"{F00:.2f}",
        xy=(-1, -2), xytext=(-0.9, -1.85),
        arrowprops=propw, usetex=True, c='w', ha='center'
    )
    ax.annotate(
        r"$f_{01}=$ "+f"{F01:.2f}",
        xy=(-1, -1), xytext=(-0.9, -1.25),
        arrowprops=propw, usetex=True, c='w', ha='center'
    )
    ax.annotate(
        r"$f_{10}=$ "+f"{F10:.2f}",
        xy=(-0.5, -2), xytext=(-0.6, -1.85),
        arrowprops=propw, usetex=True, c='w', ha='center'
    )
    ax.annotate(
        r"$f_{11}=$ "+f"{F11:.2f}",
        xy=(-0.5, -1), xytext=(-0.6, -1.25),
        arrowprops=propw, usetex=True, c='w', ha='center'
    )

    # spine settings
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_bounds(-2, -1)
    ax.spines['bottom'].set_bounds(-1, -0.5)
    ax.spines['left'].set_position(('data', -1.05))
    ax.spines['bottom'].set_position(('data', -2.05))

    # tick settings
    ax.set_xticks([-1, -0.5])
    ax.set_yticks([-2, -1])
    
    # save
    fig.savefig("../source/assets/worked_example_cellzoom.png", bbox_inches='tight')
    
    plt.show()