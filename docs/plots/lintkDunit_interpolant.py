import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')

def fn(x, y):
    return 10 - 2 * x**2 - 2 * y**2

if __name__ == "__main__":

    asp = 6 / 3.3
    fig = plt.figure(figsize=(6, 6 / asp))
    left = 0.08
    bottom = 0.2
    right = 0.85
    hgap = 0.03
    cgap = 0.03
    dX = (right - left - hgap) / 2
    dY = asp * dX
    cdX = 0.1 * dX
    ax0 = fig.add_axes([left, bottom, dX, dY])
    ax1 = fig.add_axes([left + dX + hgap, bottom, dX, dY])
    cax = fig.add_axes([left + 2*dX + hgap + cgap, bottom, cdX, dY])
    
    Ngrid = 32
    edges = np.linspace(0, 1, Ngrid)
    Xgrid, Ygrid = np.meshgrid(edges, edges, indexing='ij')
    
    f = fn(Xgrid, Ygrid)
    
    f00 = fn(0, 0)
    f01 = fn(0, 1)
    f10 = fn(1, 0)
    f11 = fn(1, 1)
    fint = f00 * (1 - Xgrid) * (1 - Ygrid) + f01 * (1 - Xgrid) * Ygrid + f10 * Xgrid * (1 - Ygrid) + f11 * Xgrid * Ygrid
   
    print(f00, f01, f10, f11)
   
    levels = np.linspace(6, 10, 21)
    fc = ax0.contourf(Xgrid, Ygrid, f, levels=levels, cmap='magma', vmin=6, vmax=10)
    fc = ax1.contourf(Xgrid, Ygrid, fint, levels=levels, cmap='magma', vmin=6, vmax=10) 
    plt.colorbar(fc, cax=cax)

    ax0.set_title("True Density", fontsize=12)
    ax1.set_title("Bilinear Interpolant", fontsize=12)
    
    for ax in [ax0, ax1]:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['bottom', 'left']:
            ax.spines[side].set_bounds(0, 1)
    ax1.spines['left'].set_visible(False)
    ax0.spines[['left', 'bottom']].set_position(('data', -0.05))
    ax1.spines[['bottom']].set_position(('data', -0.05))
    ax1.tick_params(left=False, labelleft=False, labelbottom=False)
    ax0.set_ylabel(r"$y$", usetex=True)
    ax0.set_xlabel(r"$x$", usetex=True)
    ax0.set_yticks([0, 1])
    ax0.set_xticks([0, 1])
    ax1.set_xticks([0, 1])
    cax.set_yticks([6, 10])
    cax.set_ylabel("Density")

    # save
    fig.savefig("../source/assets/lintkDunit_interpolant.png", bbox_inches='tight')
    
    plt.show()