import numpy as np
import matplotlib.pyplot as plt
plt.style.use('figstyle.mplstyle')
f00 = 10
f10 = 8
f01 = 8
f11 = 6

def f_bilinear(X, Y, X0, X1, Y0, Y1):
    t1 = f00 * (X1 - X) * (Y1 - Y)
    t2 = f01 * (X1 - X) * (Y - Y0)
    t3 = f10 * (X - X0) * (Y1 - Y)
    t4 = f11 * (X - X0) * (Y - Y0)
    return (t1 + t2 + t3 + t4) / ((X1 - X0)*(Y1 - Y0))


if __name__ == "__main__":

    asp = 6 / 3.6
    fig = plt.figure(figsize=(6, 6 / asp))
    left = 0.08
    hgap = 0.1
    bottom = 0.11
    right = 0.98
    dX = (right - left - 2 * hgap) / 3
    dY = asp * dX
    
    ax0 = fig.add_axes([left, bottom, dX, 2 * dY])
    ax1 = fig.add_axes([left + dX + hgap, bottom + 0.5 * dY, dX, dY])
    ax2 = fig.add_axes([left + 2 * (dX + hgap), bottom, dX, 2 * dY])
    
    Ngrid = 32
    xedges = np.linspace(40, 50, Ngrid)
    yedges = np.linspace(180, 200, Ngrid)
    zedges = np.linspace(0, 1, Ngrid)
    
    Xgrid, Ygrid = np.meshgrid(xedges, yedges, indexing='ij')
    Z1grid, Z2grid = np.meshgrid(zedges, zedges, indexing='ij')

    fxy = f_bilinear(Xgrid, Ygrid, 40, 50, 180, 200)
    fzz = f_bilinear(Z1grid, Z2grid, 0, 1, 0, 1)
    
    levels = np.linspace(6, 10, 21)
    fc = ax0.contourf(Xgrid, Ygrid, fxy, levels=levels, cmap='magma', vmin=6, vmax=10)
    fc = ax1.contourf(Z1grid, Z2grid, fzz, levels=levels, cmap='magma', vmin=6, vmax=10) 
    fc = ax2.contourf(Xgrid, Ygrid, fxy, levels=levels, cmap='magma', vmin=6, vmax=10)
    
    ax1.scatter([0.25], [0.25], s=20, c='skyblue')
    ax2.scatter([42.5], [185], s=20, c='skyblue')
    
    for ax in [ax0, ax1, ax2]:
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    
    for ax in [ax0, ax2]:
        ax.set_xlim(40, 50)
        ax.set_ylim(180, 200)
        ax.spines['bottom'].set_position(('data', 179.5))
        ax.spines['left'].set_position(('data', 39.5))
        ax.set_yticks([180, 200])
        ax.set_xticks([40, 50])
        ax.set_xlabel(r"$x_0$", usetex=True)
        ax.set_ylabel(r"$x_1$", usetex=True, rotation=0)
        ax.xaxis.set_label_coords(0.5, -0.08)
        ax.yaxis.set_label_coords(-0.2, 0.5)
    ax1.set_xlim(0, 1)
    ax1.set_xlim(0, 1)
    ax1.spines[['left', 'bottom']].set_position(('data', -0.05))
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xlabel(r"$z_0$", usetex=True)
    ax1.set_ylabel(r"$z_1$", usetex=True, rotation=0)
    ax1.xaxis.set_label_coords(0.5, -0.16)
    ax1.yaxis.set_label_coords(-0.2, 0.5)
    
    # save
    fig.savefig("../source/assets/lint_transformation.png", bbox_inches='tight')

    plt.show()