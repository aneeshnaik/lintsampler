import numpy as np
from .utils import _calculate_cell_volumes, _calculate_cell_averages,\
    _get_cell_corners, _unitsample_kd


def choose_gridcells(*edgearrays, corner_densities,
                     N_cells=None, seed=None):
    """From k-dimensional grid of densities, choose mass-weighted cell(s).

    Given a k-dimensional grid, shaped (N0 x N1 x ... x N{k-1}), the user
    specifies a sequence of k 1D arrays (lengths N0+1, N1+1, etc.) representing
    the (not necessarily evenly spaced) gridlines along each dimension, and a
    kD array (shape N0+1 x N1+1 x ...) representing the densities at the grid
    corners. This function then calculates the mass of each grid cell according
    to the trapezoid rule (i.e., the average density over all 2**k corners times
    the volume of the cell), then randomly chooses a cell (or several cell) from
    the set of cells, weighting each cell by its mass in this choice.

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
        spaced.
    corner_densities : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    N_cells : int, optional
        Number of cells to sample. Default is None, in which case a single cell
        is returned (1D array of cell indices). Note that if N_cells is instead
        explicitly set to 1, then the cell is batched, so a 2-d (1xk) array of
        cell indices is returned.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.
        

    Returns
    -------
    idx : 2D (N_cells x k) or 1D (k,) numpy array of ints
        Indices along each dimension of randomly sampled cells. 2D if N_cells is
        set with an integer (including 1), 1D if N_cells is set to None.
    """
    # check shapes of edge arrays and density array all make sense
    if len(edgearrays) != corner_densities.ndim:
        raise ValueError("No. of edge arrays doesn't match density grid.")
    for i, a in enumerate(edgearrays):
        if a.ndim != 1:
            raise TypeError("Expected 1D edge arrays.")
        if len(a) != corner_densities.shape[i]:
            raise ValueError(
                "Length of edge array doesn't match corresponding "
                "density grid dimension."
            )
    
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # calculate cell volumes
    V = _calculate_cell_volumes(*edgearrays)
    
    # calculate cell average densities
    f_avg = _calculate_cell_averages(corner_densities)
    
    # mass = density * volume
    m = f_avg * V
    
    # normalise mass and flatten into probability array
    m_norm = m / m.sum()
    p = m_norm.flatten()

    # choose cells
    a = np.prod([s - 1 for s in corner_densities.shape])
    cells = rng.choice(a, p=p, size=N_cells)

    # unravel 1D cell indices into k-D grid indices
    idx = np.stack(np.unravel_index(cells, m_norm.shape), axis=-1)
    return idx


def sample(N, *edgearrays, corner_densities, seed=None):
    """Batch sample from density function defined on k-dimensional grid.
    
    Given a k-dimensional grid, shaped (N0 x N1 x ... x N{k-1}), the user
    specifies a sequence of k 1D arrays (lengths N0+1, N1+1, etc.) representing
    the (not necessarily evenly spaced) gridlines along each dimension, and a
    kD array (shape N0+1 x N1+1 x ...) representing the densities at the grid
    corners. This function then draws N samples from this grid. It first chooses
    N cells (with replacement) weighting them by their mass (estimated by the
    trapezoid rule) then samples from k-linear interpolant within each of the
    chosen cells.

    Parameters
    ----------
    N : int
        Number of samples to draw.
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
        spaced.
    corner_densities : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    samples : 2D numpy array, shape (N, k)
        Samples from linear interpolant on k-dimensional grid.
    """
    # check inputs all sensible
    if not isinstance(N, int):
        raise TypeError("Expected int N_samples")
    if len(edgearrays) != corner_densities.ndim:
        raise ValueError("No. of edge arrays doesn't match density grid.")
    for i, a in enumerate(edgearrays):
        if a.ndim != 1:
            raise TypeError("Expected 1D edge arrays.")
        if len(a) != corner_densities.shape[i]:
            raise ValueError(
                "Length of edge array doesn't match corresponding "
                "density grid dimension."
            )
    
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # randomly choose grid cells
    cells = choose_gridcells(
        *edgearrays,
        corner_densities=corner_densities, N_cells=N, seed=rng
    )
    
    # get array (shape 2 x 2 x ... x 2 x N_cells) of densities at cell corners
    f = _get_cell_corners(corner_densities, cells)

    # sample on unit hypercube
    samples = _unitsample_kd(f, rng)

    # rescale coordinates (loop over dimensions)
    for d in range(samples.shape[1]):
        e = edgearrays[d]
        c = cells[:, d]
        samples[:, d] = e[c] + np.diff(e)[c] * samples[:, d]

    return samples