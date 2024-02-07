from functools import reduce
import numpy as np

def calculate_cell_averages(corner_densities):
    """Given grid of densities, evaluated at corners, calculate cell averages.

    Parameters
    ----------
    corner_densities : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.

    Returns
    -------
    average : k-dim numpy array, shape (N0 x N1 x ... x N{k-1})
        Density in each grid cell, averaging over 2**k corners. Shape is shape
        of grid.
      
    """
    # infer dimensionality and grid shape
    ndim = corner_densities.ndim
    shape = tuple([s - 1 for s in corner_densities.shape])

    # loop over corners, add density contribution to sum
    # at each corner construct slice tuple, e.g. a[:-1,:-1,etc] for 1st corner
    sum = np.zeros(shape)
    slice0 = slice(-1)
    slice1 = slice(1, None)
    for i in range(2**ndim):
        n = np.binary_repr(i, width=ndim)
        t = ()
        for d in range(ndim):
            t += ([slice0, slice1][int(n[d])],)
        sum += corner_densities[t]
    
    # div. by no. corners for average
    average = sum / 2**ndim
    return average


def calculate_cell_volumes(*edgearrays):
    """From a sequence of arrays of edge lines, calculate grid cell volumes.
    
    Calculates difference arrays with numpy.diff, then volumes with outer
    product.

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively.
    
    Returns
    -------
    vols : numpy array, k-dimensional, shape (N0 x N1 x ... x N{k-1})
        k-dimensional array containing volumes of grid cells. Shape is shape of
        grid.
    """
    diffarrays = []
    for edgearray in edgearrays:
        diffarrays.append(np.diff(edgearray))
    shape = tuple([d.size for d in diffarrays])
    vols = reduce(np.outer, diffarrays).reshape(shape)
    return vols


def choose_gridcells(*edgearrays, corner_densities,
                     N_cells=None, seed=None):
    """From k-dimensional grid of densities, choose mass-weighted cell(s).

    DESCRIPTION.

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively.
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
    V = calculate_cell_volumes(*edgearrays)
    
    # calculate cell average densities
    f_avg = calculate_cell_averages(corner_densities)
    
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
