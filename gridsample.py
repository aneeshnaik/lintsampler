import numpy as np
from functools import reduce
from .unitsample_kd import _unitsample_kd


def gridsample(*edgearrays, f, N_samples=None, seed=None):
    """Batch sample from density function defined on k-dimensional grid.
    
    Given a k-dimensional grid, shaped (N0 x N1 x ... x N{k-1}), the user
    specifies a sequence of k 1D arrays (lengths N0+1, N1+1, etc.) representing
    the (not necessarily evenly spaced) gridlines along each dimension, and a
    kD array (shape N0+1 x N1+1 x ...) representing the (not necessarily
    normalised) densities at the grid corners. This function then draws a sample
    (or N samples) from this grid. It first chooses a cell (or N cells with
    replacement), weighting them by their mass (estimated by the trapezoid rule)
    then samples from k-linear interpolant within the chosen cell(s).

    Parameters
    ----------
    *edgearrays : 1 or more 1D numpy arrays
        k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
        is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
        (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
        spaced.
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    N_samples : int, optional
        Number of samples to draw. Default is None, in which case single sample
        is drawn.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
        Sample(s) from linear interpolant. Scalar if single sample (i.e.,
        N_samples is None) in 1D. 1D array if single sample in k-D OR multiple
        samples in 1D. 2D array if multiple samples in k-D.
    """
    # check inputs all sensible
    if len(edgearrays) != f.ndim:
        raise ValueError("No. of edge arrays doesn't match density grid.")
    for i, a in enumerate(edgearrays):
        if a.ndim != 1:
            raise TypeError("Expected 1D edge arrays.")
        if len(a) != f.shape[i]:
            raise ValueError(
                "Length of edge array doesn't match corresponding "
                "density grid dimension."
            )
    
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # randomly choose grid cell(s)
    cells = _gridcell_choice(*edgearrays, f=f, N_cells=N_samples, seed=rng)
    
    # get 2^k-tuple of densities at cell corners
    corners = _gridcell_corners(f, cells)

    # sample on unit hypercube
    z = _unitsample_kd(*corners, seed=rng)

    # rescale coordinates (loop over dimensions)
    for d in range(f.ndim):
        e = edgearrays[d]
        if N_samples:
            c = cells[:, d]
            z[:, d] = e[c] + np.diff(e)[c] * z[:, d]
        else:
            c = cells[d]
            z[d] = e[c] + np.diff(e)[c] * z[d]

    # squeeze down to scalar / 1D if appropriate
    if not N_samples and (f.ndim == 1):
        z = z.item()
    elif (f.ndim == 1):
        z = z[:, 0]

    return z


def _gridcell_faverages(f):
    """Given grid of densities, evaluated at corners, calculate cell averages.

    Parameters
    ----------
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.

    Returns
    -------
    average : k-dim numpy array, shape (N0 x N1 x ... x N{k-1})
        Density in each grid cell, averaging over 2**k corners. Shape is shape
        of grid.
      
    """
    # infer dimensionality and grid shape
    ndim = f.ndim
    shape = tuple([s - 1 for s in f.shape])

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
        sum += f[t]
    
    # div. by no. corners for average
    average = sum / 2**ndim
    return average


def _gridcell_volumes(*edgearrays):
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


def _gridcell_choice(*edgearrays, f, N_cells=None, seed=None):
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
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
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
    if len(edgearrays) != f.ndim:
        raise ValueError("No. of edge arrays doesn't match density grid.")
    for i, a in enumerate(edgearrays):
        if a.ndim != 1:
            raise TypeError("Expected 1D edge arrays.")
        if len(a) != f.shape[i]:
            raise ValueError(
                "Length of edge array doesn't match corresponding "
                "density grid dimension."
            )
    
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # calculate cell volumes
    V = _gridcell_volumes(*edgearrays)
    
    # calculate cell average densities
    f_avg = _gridcell_faverages(f)
    
    # mass = density * volume
    m = f_avg * V
    
    # normalise mass and flatten into probability array
    m_norm = m / m.sum()
    p = m_norm.flatten()

    # choose cells
    a = np.prod([s - 1 for s in f.shape])
    cells = rng.choice(a, p=p, size=N_cells)

    # unravel 1D cell indices into k-D grid indices
    idx = np.stack(np.unravel_index(cells, m_norm.shape), axis=-1)
    return idx


def _gridcell_corners(f, cells):
    """From gridded densities, get densities on 2^k corners of given cells.

    Parameters
    ----------
    f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    cells : 2-d numpy array, shape (N, k)
        Grid indices of N chosen cells along the k dimensions of the grid.
        
    Returns
    -------
    corners : 2^k-tuple of 1D numpy arrays, each length N
        Densities at corners of given cells. Conventional ordering applies,
        e.g., in 3D: corners = (f000, f001, f010, f011, f100, f101, f110, f111).

    """
    # infer dimensionality from shape of corner density grid
    ndim = f.ndim
    
    # get loop over 2^k corners, get densities at each
    corners = []
    for i in range(2**ndim):
        
        # binary representation of corner, e.g. [0,0,...,0] is first corner
        n = np.binary_repr(i, width=ndim)
        n = np.array([int(c) for c in n], dtype=int)
 
        # get densities on given corners
        idx = cells + n
        idx = np.split(idx.T, ndim)
        idx = tuple([idxi.squeeze() for idxi in idx])
        corners.append(f[idx])
    return tuple(corners)
