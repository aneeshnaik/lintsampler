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

def multiply_array_slice(arr, factor, axis, idx):
    """Multiply (inplace) subslice of array by given factor along given slice.
    
    Parameters
    ----------
    arr : n-dimensional numpy array
        Array to be multiplied.
    factor : float or array broadcastable with given slice
        Factor to multiply array slice by.
    axis : int
        Dimension along which to take slice
    idx : int
        Index at which to take slice along specified axis.
    
    Returns
    -------
    None

    """
    I = [slice(None)] * arr.ndim
    I[axis] = idx
    arr[tuple(I)] *= factor
    return

def unitsample_1d(f0, f1, seed=None):
    """Batched sampling from linear interpolant between x=0 and 1.
    
    f0 and f1 are numpy arrays length N, representing the density at x=0 and x=1
    respectively. N samples are drawn from the linear interpolant between the
    two points. This works as follows: first, u ~ Uniform(0, 1), then:
    where f0 == f1: samples = u
    elsewhere: samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)

    Parameters
    ----------
    f0 : 1D numpy array, same length as f1
        Array of densities at x=0. 
    f1 : 1D numpy array, same length as f0
        Array of densities at x=1.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    samples : 1D array, same length as f0/f1
        Array of samples from interpolant.
    """
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # no. samples
    N = f0.size
    
    # generate uniform samples
    u = rng.uniform(size=N)
    
    # where f0 == f1: samples = u
    # elsewhere, samples = (-f0 + sqrt(f0^2 + (f1^2-f0^2)*u)) / (f1-f0)
    m = f0 == f1
    if m.any():
        z = np.copy(u)
        u = u[~m]
        f0 = f0[~m]
        f1 = f1[~m]
        z[~m] = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    else:
        z = (-f0 + np.sqrt(f0**2 + (f1**2 - f0**2) * u)) / (f1 - f0)
    return z


def unitsample_kd(f, seed=None):
    """Batched sampling from linear interpolant in k-dimensional unit cube.
    
    f is a (k+1)-dimensional numpy array, shaped (2 x 2 x ... x 2 x N),
    comprising the density values at the 2^k corners of a batch of N k-d unit
    cubes. Samples are drawn from the n-linear interpolant between these
    corners. This works by first 1D sampling from p(x0), then 1D sampling
    from p(x1|x0), then p(x2|x0,x1), and so on until all k dimensions are
    sampled.

    Parameters
    ----------
    f : (k+1)-dimensional numpy array, shape (2 x 2 x ... x 2 x N)
        Densities at corners of N k-d unit cubes.
    seed : None/int/numpy random Generator, optional
        Seed for numpy random generator. Can be random generator itself, in
        which case it is left unchanged. Default is None, in which case new
        default generator is created. See numpy random generator docs for more
        information.

    Returns
    -------
    samples : 2D array, shape (N x k)
        Array of samples from n-linear interpolant.
    """
    # prepare RNG
    rng = np.random.default_rng(seed)
    
    # infer dimensionality
    ndim = f.ndim - 1
    
    # set up empty array for samples
    samples = np.zeros((f.shape[-1], ndim))

    # loop over dims, starting with first
    # at each dim sample from p(current dim | previous dims)
    for d in range(ndim):

        # construct 'reduced' hypercube, averaging over higher dims
        freduced = np.average(f, axis=tuple(range(d + 1, ndim)))

        # loop over dims lower than current
        # at each dim multiply slice 0 by (1-xi) and slice 1 by xi
        for i in range(d):     
            multiply_array_slice(freduced, (1 - samples[:, i]), i, 0)
            multiply_array_slice(freduced, samples[:, i], i, 1)

        # sum up to current dim and take slices
        f0, f1 = np.sum(freduced, axis=tuple(range(d)))

        # get samples
        samples[:, d] = unitsample_1d(f0, f1, rng)
    
    return samples

def get_cellcorners(corner_densities, cells):
    """From gridded densities, get densities on corners of given cells.

    Parameters
    ----------
    corner_densities : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
        Grid of densities evaluated at corners of k-dimensional grid.
    cells : 2-d numpy array, shape (N, k)
        Grid indices of N chosen cells along the k dimensions of the grid.
        
    Returns
    -------
    f : (k+1)-dim numpy array, shape (2 x 2 x ... x 2 x N)
        Densities at corners of given cells.
    """
    # infer dimensionality from shape of corner density grid
    ndim = corner_densities.ndim
    
    # get array of densities cell corners, initially has shaped (2^k, N_samples)
    f = []
    for i in range(2**ndim):
        n = np.binary_repr(i, width=ndim)
        n = np.array([int(c) for c in n], dtype=int)
        idxi = cells + n
        f.append(corner_densities[*idxi.T])
    f = np.stack(f)
    
    # reshape into (2 x ... x 2 x N_samples)
    f = f.reshape(((2,) * ndim) + (cells.shape[0],))
    return f


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
    f = get_cellcorners(corner_densities, cells)

    # sample on unit hypercube
    samples = unitsample_kd(f, rng)

    # rescale coordinates (loop over dimensions)
    for d in range(samples.shape[1]):
        e = edgearrays[d]
        c = cells[:, d]
        samples[:, d] = e[c] + np.diff(e)[c] * samples[:, d]

    return samples