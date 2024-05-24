import numpy as np
from functools import reduce
from warnings import warn
from .utils import _is_1D_iterable, _choice


class DensityGrid:
    """Grid-like object over which density function is evaluated.

    #TODO extended description
    #TODO: examples

    Parameters
    ----------
    cells : 1D iterable or tuple of 1D iterables
        If a single 1D iterable (i.e., array, tuple, list of numbers), then this
        represents the cell edges of a 1D grid. So a 1D grid with N cells should
        have `cells` parameter as a 1D array with N+1 elements. If a tuple of 1D
        iterables, then this represents the edges values for a grid with
        dimensionality equal to the length of the tuple. So, a kD grid with
        (N0 x N1 x ... x N{k-1}) cells should have a length-k tuple, with arrays
        length N0+1, N1+1 etc.

    Attributes
    ----------
    dim : int
        Dimensionality of grid (referred to as "k" elsewhere in the docs).
    shape : tuple
        Length `dim` tuple giving grid shape. For example, if `edgearrays` are
        lengths N0+1, N1+1, ... N{k-1}+1, then `shape` is (N0, N1, ..., N{k-1}).
    ncells : int
        Total number of cells in grid, i.e., the product over the shape tuple.
    edgearrays : list
        Length `dim` list of arrays of grid edges along each dimension.
    edgeshape : tuple
        Length `dim` tuple giving lengths of `edgearrays`. This is equal to
        `shape` tuple, but plus one to each element.
    mins : numpy array
        1D numpy array (length `dim`), giving lower coordinate bound of grid
        along each dimension.
    maxs : numpy array
        1D numpy array (length `dim`), giving upper coordinate bound of grid
        along each dimension.
    densities_evaluated : bool
        Flag indicating whether density function has been evaluated on grid (via
        `evaluate` method) yet. 
    vertex_densities : {None, numpy array}
        If `densities_evaluated`, k-dimensional array giving densities at
        vertices of grid. Shape is equal to `edgeshape` attribute, i.e.
        (N0+1, N1+1, ...) if grid has shape (N0, N1, ...). None if not 
        `densities_evaluated`.
    masses : {None, numpy array}
        If `densities_evaluated`, k-dimensional array of probability masses of
        grid cells. Shape is equal to grid shape (see `shape` attribute). Masses
        are calculated according to trapezoid rule, i.e., cell volumes
        multiplied by average vertex densities. None if not 
        `densities_evaluated`.
    total_mass : {None, float}
        If `densities_evaluated`, total probability mass of this grid; sum over
        `masses` array. None if not `densities_evaluated`.

    Methods
    -------
    evaluate(pdf, vectorizedpdf=False, pdf_args=(), pdf_kwargs={})
        Evaluate given PDF on grid.
    choose(u)
        Given array of uniform (~U(0, 1)) samples, choose series of grid cells.
    get_cell_corner_densities(cells)
        Get densities at 2^k corners of given cells.
    """
    def __init__(self, cells):
        
        # 1D case: cells is 1D iterable (array, tuple, list)
        # e.g. cells = np.linspace(-4,4,50)
        if _is_1D_iterable(cells):

            # store dimensionality (1) and single edgearray and dims
            self.dim = 1
            self.edgearrays = [np.array(cells)]
            self.edgeshape = (len(cells),)

        # kD case: cells is tuple of 1D iterables
        # e.g. cells = (np.linspace(-12,12,100),np.linspace(-4,4,50))
        elif isinstance(cells, tuple) and _is_1D_iterable(cells[0]):
            
            # infer dimensionality
            self.dim = len(cells)

            # loop over dimensions, store edge arrays and dims
            self.edgearrays = []
            self.edgeshape = ()
            for d in range(0,self.dim):
                self.edgearrays.append(np.array(cells[d]))
                self.edgeshape += (len(cells[d]),)
        
        # cells type not recognised
        else:
            raise TypeError(
                "Grid.__init__: "\
                "you must specify an evaluation domain with a single 1D "\
                "iterable or a tuple of 1D iterables. "\
                "See documentation for details."
            )
        
        # check edge arrays all monotonically increasing and finite-valued
        for arr in self.edgearrays:
            if np.any(np.diff(arr) <= 0):
                raise ValueError(
                    "Grid.__init__: Edges not monotically increasing."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Grid.__init__: Edges not finite-valued."
                )

        # geometry-related attrs
        self.mins = np.array([arr[0] for arr in self.edgearrays])
        self.maxs = np.array([arr[-1] for arr in self.edgearrays])
        self.shape = tuple(d - 1 for d in self.edgeshape)
        self.ncells = np.prod(self.shape)

        # density-related attrs (None because densities not yet evaluated)
        self.reset_densities()

    def reset_densities(self):
        """Unset density flag and remove density-related attributes.

        Returns
        -------
        None
        """
        self.densities_evaluated = False
        self.vertex_densities = None
        self.masses = None
        self.total_mass = None

    def evaluate(self, pdf, vectorizedpdf=False, pdf_args=(), pdf_kwargs={}):
        """Evaluate the user-provided pdf on grid and set related attributes.
        
        Parameters
        ----------
        pdf : function
            Probability density function to evaluate on grid. Function should
            take coordinate vector (or batch of vectors if vectorized; see
            `vectorizedpdf` parameter) and return (unnormalised) density (or
            batch of densities). Additional arguments can be passed to the
            function via `pdf_args` and `pdf_kwargs` parameters.
        vectorizedpdf : bool, optional
            if True, assumes that the pdf function is vectorized, i.e., it
            accepts  (..., k)-shaped batches of coordinate vectors and returns
            (...)-shaped batches of densities. If False, assumes that the pdf
            function simply accepts (k,)-shaped coordinate vectors and returns
            single densities. Default is False.
        pdf_args : tuple, optional
            Additional positional arguments to pass to pdf function; function
            call is `pdf(position, *pdf_args, **pdf_kwargs)`. Default is empty
            tuple (no additional positional arguments).
        pdf_kwargs : dict, optional
            Additional keyword arguments to pass to pdf function; function call
            is `pdf(position, *pdf_args, **pdf_kwargs)`. Default is empty dict
            (no additional keyword arguments).
        
        Returns
        -------
        None
        """
        if self.densities_evaluated:
            warn("Grid.evaluate: PDF being unnecessarily evaluated again.")
            self.reset_densities()

        # number of vertices in grid
        npts = np.prod(self.edgeshape)
        
        # create the flattened grid for evaluation in k>1 case
        if self.dim > 1:
            g = np.stack(np.meshgrid(*self.edgearrays, indexing='ij'), axis=-1)
            edgegrid = g.reshape(npts, self.dim)
        else:
            edgegrid = self.edgearrays[0]

        # evaluate PDF over edge grid (iterate over grid if not vectorized)
        if vectorizedpdf:
            f = pdf(edgegrid, *pdf_args, **pdf_kwargs)
            densities = np.array(f, dtype=np.float64)         
        else:            
            densities = np.zeros(npts, dtype=np.float64)
            for pt in range(npts):
                f = pdf(edgegrid[pt], *pdf_args, **pdf_kwargs)
                densities[pt] = f

        # check densities all non-negative and finite
        if np.any(densities < 0):
            raise ValueError("Grid._evaluate_pdf: Densities can't be negative")
        if not np.all(np.isfinite(densities)):
            raise ValueError("Grid._evaluate_pdf: Detected non-finite density")

        # reshape densities to grid
        densities = densities.reshape(self.edgeshape)

        # store attributes
        self.densities_evaluated = True
        self.vertex_densities = densities
        self.masses = self._calculate_faverages() * self._calculate_volumes()
        self.total_mass = np.sum(self.masses)

    def choose(self, u):
        """Given 1D array of uniform samples, return indices of chosen cells.
        
        Parameters
        ----------
        u : 1D array of ints, shape
            Array of uniform samples ~ U(0, 1).
            
        Returns
        -------
        cells : 2D array of ints, shape (N, k)
            Grid indices of N chosen cells along the k dimensions of the grid.
        """
        if self.ncells == 1:
            return np.zeros((len(u), self.dim), dtype=np.int32)
            
        # get flattened array of grid cell probs: 1D array (len: #gridcells)
        p = (self.masses / self.masses.sum()).flatten()

        # choose cells (1D array of flattened indices)
        cells = _choice(p=p, u=u)

        # unravel 1D cell indices into 2D grid indices (N, k)
        cells = np.stack(np.unravel_index(cells, self.shape), axis=-1)
        return cells

    def get_cell_corner_densities(self, cells):
        """Return densities on 2^k corners of given cells.

        Parameters
        ----------
        cells : 2D array of ints, shape (N, k)
            Grid indices of N chosen cells along the k dimensions of the grid.
            
        Returns
        -------
        corners : 2^k-tuple of 1D numpy arrays, each length N
            Densities at corners of given cells. Conventional ordering applies,
            e.g., in 3D: (f000, f001, f010, f011, f100, f101, f110, f111)
        """
        # loop over 2^k corners, get densities at each
        corners = []
        for i in range(2**self.dim):
            
            # binary representation of corner, e.g. [0,0,...,0] is first corner
            n = np.binary_repr(i, width=self.dim)
            n = np.array([int(c) for c in n], dtype=int)
    
            # get densities on given corners
            idx = cells + n
            idx = np.split(idx.T, self.dim)
            idx = tuple([idxi[0] for idxi in idx])
            corners.append(self.vertex_densities[idx])
        return tuple(corners)

    def _calculate_faverages(self):
        """Calculate cell average densities.
        
        Averages over 2^k corner densities at each cell and returns averages.

        Returns
        -------
        average : numpy array, k-dimensional, shape (N0 x N1 x ... x N{k-1})
            k-dimensional array containing average vertex density of each grid
            cell (i.e., averaging over 2^k vertices). Shape is shape of grid.        
        """
        # infer dimensionality and grid shape
        shape = tuple([s - 1 for s in self.vertex_densities.shape])

        # initialise array to contain sum
        sum = np.zeros(shape)
        
        # loop over corners, add density contribution to sum
        slice0 = slice(-1)
        slice1 = slice(1, None)
        for i in range(2**self.dim):
            n = np.binary_repr(i, width=self.dim)
            t = ()
            for d in range(self.dim):
                t += ([slice0, slice1][int(n[d])],)
            sum += self.vertex_densities[t]
        
        # div. by no. corners for average
        averages = sum / 2**self.dim
        return averages


    def _calculate_volumes(self):
        """Calculate grid cell volumes.
        
        Calculates differences of edgearrays with numpy.diff, then volumes with
        outer product.
        
        Returns
        -------
        vols : numpy array, k-dimensional, shape (N0 x N1 x ... x N{k-1})
            k-dimensional array containing volumes of grid cells. Shape is shape
            of grid.
        """
        diffarrays = []
        for edgearray in self.edgearrays:
            diffarrays.append(np.diff(edgearray))
        shape = tuple([d.size for d in diffarrays])
        vols = reduce(np.outer, diffarrays).reshape(shape)
        return vols
