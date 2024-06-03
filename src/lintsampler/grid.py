import numpy as np
from functools import reduce
from warnings import warn
from .utils import _is_1D_iterable, _choice

#TODO: remove edgeshape attribute
class DensityGrid:
    """Grid-like object over which density function is evaluated.

    ``DensityGrid`` takes a single parameter, ``edges``, and uses it to
    construct a rectilinear grid. ``edges`` should contain one or k sequences of
    numbers representing the `edges` of a one- or k-dimensional grid. The given
    edges need not be evenly spaced, but should be monotonically increasing.
    After construction, various grid-related attributes become available.

    The key method of the class is ``evaluate``, in which a given PDF is
    evaluated on the `vertices` of the grid and stored as an attribute
    (``vertex_densities``) alongside several other new attributes relating
    to the densities/probability masses on the grid. The Boolean flag
    ``densities_evaluated`` switches to ``True`` at this point. After calling
    this method, several new methods become available: ``choose`` returns
    random cells weighted by their probabilities, and 
    ``get_cell_corner_densities`` returns the densities on the corners of given
    cells.
 
    See the examples below for the various usage patterns.

    Parameters
    ----------
    edges : 1D iterable or tuple of 1D iterables
        If a single 1D iterable (i.e., array, tuple, list of numbers), then this
        represents the cell `edges` of a 1D grid. So a 1D grid with N cells
        should have ``edges`` parameter as a 1D array with N+1 elements. If a
        tuple of 1D iterables, then this represents the edge values for a 
        k-dimensional grid with dimensionality equal to the length of the tuple.
        So, a kD grid with (N1 x N2 x ... x Nk) cells should take a length-k
        tuple, with arrays length N1+1, N2+1 etc.

    Attributes
    ---------- 
    dim : int
        Dimensionality of grid.
    shape : tuple
        ``dim``-length tuple giving grid shape. For example, if 
        ``edgearrays`` have lengths N1+1, N2+1, ... Nk+1, then ``shape`` is
        (N1, N2, ..., Nk).
    ncells : int
        Total number of cells in grid, i.e., the product over the shape tuple.
    edgearrays : list
        Length ``dim`` list of arrays of grid edges along each dimension.
    edgeshape : tuple
        Length ``dim`` tuple giving lengths of `edgearrays`. This is equal to
        ``shape`` tuple, but plus one to each element.
    mins : ``numpy`` array
        1D array (length ``dim``), giving lower coordinate bound of grid
        along each dimension.
    maxs : ``numpy`` array
        1D array (length ``dim``), giving upper coordinate bound of grid
        along each dimension.
    densities_evaluated : bool
        Flag indicating whether density function has been evaluated on grid (via
        ``evaluate`` method) yet. 
    vertex_densities : {``None``, ``numpy`` array}
        If ``densities_evaluated``, k-dimensional array giving densities at
        vertices of grid. Shape is equal to ``edgeshape`` attribute, i.e.
        (N1+1, N2+1, ...) if grid has shape (N1, N2, ...). ``None`` if not 
        ``densities_evaluated``.
    masses : {``None``, ``numpy`` array}
        If ``densities_evaluated``, k-dimensional array of probability masses of
        grid cells. Shape is equal to grid shape (``shape`` attribute). Masses
        are calculated according to trapezoid rule, i.e., cell volumes
        multiplied by average vertex densities. ``None`` if not 
        ``densities_evaluated``.
    total_mass : {``None``, float}
        If ``densities_evaluated``, total probability mass of this grid; sum over
        ``masses`` array. ``None`` if not ``densities_evaluated``.


    Examples
    --------

    These examples demonstrate the various ways to set up and use an instance
    of ``DensityGrid``.

    - A one-dimensional grid, spanning x=0 to x=10 with 32 cells (so 33 edges).
    
      >>> g = DensityGrid(np.linspace(0, 30, 33))
    
      At this point, the object ``g`` has various attributes set up describing
      the grid and its geometry. We'll save a demonstration of these for the
      next example, where they will be more interesting.

    - A two-dimensional grid, with 32 x 64 cells (so 33 gridlines along one
      axis and 65 along the other).
    
      >>> x = np.linspace(0, 10, 33)
      >>> y = np.linspace(100, 200, 65)
      >>> g = DensityGrid((x, y))

      Let's explore some of the attributes of ``g``. First, some basic
      descriptors of the grid geometry:
       
      >>> g.dim
      2
      >>> g.shape
      (32, 64)
      >>> g.ncells
      2048
       
      There are also array attributes called ``mins`` and ``maxs`` which give
      the coordinates of the 'first' and 'last' corners of the grid. In 2D,
      these are the bottom-left and top-right:
       
      >>> g.mins
      array([  0., 100.])
      >>> g.maxs
      array([ 10., 200.])
       
      Meanwhile, ``edgearrays`` gives a list of the input edge arrays:
       
      >>> len(g.edgearrays)
      2
      >>> all(g.edgearrays[0] == x)
      True
      >>> all(g.edgearrays[1] == y)
      True
      
      Finally, the flag ``densities_evaluated`` tells us that we have not yet
      called the ``evaluate`` method to get the grid vertex densities:
      
      >>> g.densities_evaluated
      False

    - It is also possible to construct a grid with just a single cell.
    
      In one dimension:
    
      >>> g = DensityGrid([0, 10])
      >>> g.ncells
      1
       
      In multiple dimensions:
       
      >>>  g = DensityGrid(([0, 10], [100, 200]))
      >>> g.ncells
      1

    - Now, a demonstration of using ``evaluate`` to evaluate a given 1D PDF
      function.
    
      As an example in 1D we can take an unnormalised Gaussian:
      >>> pdf = lambda x: np.exp(-x**2)      
      >>> g = DensityGrid(np.linspace(-3, 3, 7))
      >>> g.evaluate(pdf)
      
      Having called ``evaluate``, the ``densities_evaluated`` flag is now
      ``True`` and various other attributes are now meaningful:
      
      >>> g.densities_evaluated
      True
      >>> g.vertex_densities
      array([1.23409804e-04, 1.83156389e-02, 3.67879441e-01, 1.00000000e+00,
             3.67879441e-01, 1.83156389e-02, 1.23409804e-04])
      >>> g.masses
      array([0.00921952, 0.19309754, 0.68393972, 0.68393972, 0.19309754,
             0.00921952])
      >>> g.total_mass
      1.7725135699244396
      
      ``vertex_densities`` contains the densities on the 7 grid vertices, while
      ``masses`` contains the masses of the 6 cells, and ``total_mass`` gives
      the total probability mass across the grid.
      
      Things are slightly more efficient when the PDF function is vectorized,
      i.e., the PDF function takes a batch of input positions and returns a
      corresponding batch of densities. As it happens, the PDF function written
      above is already nicely vectorized, so we can let ``evaluate`` know with
      a flag. We can use the same grid object as above, taking care to first
      reset the densities.
      
      >>> g.reset_densities()
      >>> g.evaluate(pdf, vectorizedpdf=True)
    
    - Another example of ``evaluate``, now with a bivariate PDF.
      
      For the PDF function, rather than writing our own we'll use the bivariate
      standard normal PDF from ``scipy.stats.multivariate_normal``:
    
      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> g = DensityGrid((np.linspace(-3, 3, 129), np.linspace(-3, 3, 129)))
      >>> g.evaluate(pdf, vectorizedpdf=True)
      >>> g.total_mass
      0.9945979872720603
      
    - Example usage of ``choose`` and ``get_cell_corner_densities`` methods
      
      Once the ``evaluate`` method has been called (so that the attribute 
      ``densities_evaluated`` is ``True``), the ``choose`` method can be used
      to randomly select grid cells (weighted by their masses), given a 1D
      array of uniform samples.
      
      Taking the same 2D grid and PDF as in the previous example:
      
      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> g = DensityGrid((np.linspace(-3, 3, 129), np.linspace(-3, 3, 129)))
      >>> g.evaluate(pdf, vectorizedpdf=True)
      >>> u = np.random.default_rng().uniform(size=10)  
      >>> cells = g.choose(u)
      >>> cells
      array([[70, 89],
             [72, 36],
             [46, 41],
             [67, 41],
             [74, 92],
             [77, 60],
             [38, 56],
             [83, 40],
             [69, 64],
             [82, 49]])

      This returns a 2D array, shaped (10, 2), 10 because we fed in 10 uniform
      samples and 2 because we have a 2D grid (and bivariate PDF). The integer
      array elements represent the cell indices along each dimension of the
      grid.
      
      We can take this array of cells and feed it to the method 
      ``get_cell_corner_densities`` to get the vertex densities at the corners
      of the given cells.
      
      >>> g.get_cell_corner_densities(cells)
      (array([0.0680767 , 0.09707593, 0.12093098, 0.05393205, 0.09988883,
              0.02469491, 0.13291402, 0.04088934, 0.14672783, 0.01712865]),
       array([0.06479294, 0.09382542, 0.12213264, 0.05247092, 0.10380458,
              0.025776  , 0.13045465, 0.04401268, 0.1449653 , 0.01703481]),
       array([0.0655087 , 0.1002186 , 0.12484592, 0.0505471 , 0.09761065,
              0.02279171, 0.13571763, 0.0416602 , 0.14883847, 0.01888807]),
       array([0.0623488 , 0.09686286, 0.12608648, 0.04917767, 0.10143709,
              0.02378948, 0.13320639, 0.04484242, 0.14705059, 0.0187846 ]))

      This returns a length-4 tuple of length-10 arrays. 4 because each of the
      10 cells has 4 corners. So, the first array contains the densities at
      the 00-corner of each cell, the second array contains the 01-densities,
      the third array gives the 10-densities, and the fourth array gives the
      11-densities.
      
    """
    def __init__(self, edges):
        
        # 1D case: cells is 1D iterable (array, tuple, list)
        # e.g. cells = np.linspace(-4,4,50)
        if _is_1D_iterable(edges):

            # store dimensionality (1) and single edgearray and dims
            self.dim = 1
            self.edgearrays = [np.array(edges)]
            self.edgeshape = (len(edges),)

        # kD case: cells is tuple of 1D iterables
        # e.g. cells = (np.linspace(-12,12,100),np.linspace(-4,4,50))
        elif isinstance(edges, tuple) and _is_1D_iterable(edges[0]):
            
            # infer dimensionality
            self.dim = len(edges)

            # loop over dimensions, store edge arrays and dims
            self.edgearrays = []
            self.edgeshape = ()
            for d in range(0,self.dim):
                self.edgearrays.append(np.array(edges[d]))
                self.edgeshape += (len(edges[d]),)
        
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
        u : 1D array of ints, shape (N,)
            Array of uniform samples ~ U(0, 1).
            
        Returns
        -------
        cells : 2D array of ints, shape (N, k)
            Grid indices of N chosen cells along the k dimensions of the grid.
        """
        #TODO throw error if not densities evaluated
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
        #TODO throw error if not densities evaluated
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
