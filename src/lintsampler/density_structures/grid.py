import numpy as np
from functools import reduce
from ..utils import _is_1D_iterable, _choice
from .base import DensityStructure


class DensityGrid(DensityStructure):
    """Grid-like object over which density function is evaluated.

    ``DensityGrid`` uses the parameter ``edges`` to construct a rectilinear
    grid. ``edges`` should contain one or k sequences of numbers representing
    the `edges` of a one- or k-dimensional grid. The given edges need not be
    evenly spaced, but should be monotonically increasing. After the grid is
    constructed, the given PDF function is evaluated on the `vertices` of the
    grid.
 
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

    Attributes
    ---------- 
    mins : ``numpy`` array
        1D array (length ``dim``), giving lower coordinate bound of grid
        along each dimension. Enforced by base class ``DensityStructure``.
    maxs : ``numpy`` array
        1D array (length ``dim``), giving upper coordinate bound of grid
        along each dimension. Enforced by base class ``DensityStructure``.
    dim : int
        Dimensionality of grid. Enforced by base class ``DensityStructure``.
    total_mass : float
        Total probability mass of this grid; summed over ``masses`` array.
        Enforced by base class ``DensityStructure``.
    shape : tuple
        ``dim``-length tuple giving grid shape. For example, if 
        ``edgearrays`` have lengths N1+1, N2+1, ... Nk+1, then ``shape`` is
        (N1, N2, ..., Nk).
    ncells : int
        Total number of cells in grid, i.e., the product over the shape tuple.
    edgearrays : list
        Length ``dim`` list of arrays of grid edges along each dimension.
    vertex_densities : ``numpy`` array
        k-dimensional array giving densities at vertices of grid. Shape is
        (N1+1, N2+1, ...) if grid has (N1, N2, ...) cells along each dimension.
    masses : ``numpy`` array
        k-dimensional array of probability masses of grid cells. Shape is equal
        to grid shape (``shape`` attribute). Masses are calculated according to
        trapezoid rule, i.e., cell volumes multiplied by average vertex
        densities.

    Examples
    --------

    These examples demonstrate the various ways to set up and use an instance
    of ``DensityGrid``.

    - A one-dimensional grid, spanning x=0 to x=10 with 32 cells (so 33 edges).
      As an example PDF we can take an unnormalised Gaussian:

      >>> pdf = lambda x: np.exp(-x**2)
      >>> g = DensityGrid(np.linspace(0, 30, 33), pdf)
    
      At this point, the object ``g`` has various attributes set up describing
      the grid and its geometry. We'll save a demonstration of these for the
      next example, where they will be more interesting.

    - A two-dimensional grid, with 32 x 64 cells (so 33 gridlines along one
      axis and 65 along the other). For the PDF function, rather than writing
      our own we'll use the bivariate standard normal PDF from
      ``scipy.stats.multivariate_normal``:
    
      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> x = np.linspace(-2, 2, 33)
      >>> y = np.linspace(-4, 4, 65)
      >>> g = DensityGrid((x, y), pdf)

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
      
      There are also various attributes which relate to the evaluated
      probability densities and masses:
      
      >>> g.vertex_densities
      array([[7.22562324e-06, 1.18203307e-05, 1.90369817e-05, ...,
              1.90369817e-05, 1.18203307e-05, 7.22562324e-06],
             [9.20568282e-06, 1.50594920e-05, 2.42537439e-05, ...,
              2.42537439e-05, 1.50594920e-05, 9.20568282e-06],
             [1.15465131e-05, 1.88888347e-05, 3.04210101e-05, ...,
              3.04210101e-05, 1.88888347e-05, 1.15465131e-05],
             ...,
             [1.15465131e-05, 1.88888347e-05, 3.04210101e-05, ...,
              3.04210101e-05, 1.88888347e-05, 1.15465131e-05],
             [9.20568282e-06, 1.50594920e-05, 2.42537439e-05, ...,
              2.42537439e-05, 1.50594920e-05, 9.20568282e-06],
             [7.22562324e-06, 1.18203307e-05, 1.90369817e-05, ...,
              1.90369817e-05, 1.18203307e-05, 7.22562324e-06]])
      >>> g.masses
      array([[1.69184097e-07, 2.74103704e-07, 4.37229522e-07, ...,
              4.37229522e-07, 2.74103704e-07, 1.69184097e-07],
             [2.13673916e-07, 3.46183909e-07, 5.52206419e-07, ...,
              5.52206419e-07, 3.46183909e-07, 2.13673916e-07],
             [2.65695257e-07, 4.30466311e-07, 6.86647340e-07, ...,
              6.86647340e-07, 4.30466311e-07, 2.65695257e-07],
              ...,
             [2.65695257e-07, 4.30466311e-07, 6.86647340e-07, ...,
              6.86647340e-07, 4.30466311e-07, 2.65695257e-07],
             [2.13673916e-07, 3.46183909e-07, 5.52206419e-07, ...,
              5.52206419e-07, 3.46183909e-07, 2.13673916e-07],
             [1.69184097e-07, 2.74103704e-07, 4.37229522e-07, ...,
              4.37229522e-07, 2.74103704e-07, 1.69184097e-07]])
      >>> g.total_mass
      0.9541568382986452
      
      ``vertex_densities`` is a 2D (33 x 65) array containing the densities on
      all of the grid vertices, while ``masses`` is a 2D (32 x 64) array giving
      the probability masses of grid cells, and ``total_mass`` gives the total
      probability mass across the grid.

    - Things are slightly more efficient when the PDF function is vectorized,
      i.e., the PDF function takes a batch of input positions and returns a
      corresponding batch of densities. As it happens, the PDF function we used
      above is already nicely vectorized, so we can let DensityGrid know about
      this with a simple Boolean flag:
      
      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> x = np.linspace(0, 10, 33)
      >>> y = np.linspace(100, 200, 65)
      >>> g = DensityGrid((x, y), pdf, vectorizedpdf=True)

    - It is also possible to construct a grid with just a single cell.
    
      In one dimension:
    
      >>> pdf = lambda x: np.exp(-x**2)
      >>> g = DensityGrid([-5, 5], pdf)
      >>> g.ncells
      1
       
      In multiple dimensions:
    
      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> g = DensityGrid(([0, 10], [100, 200]), pdf)
      >>> g.ncells
      1
      
    - Example usage of ``choose_cells`` method
      
      The ``choose_cells`` method can be used to randomly select grid cells
      (weighted by their masses), given a 1D array of uniform samples.

      >>> pdf = multivariate_normal(mean=np.zeros(2), cov=np.eye(2)).pdf
      >>> x = y = np.linspace(-3, 3, 129)
      >>> g = DensityGrid((x, y), pdf, vectorizedpdf=True)
      >>> u = np.random.default_rng().uniform(size=10)  
      >>> mins, maxs, corners = g.choose_cells(u)
      
      Let's inspect the returned arrays:
      
      >>> mins
      array([[ 1.546875,  0.703125],
             [-0.234375, -0.234375],
             [ 0.      ,  0.84375 ],
             [ 1.03125 ,  1.359375],
             [-0.9375  ,  0.      ],
             [ 0.09375 , -0.9375  ],
             [-1.453125, -0.046875],
             [-0.890625, -0.84375 ],
             [-1.265625, -0.515625],
             [-1.3125  , -0.84375 ]])
      >>> maxs
      array([[ 1.59375 ,  0.75    ],
             [-0.1875  , -0.1875  ],
             [ 0.046875,  0.890625],
             [ 1.078125,  1.40625 ],
             [-0.890625,  0.046875],
             [ 0.140625, -0.890625],
             [-1.40625 ,  0.      ],
             [-0.84375 , -0.796875],
             [-1.21875 , -0.46875 ],
             [-1.265625, -0.796875]])
      >>> corners
      (array([0.03757259, 0.15064809, 0.11148847, 0.03712126, 0.10255765,
              0.10210795, 0.0553122 , 0.074987  , 0.06255463, 0.04711508]),
       array([0.0363145 , 0.15214504, 0.1070474 , 0.03479141, 0.10244504,
              0.10657801, 0.055373  , 0.07792656, 0.06401463, 0.04896204]),
       array([0.03490626, 0.15214504, 0.11136605, 0.03533066, 0.1070474 ,
              0.10154859, 0.05914606, 0.07809798, 0.06630517, 0.05004977]),
       array([0.03373746, 0.15365686, 0.10692986, 0.0331132 , 0.10692986,
              0.10599417, 0.05921108, 0.0811595 , 0.0678527 , 0.05201177]))

      ``mins`` and ``maxs`` are both arrays shaped (10, 2), 10 because we fed in
      10 uniform samples and 2 because we have a 2D grid (and bivariate PDF). 
      ``corners`` is a length-4 tuple of length-10 arrays. 4 because each of the
      10 cells has 4 corners. So, the first array contains the densities at
      the 00-corner of each cell, the second array contains the 01-densities,
      the third array gives the 10-densities, and the fourth array gives the
      11-densities.
    """
    def __init__(self, edges, pdf, vectorizedpdf=False, pdf_args=(), pdf_kwargs={}):
        
        # 1D case: cells is 1D iterable (array, tuple, list)
        # e.g. cells = np.linspace(-4,4,50)
        if _is_1D_iterable(edges):

            # store dimensionality (1) and single edgearray and dims
            self._dim = 1
            self.edgearrays = [np.array(edges)]
            self._edgeshape = (len(edges),)

        # kD case: cells is tuple of 1D iterables
        # e.g. cells = (np.linspace(-12,12,100),np.linspace(-4,4,50))
        elif isinstance(edges, tuple) and _is_1D_iterable(edges[0]):
            
            # infer dimensionality
            self._dim = len(edges)

            # loop over dimensions, store edge arrays and dims
            self.edgearrays = []
            self._edgeshape = ()
            for d in range(0,self.dim):
                self.edgearrays.append(np.array(edges[d]))
                self._edgeshape += (len(edges[d]),)
        
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
        self._mins = np.array([arr[0] for arr in self.edgearrays])
        self._maxs = np.array([arr[-1] for arr in self.edgearrays])
        self.shape = tuple(d - 1 for d in self._edgeshape)
        self.ncells = np.prod(self.shape)
        self._cell_mins = np.stack(np.meshgrid(*[a[:-1] for a in self.edgearrays], indexing='ij'), axis=-1)
        self._cell_maxs = np.stack(np.meshgrid(*[a[1:] for a in self.edgearrays], indexing='ij'), axis=-1)
    
        # evaluate PDF (set relevant attrs)
        self._evaluate_pdf(pdf, vectorizedpdf, pdf_args, pdf_kwargs)
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def mins(self):
        return self._mins

    @property
    def maxs(self):
        return self._maxs
    
    @property
    def total_mass(self):
        return self._total_mass
    
    def choose_cells(self, u):
        """Choose cells given 1D array of uniform samples.
        
        Method enforced by base class ``DensityStructure``.
        
        Parameters
        ----------
        u : 1D array of floats, shape (N,)
            Array of uniform samples ~ U(0, 1).
        
        Returns
        -------
        mins : 2D array of floats, shape (N, k)
            Coordinate vector of first corner of each cell.
        maxs : 2D array of floats, shape (N, k)
            Coordinate vector of last corner of each cell.
        corners : 2^k-tuple of 1D arrays, each length N
            Densities at corners of given cells. Conventional ordering applies,
            e.g., in 3D: (f000, f001, f010, f011, f100, f101, f110, f111)
        """
        # get (N, k) array of cell indices
        cells = self._choose_indices(u)

        # convert to grid indexing arrays
        idx = np.split(cells.T, self.dim)
        idx = tuple([idxi[0] for idxi in idx])

        # cell mins, cell maxs, corner densities for chosen cells        
        mins = self._cell_mins[idx]
        maxs = self._cell_maxs[idx]
        densities = self._get_cell_corner_densities(cells)
        return mins, maxs, densities

    def _evaluate_pdf(self, pdf, vectorizedpdf, pdf_args, pdf_kwargs):
        """Evaluate the user-provided pdf on grid and set related attributes.
        
        Parameters
        ----------
        pdf : function
            Probability density function to evaluate on grid. See corresponding
            parameter in class constructor.
        vectorizedpdf : bool
            Whether pdf function is vectorized. See corresponding parameter in
            class constructor.
        pdf_args : tuple
            Additional positional arguments to pass to pdf function. See
            corresponding parameter in class constructor.
        pdf_kwargs : dict
            Additional keyword arguments to pass to pdf function. See
            corresponding parameter in class constructor.

        Returns
        -------
        None
        """
        # number of vertices in grid
        npts = np.prod(self._edgeshape)
        
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
        densities = densities.reshape(self._edgeshape)

        # store attributes
        self.vertex_densities = densities
        self.masses = self._calculate_faverages() * self._calculate_volumes()
        self._total_mass = np.sum(self.masses)

    def _choose_indices(self, u):
        """Given 1D array of uniform samples, return indices of chosen cells.
        
        Parameters
        ----------
        u : 1D array of floats, shape (N,)
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

    def _get_cell_corner_densities(self, cells):
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
