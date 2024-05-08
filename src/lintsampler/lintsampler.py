
import numpy as np
from .gridsample import _gridsample
from .freesample import _freesample

class LintSampler:
    """Draw sample(s) from density function defined for a list of cells (that may or may not be in a grid)

    Examples
    --------
    
    These examples demonstrate the multiple ways to use ``LintSampler``. In
    each case, we'll just generate densities from a uniform distribution, but
    in general they might come from any arbitrary density function.

    1. A single sample from a 1D grid. The grid spans x=0 to x=10, and has 32
    cells (so 33 edges). 
    
    >>> grid = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,grid).sample()
    0.7355598727871656

    This returns a single scalar: the sampling point within the grid.
    
    2. Multiple samples from a 1D grid (same grid as previous example).
    
    >>> grid = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,grid).sample(4)
    array([0.7432799 , 6.64118763, 9.65968316, 5.39087554])

    This returns a 1D array: the ``N_samples`` sampling points within the grid.

    3. Single sample from a k-D grid. In this case we'll take a 2D grid, with
    32 x 64 cells (so 33 gridlines along one axis and 65 along the other, and
    33x65=2145 intersections with known densities).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> def pdfrandom(X): return np.random.uniform(size=(X.shape[0],X.shape[1]))
    >>> LintSampler(pdfrandom,grid).sample()

    >>> f = np.random.uniform(size=(33, 65))
    >>> gridsample(x, y, f=f)
    array([  7.67294632, 190.45302915])

    This returns a 1D array: the single k-D sampling point within the grid.

    4. Multiple samples from a k-D grid (same grid as previous example).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> f = np.random.uniform(size=(33, 65))
    >>> gridsample(x, y, f=f, N_samples=5)
    array([[1.35963966e-01, 1.38182930e+02],
           [6.52704300e+00, 1.63109912e+02],
           [4.35226761e+00, 1.49753235e+02],
           [3.56093155e+00, 1.48548481e+02],
           [1.31163401e+00, 1.59335676e+02]])

    This returns a 2D array, shape ``N_samples`` x k: the ``N_samples`` k-D
    samples within the grid.
    """

    def __init__(self, pdf, cells=(), rngseed=42):
        """Initialise a LintSampler instance.

        Parameters
        ----------
        pdf : function
            pdf needs to take an input with dimension [...,k] where k is the number of dimensions

        cells : a tuple with the following optional inputs:
            1. A tuple with k (min,max,num) entries
            2. A single nd-array that defines the cells
            3. A tuple of nd-arrays

        Returns
        -------
        None

        """

        # set the pdf to be widely accessible
        self.pdf = pdf

        # set the random seed
        self.rng = np.random.default_rng(rngseed)

        # check the validity of the cells
        if len(cells)==0:
            raise ValueError("LintSampler.__init__: you must specify an evaluation domain with a tuple of boundaries, arrays, or grid points. See documentation for details.")
            # or, we could here drop into an adaptive grid selection
            # or, we could sample on a unit hypercube with dimensions of the pdf

        # check element grid for which sampling method we will use and design the grid
        self._setgrid(cells)



    def sample(self, N_samples=None, funcargs=()):
        """Draw samples from the pdf on the grid.
        
        This function then draws a sample (or N samples) from the specified grid. It first chooses a cell (or N cells with replacement), weighting them by their mass (estimated by the trapezoid rule) then samples from k-linear interpolant within the chosen cell(s).
        
        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.
        N_samples : {None, int}, optional
            Number of samples to draw. Default is None, in which case a single
            sample is drawn.
        seed : {None, int, ``numpy.random.Generator``}, optional
            Seed for ``numpy`` random generator. Can be random generator itself, in
            which case it is left unchanged. Default is None, in which case new
            default generator is created. See ``numpy`` random generator docs for
            more information.

        Returns
        -------
        X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
            Sample(s) from linear interpolant. Scalar if single sample (i.e.,
            N_samples is None) in 1D. 1D array if single sample in k-D OR multiple
            samples in 1D. 2D array if multiple samples in k-D.
        
        
        """
        
        # check that the function can be evaluated at the passed-in points (i.e. needs to take the dimensions as an argument)

        if self.eval_type == 'gridsample':

            if self.dim > 1:
                # create the flattened grid for evaluation
                edgegrid = np.stack(np.meshgrid(*self.edgearrays, indexing='ij'), axis=-1).reshape(np.prod(self.edgedims), self.dim)

                # reshape the grid: assumes function takes same number of arguments as dimensions
                evalf = self.pdf(edgegrid,*funcargs).reshape(*self.edgedims)

                # call the gridded sampler
                X = _gridsample(*self.edgearrays,f=evalf,N_samples=N_samples,seed=self.rng)

            else: 
                # the 1d case: no flattening needed
                evalf = self.pdf(self.edgearrays,*funcargs)

                X = _gridsample(self.edgearrays,f=evalf,N_samples=N_samples,seed=self.rng)

            return X


        elif self.eval_type == 'freesample':

            if self.dim == 1:

                # pdf call needs to be 1d, so rearrange
                corners = (np.array([self.pdf(x[0],*funcargs) for x in self.x0]),np.array([self.pdf(x[0]) for x in self.x1]))

            else:

                # get all combinations of starting and ending corner points (only depends on dimension)
                combinations = self._get_startend_points()

                # in case of a single grid being passed:
                if self.ngrids == 0:
                    # evaluate all points on the initial input grid and reshape
                    evalfgrid = self.pdf(self.edgearrays,*funcargs).reshape(self.gridshape[0:self.dim])

                    # now get the values corners: there will be 2^dim of them
                    corners = []
                    for combo in combinations: 

                        arrslice = []
                        for griddim,entry in enumerate(combo):

                            if entry==0:
                                arrslice.append(slice(0,self.gridshape[griddim]-1))

                            else: # must be a 1
                                arrslice.append(slice(1,self.gridshape[griddim]))

                        corners.append(evalfgrid[tuple(arrslice)].flatten())
                    
                else:

                    #raise NotImplementedError("LintSampler: freesample with ngrids>=1 to be implemented.")

                    evalfgrid = []
                    allcorners = []
                    for ngrid in range(0,self.ngrids):
                        allcorners.append([]) # make a list of lists, to later concatenate
                        evalfgrid.append(self.pdf(self.edgearrays[ngrid],*funcargs).reshape(self.gridshape[ngrid][0:self.dim]))

                        for combo in combinations: 

                            arrslice = []
                            for griddim,entry in enumerate(combo):

                                if entry==0:
                                    arrslice.append(slice(0,self.gridshape[ngrid][griddim]-1))

                                else: # must be a 1
                                    arrslice.append(slice(1,self.gridshape[ngrid][griddim]))

                            allcorners[ngrid].append(evalfgrid[ngrid][tuple(arrslice)].flatten())

                    # now go back through and remake the corners to stack everything up
                    corners = []
                    for ncombo in range(0,len(combinations)):
                        corners.append(np.hstack([allcorners[ngrid][ncombo] for ngrid in range(0,self.ngrids)]))


            # do the sampling
            X = _freesample(self.x0,self.x1,*corners,N_samples=N_samples,seed=self.rng)

            return X


        else:
            raise ValueError("LintSampler.sample: eval_type is expected to be either gridsample or freesample.")

        pass


    def resetgrid(self,grid=()):
        """Reset the sample grid without changing the pdf.
        
        
        """
        self._setgrid(grid)


    def _setgrid(self,cells):
        """Construct the cells for sampling.


        eval_type
        dim
        edgearrays
        edgedims

        Cases we handle:
            1. cells = (edges,edges)
            2. cells = [(edges,edges),(different_edges,different_edges)]
        
        
        """
        # 0. are we in the 1d case? the input is just a single tuple or 1d array.
        # i.e. cells = (-12,12,100) or cells = np.linspace(-12,12,100)
        if (isinstance(cells[0],float) | (isinstance(cells[0],int))):
            self.eval_type='gridsample'

            # override the inferred dimensionality
            self.dim = 1

            # set the arrays
            if len(cells)==3: # we are being passed a (min,max,n) tuple
                self.edgearrays = np.linspace(cells[0],cells[1],cells[2])
                self.edgedims = cells[2]
            else: # we are being passed the array directly
                self.edgearrays = cells
                self.edgedims = len(cells)


        # 1. tuples defining the array -> make arrays, pass to gridsample
        # i.e. cells = ((-12,12,100),(-4,4,50))
        if isinstance(cells[0],tuple):
            self.eval_type = 'gridsample'

            # keep track of the edge arrays
            self.edgearrays = []

            # keep track of the dimensions
            self.edgedims = np.ones(self.dim,dtype='int')

            # create the grid
            for d in range(0,self.dim):
                self.edgearrays.append(np.linspace(cells[d][0],cells[d][1],cells[d][2]))
                self.edgedims[d] = cells[d][2]


        # two switches for an array input
        if isinstance(cells[0],np.ndarray):

            # 2. the grid values themselves -> gridsample. check the .dim numpy attribute
            if cells[0].ndim == 1:
                self.eval_type = 'gridsample'

                # keep track of the edge arrays
                self.edgearrays = []

                # keep track of the dimensions
                self.edgedims = np.ones(self.dim,dtype='int')

                # store the grid
                for d in range(0,self.dim):
                    self.edgearrays.append(cells[d])
                    self.edgedims[d] = len(cells[d])


            # 3. the grid points themselves; each entry in grid is a separate grid 
            else:
                self.eval_type = 'freesample'
            
                # the grids must be [...,dim]


                # infer the dimensionality: this will work whether the input is a single grid or list of grids
                self.dim = cells[0].shape[-1]

                if self.dim==1:
                    # really, we shouldn't ever be in this case, because it's just a grid.
                    # now we are looking at the edges of the grids, 
                    self.x0 = cells[0] # the lower boundaries for the cells
                    self.x1 = cells[1] # the upper boundaries for the cells

                else:

                    # check: is this a list of grids?
                    if (isinstance(cells,list)|(isinstance(cells,tuple))):
                        # how many grids are we being handed?
                        self.ngrids = len(cells)

                        # loop through the grids and flatten
                        self.gridshape = []
                        self.ngridentries = [] # the number of entries in the grid
                        self.nedgegridentries = [] # the number of entries in the edge grid
                        self.edgearrays = [] # the flattened grids, each [ngridentries,dim]
                        for ngrid in range(0,self.ngrids):
                            self.gridshape.append(cells[ngrid].shape)
                            self.ngridentries.append(np.prod(cells[ngrid].shape[0:self.dim]))
                            self.nedgegridentries.append(np.prod(np.array(cells[ngrid].shape[0:self.dim])-1))
                            self.edgearrays.append(cells[ngrid].reshape(self.ngridentries[ngrid],self.dim))
                            # now self.edgearrays is a list of flattened grids, and can be passed to self.pdf
                        
                        self.x0 = np.vstack([cells[ngrid][tuple([slice(0,self.gridshape[ngrid][griddim]-1) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries[ngrid],self.dim)) for ngrid in range(0,self.ngrids)])
                        self.x1 = np.vstack([cells[ngrid][tuple([slice(1,self.gridshape[ngrid][griddim]  ) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries[ngrid],self.dim)) for ngrid in range(0,self.ngrids)])

                    
                    else:
                        # not a list of grids, so this must be a single grid. special case handling...
                        self.ngrids = 0

                        # special case of a single grid
                        self.gridshape = cells.shape
                        self.ngridentries = np.prod(cells.shape[0:self.dim]) # the number of entries in the grid
                        self.nedgegridentries = np.prod(np.array(cells.shape[0:self.dim])-1) # the number of entries in the grid
                        self.edgearrays = cells.reshape(self.ngridentries,self.dim) # the flattened grid, [nengrids,dim]

                        self.x0 = np.vstack([cells[tuple([slice(0,self.gridshape[griddim]-1) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries,self.dim))])
                        self.x1 = np.vstack([cells[tuple([slice(1,self.gridshape[griddim]  ) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries,self.dim))])


    def _get_startend_points(self):

        # Create an array representing the start/end points
        arr = np.array([0, 1]) 

        # get all combinations
        combinations = np.array(np.meshgrid(*[arr]*self.dim)).T.reshape(-1,self.dim)

        # Rearrange the combinations
        sorted_combinations = np.lexsort(combinations.T[::-1])

        # Reorder the combinations
        combinations = combinations[sorted_combinations]

        return combinations