
import numpy as np
from .gridsample import _gridsample
from .freesample import _freesample

class LintSampler:
    """Draw sample(s) from density function defined for a list of cells (that may or may not be in a single grid)

    Examples
    --------
    
    These examples demonstrate the multiple ways to use ``LintSampler``. In
    each case, we'll just generate densities from a uniform distribution, but
    in general they might come from any arbitrary density function.

    1. A single sample from a 1D grid. The grid spans x=0 to x=10, and has 32
    cells (so 33 edges). 
    
    >>> cells = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,cells).sample()
    6.984134639227398

    This returns a single scalar: the sampling point within the grid.
    
    2. Multiple samples from a 1D grid (same grid as previous example).
    
    >>> cells = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,cells).sample(4)
    array([8.16447008, 5.30536088, 8.96135879, 7.73572977])

    This returns a 1D array: the ``N_samples`` sampling points within the grid.

    3. Single sample from a k-D grid. In this case we'll take a 2D grid, with
    32 x 64 cells (so 33 gridlines along one axis and 65 along the other, and
    33x65=2145 intersections with known densities).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> cells = (x,y)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,cells).sample()
    array([  7.67294632, 190.45302915])

    This returns a 1D array: the single k-D sampling point within the grid.

    4. Multiple samples from a k-D grid (same grid as previous example).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> cells = (x,y)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(pdfrandom,cells).sample(5)
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

        cells : tuple of arrays or list of tuples of arrays
            If a tuple of arrays, the boundary values for a grid with dimensionality of the length
            of the tuple. If a list of tuples of arrays, the boundary values for an arbitrary
            number of grids with dimensionality the length of the tuples.

        rngseed : {None, int, ``numpy.random.Generator``}, optional
            Seed for ``numpy`` random generator. Can be random generator itself, in
            which case it is left unchanged. Default is None, in which case new
            default generator is created. See ``numpy`` random generator docs for
            more information.

        Returns
        -------
        None

        """

        # set the pdf to be widely accessible
        self.pdf = pdf

        # set the random seed
        self.rng = np.random.default_rng(rngseed)

        # set up the sampling grid under the hood
        self._setgrid(cells)



    def sample(self, N_samples=None, funcargs=()):
        """Draw samples from the pdf on the constructed grid(s).
        
        This function then draws a sample (or N samples) from the specified grid. It first chooses a cell 
        (or N cells with replacement), weighting them by their mass (estimated by the trapezoid rule) 
        then samples from k-linear interpolant within the chosen cell(s).
        
        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.
        N_samples : {None, int}, optional
            Number of samples to draw. Default is None, in which case a single
            sample is drawn.
        funcargs : optional
            A tuple of optional arguments to be passed directly to the pdf function call.

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


    def resetgrid(self,cells=()):
        """Reset the sample grid without changing the pdf.
        
        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.

        Returns
        -------
        combinations: numpy array
            All possible combinations of starting and ending points for the
            inferred dimensionality of the problem.
                
        """
        self._setgrid(cells)


    def _setgrid(self,cells):
        """Construct the cells for sampling.

        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.
        cells : tuple of arrays or list of tuples of arrays
            If a tuple of arrays, the boundary values for a grid with dimensionality of the length
            of the tuple. If a list of tuples of arrays, the boundary values for an arbitrary
            number of grids with dimensionality the length of the tuples.

        Returns
        -------
        None
        
        Attributes
        -------
        eval_type : string
            The evaluation type, either 'gridsample' or 'freesample'
        dim : int
            The inferred dimensionality of the problem
        edgearrays : list of numpy arrays
            The arrays defining the edge of the grids
        edgedims   : list of integers, optional, used if eval_type == 'gridsample'
            The shape of the gridsample grid
        ngrids : integer, optional, used if eval_type == 'freesample'
            The number of distinct grids to be constructed
        gridshape : list of tuple of integers, optional, used if eval_type == 'freesample'
            The shape of each distinct grid
        ngridentries : list of integers, optional, used if eval_type == 'freesample'
            The number of entries in the 
        nedgegridentries : list of integers, optional, used if eval_type == 'freesample'
            The number of entries in the offset grids
        x0 : list of numpy arrays, optional, used if eval_type == 'freesample'
            The array of the first vertex of the grids
        x1 : list of numpy arrays, optional, used if eval_type == 'freesample'
            The array of the last vertex of the grids

        
        """

        # check the validity of the input
        if len(cells)==0:
            raise ValueError("LintSampler._setgrid: you must specify an evaluation domain with a tuple of arrays or a list of tuples of arrays. See documentation for details.")
            # or, we could here drop into an adaptive grid selection
            # or, we could sample on a unit hypercube with dimensions of the pdf

        # 0. are we in the 1d case? the input is just a single tuple or 1d array.
        # i.e. cells = np.linspace(-12,12,100)
        if (isinstance(cells[0],float) | (isinstance(cells[0],int))): # do we need the integer option here? I think no?
            self.eval_type='gridsample'

            # override the inferred dimensionality
            self.dim = 1

            # set the arrays
            # we are being passed the array directly
            self.edgearrays = cells
            self.edgedims = len(cells)


        # 1. tuples defining the array -> make arrays, pass to gridsample
        # i.e. cells = (np.linspace(-12,12,100),np.linspace(-4,4,50))
        if isinstance(cells,tuple):

            if isinstance(cells[0],tuple):
                raise ValueError("LintSampler: Cells must be a single tuple or a list of tuples (i.e. not a tuple of tuples).")
            
            # infer the dimensionality
            self.dim = len(cells)

            # set the evaluation type
            self.eval_type = 'gridsample'

            # keep track of the edge arrays
            self.edgearrays = []

            # keep track of the dimensions
            self.edgedims = np.ones(self.dim,dtype='int')

            # create the grid
            for d in range(0,self.dim):
                self.edgearrays.append(cells[d])
                self.edgedims[d] = len(cells[d])

        # 2. a list of tuples defining multiple arrays
        # i.e. cells = [(np.linspace(-12,0,100),np.linspace(-4,0,50)),(np.linspace(0,12,100),np.linspace(0,4,50))]
        if isinstance(cells,list):
            self.eval_type = 'freesample'

            # variable controlling whether grids have already been constructed or not
            _gridsconstructed = False

            # how many grid boundaries are we being handed?
            self.ngrids = len(cells)

            # check if we are being passed grids already
            if isinstance(cells[0],np.ndarray):
                _gridsconstructed = True

            # infer the dimensionality: this will work whether the input is a single cell or list of cells
            if _gridsconstructed:
                self.dim = cells[0].shape[-1]
            else:
                self.dim = len(cells[0])

            # check special case dimensionality
            if self.dim==1:
                raise NotImplementedError("LintSampler: For 1d problems, the grid must be continuous (i.e. you cannot pass multiple grids).")

            # create holding arrays
            grids                 = [] # the grids constructed from the inputs (no need to expose)
            self.gridshape        = [] # the shape of the input grids
            self.ngridentries     = [] # the number of entries in the grid
            self.nedgegridentries = [] # the number of entries in the edge grid
            self.edgearrays       = [] # the flattened grids, each [ngridentries,dim]

            # loop through the boundaries, construct the grids, and flatten
            for ngrid in range(0,self.ngrids):

                
                if _gridsconstructed:
                    # check that the grid has the same dimensionality as the others
                    if cells[ngrid].shape[-1]!=self.dim:
                        raise ValueError("LintSampler._setgrid: All input grids must have the same dimensionality.")
                    
                    grid = cells[ngrid]

                else:
                    # check that the dimensions are properly specified for the grid
                    if len(cells[ngrid])!=self.dim:
                        raise ValueError("LintSampler._setgrid: All input cells must have the same dimensionality.")

                    # construct the grid
                    grid = np.stack(np.meshgrid(*cells[ngrid], indexing='ij'), axis=-1)

                # save the grid
                grids.append(grid)

                # construct the grid boundary helpers
                self.gridshape.append(grid.shape)
                self.ngridentries.append(np.prod(grid.shape[0:self.dim]))
                self.nedgegridentries.append(np.prod(np.array(grid.shape[0:self.dim])-1))

                # flatten the grid into a [...,dim] array for passing to .pdf
                self.edgearrays.append(grid.reshape(self.ngridentries[ngrid],self.dim))
            
            self.x0 = np.vstack([grids[ngrid][tuple([slice(0,self.gridshape[ngrid][griddim]-1) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries[ngrid],self.dim)) for ngrid in range(0,self.ngrids)])
            self.x1 = np.vstack([grids[ngrid][tuple([slice(1,self.gridshape[ngrid][griddim]  ) for griddim in range(0,self.dim)])].reshape((self.nedgegridentries[ngrid],self.dim)) for ngrid in range(0,self.ngrids)])


    def _get_startend_points(self):
        """Return all combinations of start and end points for offset grids.
        
        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.

        Returns
        -------
        combinations: numpy array
            All possible combinations of starting and ending points for the
            inferred dimensionality of the problem.
        
        """

        # Create an array representing the start/end points
        arr = np.array([0, 1]) 

        # get all combinations
        combinations = np.array(np.meshgrid(*[arr]*self.dim)).T.reshape(-1,self.dim)

        # Rearrange the combinations
        sorted_combinations = np.lexsort(combinations.T[::-1])

        # Reorder the combinations
        combinations = combinations[sorted_combinations]

        return combinations
