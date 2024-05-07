
import numpy as np
from .gridsample import _gridsample
from .freesample import _freesample

class LintSampler:

    def __init__(self, pdf, grid=(), rngseed=42):
        """
        
        pdf needs to take an input with dimension [...,k] where k is the number of dimensions

        grid needs to be a tuple of the edges

        1. grid is a tuple, with k (min,max,num) entries
        2. grid is a list of arrays
        3. grid is a list of grid points: this must be 4 d x n arrays, where d is the dimensionality and n is the number of grid points
        
        """

        # set the pdf to be widely accessible
        self.pdf = pdf

        # set the random seed
        self.rng = np.random.default_rng(rngseed)

        # infer the dimensionality of the grid
        #if all(isinstance(item, tuple) for item in grid):
        self.dim = len(grid)

        if self.dim==0:
            raise ValueError("LintSampler.__init__: you must specify an evaluation domain with a tuple of boundaries, arrays, or grid points. See documentation for details.")
            # or, we could here drop into an adaptive grid selection
            # or, we could sample on a unit hypercube with dimensions of the pdf

        # check element grid for which sampling method we will use
        
        # 0. are we in the 1d case? the input is just a single tuple or array.
        if (isinstance(grid[0],float) | (isinstance(grid[0],int))):
            self.eval_type='gridsample'

            # override the inferred dimensionality
            self.dim = 1

            # set the arrays
            if len(grid)==3:
                self.edgearrays = np.linspace(grid[0],grid[1],grid[2])
                self.edgedims = grid[2]
            else:
                self.edgearrays = grid
                self.edgedims = len(grid)


        # 1. tuples defining the array -> make arrays, pass to gridsample
        if isinstance(grid[0],tuple):
            self.eval_type = 'gridsample'

            # keep track of the edge arrays
            self.edgearrays = []

            # keep track of the dimensions
            self.edgedims = np.ones(self.dim,dtype='int')

            # create the grid
            for d in range(0,self.dim):
                self.edgearrays.append(np.linspace(grid[d][0],grid[d][1],grid[d][2]))
                self.edgedims[d] = grid[d][2]


        # two switches for an array input
        if isinstance(grid[0],np.ndarray):

            # 2. the grid values themselves -> gridsample. check the .ndim numpy attribute
            if grid[0].ndim == 1:
                self.eval_type = 'gridsample'

                # keep track of the edge arrays
                self.edgearrays = []

                # keep track of the dimensions
                self.edgedims = np.ones(self.dim,dtype='int')

                # store the grid
                for d in range(0,self.dim):
                    self.edgearrays.append(grid[d])
                    self.edgedims[d] = len(grid[d])


            # 3. the grid points themselves; each entry in grid is a separate grid 
            else:
                self.eval_type = 'freesample'
            
                # the grids must be [...,ndim]


                # infer the dimensionality: this will work whether the input is a single grid or list of grids
                self.ndim = grid[0].shape[-1]

                if self.ndim==1:
                    # really, we shouldn't ever be in this case, because it's just a grid.
                    # now we are looking at the edges of the grids, 
                    self.x0 = grid[0] # the lower boundaries for the cells
                    self.x1 = grid[1] # the upper boundaries for the cells

                else:

                    # check: is this a list of grids?
                    if isinstance(grid,list):
                        # how many grids are we being handed?
                        self.ngrids = len(grid)

                        # loop through the grids and flatten
                        self.gridshape = []
                        self.ngridentries = [] # the number of entries in the grid
                        self.nedgegridentries = [] # the number of entries in the edge grid
                        self.edgearrays = [] # the flattened grids, each [ngridentries,ndim]
                        for ngrid in range(0,self.ngrids):
                            self.gridshape.append(grid[ngrid].shape)
                            self.ngridentries.append(np.prod(grid[ngrid].shape[0:self.ndim]))
                            self.nedgegridentries.append(np.prod(np.array(grid[ngrid].shape[0:self.ndim])-1))
                            self.edgearrays.append(grid[ngrid].reshape(self.ngridentries[ngrid],self.ndim))
                            # now self.edgearrays is a list of flattened grids, and can be passed to self.pdf
                        
                        self.x0 = np.vstack([grid[ngrid][tuple([slice(0,self.gridshape[ngrid][griddim]-1) for griddim in range(0,self.ndim)])].reshape((self.nedgegridentries[ngrid],self.ndim)) for ngrid in range(0,self.ngrids)])
                        self.x1 = np.vstack([grid[ngrid][tuple([slice(1,self.gridshape[ngrid][griddim]  ) for griddim in range(0,self.ndim)])].reshape((self.nedgegridentries[ngrid],self.ndim)) for ngrid in range(0,self.ngrids)])

                    
                    else:
                        # not a list of grids, so this must be a single grid. special case handling...
                        self.ngrids = 0

                        # special case of a single grid
                        self.gridshape = grid.shape
                        self.ngridentries = np.prod(grid.shape[0:self.ndim]) # the number of entries in the grid
                        self.nedgegridentries = np.prod(np.array(grid.shape[0:self.ndim])-1) # the number of entries in the grid
                        self.edgearrays = grid.reshape(self.ngridentries,self.ndim) # the flattened grid, [nengrids,ndim]

                        self.x0 = np.vstack([grid[tuple([slice(0,self.gridshape[griddim]-1) for griddim in range(0,self.ndim)])].reshape((self.nedgegridentries,self.ndim))])
                        self.x1 = np.vstack([grid[tuple([slice(1,self.gridshape[griddim]  ) for griddim in range(0,self.ndim)])].reshape((self.nedgegridentries,self.ndim))])




    def sample(self, N_samples=None, funcargs=()):
        """
        
        # pdf needs to be treated carefully to evaluate on a specific array: how do we help the user with this?
        # do we want to pass funcargs as a tuple, or have the user define these externally?
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

            if self.ndim == 1:

                # pdf call needs to be 1d, so rearrange
                corners = (np.array([self.pdf(x[0],*funcargs) for x in self.x0]),np.array([self.pdf(x[0]) for x in self.x1]))

            else:

                # get all combinations of starting and ending corner points (only depends on dimension)
                combinations = self._get_startend_points()

                # in case of a single grid being passed:
                if self.ngrids == 0:
                    # evaluate all points on the initial input grid and reshape
                    evalfgrid = self.pdf(self.edgearrays,*funcargs).reshape(self.gridshape[0:self.ndim])

                    # now get the values corners: there will be 2^ndim of them
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
                        evalfgrid.append(self.pdf(self.edgearrays[ngrid],*funcargs).reshape(self.gridshape[ngrid][0:self.ndim]))

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

    def _get_startend_points(self):

        # Create an array representing the start/end points
        arr = np.array([0, 1]) 

        # get all combinations
        combinations = np.array(np.meshgrid(*[arr]*self.ndim)).T.reshape(-1,self.ndim)

        # Rearrange the combinations
        sorted_combinations = np.lexsort(combinations.T[::-1])

        # Reorder the combinations
        combinations = combinations[sorted_combinations]

        return combinations
