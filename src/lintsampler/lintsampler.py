
import numpy as np
from .gridsample import gridsample
from .freesample import freesample

class LintSampler:

    def __init__(self, pdf, grid=(), rngseed=42):
        """
        
        grid needs to be a tuple of the edges

        1. grid is a tuple
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

            # override the inferred dimensionality
            self.dim = 1

            # set the arrays
            if len(grid)==3:
                self.edgearrays = np.linspace(grid[0],grid[1],grid[2])
                self.edgedims = grid[2]
            else:
                self.edgearrays = grid
                self.edgedims = len(grid)

            self.eval_type='gridsample'

        # 1. tuples defining the array -> make arrays, pass to gridsample
        if isinstance(grid[0],tuple):

            # keep track of the edge arrays
            self.edgearrays = []

            # keep track of the dimensions
            self.edgedims = np.ones(self.dim,dtype='int')

            # create the grid
            for d in range(0,self.dim):
                self.edgearrays.append(np.linspace(grid[d][0],grid[d][1],grid[d][2]))
                self.edgedims[d] = grid[d][2]

            self.eval_type = 'gridsample'

        # two switches for an array input
        if isinstance(grid[0],np.ndarray):

            # 2. the grid values themselves -> gridsample
            if grid[0].ndim == 1:

                # keep track of the edge arrays
                self.edgearrays = []

                # keep track of the dimensions
                self.edgedims = np.ones(self.dim,dtype='int')

                # store the grid
                for d in range(0,self.dim):
                    self.edgearrays.append(grid[d])
                    self.edgedims[d] = len(grid[d])

                self.eval_type = 'gridsample'

            # 3. the boundaries of the grid
            else:
                self.eval_type = 'freesample'

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
                X = gridsample(*self.edgearrays,f=evalf,N_samples=N_samples,seed=self.rng)

            else: 
                # the 1d case: no flattening needed
                evalf = self.pdf(self.edgearrays,*funcargs)

                X = gridsample(self.edgearrays,f=evalf,N_samples=N_samples,seed=self.rng)

            return X


        elif self.eval_type == 'freesample':

            raise NotImplementedError("LintSampler: freesample to be implemented.")

            #x0,x1 = None,None

            #corners = None
            
            # X = greesample(x0,x1,*corners)

            #return X


        else:
            raise ValueError("LintSampler.sample: eval_type is expected to be either gridsample or freesample.")

        pass
