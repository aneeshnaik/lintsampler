import numpy as np
import warnings
from scipy.stats.qmc import QMCEngine, Sobol
from .grid import SamplingGrid
from .utils import _check_N_samples, _is_1D_iterable, _choice, _check_hyperbox_overlap
from .sampling import _grid_sample


class LintSampler:
    # TODO: update examples in docstring
    # TODO: update docstring parameters/attrs etc
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

    def __init__(
        self, pdf, cells,
        seed=None, vectorizedpdf=False, pdf_args=(), pdf_kwargs={},
        qmc=False, qmc_engine=None
    ):
        """Initialise a LintSampler instance.

        Parameters
        ----------
        pdf : function
            Probability density function from which to draw samples.

        cells : single array, tuple of arrays or list of tuples of arrays
            If a single array, the boundary values for a 1D grid. If a tuple of
            arrays, the boundary values for a grid with dimensionality of the
            length of the tuple. If a list of tuples of arrays, the boundary
            values for an arbitrary number of grids with dimensionality the
            length of the tuples.

        seed : {None, int, ``numpy.random.Generator``}, optional
            Seed for ``numpy`` random generator. Can be random generator itself,
            in which case it is left unchanged. Can also be the seed to a
            default generator. Default is None, in which case new default
            generator is created. See ``numpy`` random generator docs for more
            information.

        vectorizedpdf : boolean
            if True, assumes that the pdf passed is vectorized (i.e. can accept [...,k]-shaped arguments).
            if False, assumes that the pdf passed accepts k arguments.
        
        qmc : bool, optional
            Whether to use Quasi-Monte Carlo sampling. Default is False.
    
        qmc_engine : {None, scipy.stats.qmc.QMCEngine}, optional
            QMC engine to use if qmc flag above is True. Should be subclass of
            scipy QMCEngine, e.g. qmc.Sobol. Should have dimensionality k+1, because
            first k dimensions are used for lintsampling, while last dimension is
            used for cell choice (this happens even if only one cell is given).
            Default is None. In that case, if qmc is True, then a scrambled Sobol
            sequence is used.

        Returns
        -------
        None


        Attributes
        ----------
        pdf : function
        vectorizedpdf : boolean

        """
        # set pdf-related parameters as attributes
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs

        # set up the sampling grid under the hood
        self._set_grids(cells=cells)
        
        # configure random state according to given random seed and QMC params
        self._set_random_state(seed, qmc, qmc_engine)
    
    def _set_random_state(self, seed, qmc, qmc_engine):
        """Parse QMC-related parameters and set as attributes.
        
        Parameters
        ----------
        
        seed : {None, int, ``numpy.random.Generator``}, optional
            Seed for ``numpy`` random generator. Can be random generator itself,
            in which case it is left unchanged. Can also be the seed to a
            default generator. Default is None, in which case new default
            generator is created. See ``numpy`` random generator docs for more
            information.

        qmc : bool, optional
            Whether to use Quasi-Monte Carlo sampling. Default is False.
    
        qmc_engine : {None, scipy.stats.qmc.QMCEngine}, optional
            QMC engine to use if qmc flag above is True. Should be subclass of
            scipy QMCEngine, e.g. qmc.Sobol. Should have dimensionality k+1, because
            first k dimensions are used for lintsampling, while last dimension is
            used for cell choice (this happens even if only one cell is given).
            Default is None. In that case, if qmc is True, then a scrambled Sobol
            sequence is used.

        Returns
        -------
        None
        
        Attributes
        ----------
        qmc : boolean
        qmc_engine : None or ``scipy`` QMCEngine, used if qmc is True
        rng : ``numpy`` random generator, used if qmc is False

        """
        # store QMC flag as attribute
        self.qmc = qmc

        # set up numpy RNG; store as attribute
        self.rng = np.random.default_rng(seed)
        
        # if using quasi-MC, configure qmc engine, else use numpy RNG
        if self.qmc:
            
            # default: scrambled Sobol
            if qmc_engine is None:
                qmc_engine = Sobol(d=self.dim + 1, bits=32, seed=seed)
            
            # user-provided QMC engine
            elif isinstance(qmc_engine, QMCEngine):
                
                # check dimension appropriate
                if qmc_engine.d != self.dim + 1:
                    raise ValueError(
                        "LintSampler.__init__: " \
                        f"qmc_engine inconsistent: expected {self.dim + 1}."
                    )
                
                # warn if qmc engine provided and RNG seed provided
                if seed is not None:
                    warnings.warn(
                        "LintSampler.__init__: " \
                        "pre-condigured qmc_engine provided, so given random "\
                        "seed won't be used except for final array shuffle."
                    )

            # QMC engine type not recognized
            else:
                raise TypeError("qmc_engine must be QMCEngine instance or None")

            # store attribute
            self.qmc_engine = qmc_engine
        else:

            # if qmc engine provided but QMC flag off, warn
            if qmc_engine is not None:
                warnings.warn(
                    "LintSampler.__init__: " \
                    "provided qmc_engine won't be used as qmc switched off."
                )

    def _generate_usamples(self, N):
        """Generate uniform samples (N x dim + 1), either with RNG or QMC engine."""
        if self.qmc:
            u = self.qmc_engine.random(N)
        else:
            u = self.rng.random((N, self.dim + 1))
        return u        

    def sample(self, N_samples=None):
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
        _check_N_samples(N_samples)

        # generate uniform samples (N_samples, k+1) if N_samples, else (1, k+1)
        # first k dims used for lintsampling, last dim used for cell choice
        if N_samples:
            u = self._generate_usamples(N_samples)
        else:
            u = self._generate_usamples(1)
        
        # check that the function can be evaluated at the passed-in points (i.e. needs to take the dimensions as an argument)
        #if self.eval_type == 'gridsample':
        if self.ngrids == 1:

            # call the gridded sampler
            X = _grid_sample(self.grids[0], u)

        else:
            grid_masses = np.array([g.total_mass for g in self.grids])
            p = grid_masses / grid_masses.sum()

            grid_choice, cdf = _choice(p, u[:, -1], return_cdf=True)
            cdf = np.append(0, cdf)
            starts = cdf[:-1]
            ends = cdf[1:]
            X = np.empty((0, self.dim))
            for i in range(self.ngrids):
                m = grid_choice == i
                if m.any():
                    usub = u[grid_choice == i]
                    usub[:, -1] = (usub[:, -1] - starts[i]) / (ends[i] - starts[i])
                    X = np.vstack((X, _grid_sample(self.grids[i], usub)))

        # final shuffle (important if using QMC with multiple grids)
        if N_samples:
            self.rng.shuffle(X, axis=0)

        # squeeze down to scalar / 1D if appropriate
        if not N_samples and (self.dim == 1):
            X = X.item()
        elif not N_samples:
            X = X[0]
        elif (self.dim == 1):
            X = X.squeeze()

        return X

    def reset_grid(self, cells):
        """Reset the sampling grid(s) without changing the pdf.
        
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
        self._set_grids(cells=cells)

    def _set_grids(self, cells):
        """Construct the cells for sampling.

        Parameters
        ----------
        self : LintSampler
            The LintSampler instance.
        cells : single array, tuple of arrays or list of tuples of arrays
            If a single array, the boundary values for a 1D grid. If a tuple of arrays, the
            boundary values for a grid with dimensionality of the length of the tuple. If
            a list of tuples of arrays, the boundary values for an arbitrary number of
            grids with dimensionality the length of the tuples.

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
        edgedims   : tuple of integers, optional, used if eval_type == 'gridsample'
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
        
        # if cells is a non-1D list, then multiple grids, otherwise 1 grid
        gargs = {
            'pdf': self.pdf,
            'vectorizedpdf': self.vectorizedpdf,
            'pdf_args': self.pdf_args,
            'pdf_kwargs': self.pdf_kwargs,
        }
        if isinstance(cells, list) and not _is_1D_iterable(cells):
            self.ngrids = len(cells)
            self.grids = [SamplingGrid(cells=ci, **gargs) for ci in cells]
        else:
            self.ngrids = 1
            self.grids = [SamplingGrid(cells=cells, **gargs)]
        self.dim = self.grids[0].dim
    
        # loop over grid pairs and check no overlap
        for i in range(self.ngrids - 1):
            for j in range(i + 1, self.ngrids):
                overlap = _check_hyperbox_overlap(
                    self.grids[i].mins, self.grids[i].maxs,
                    self.grids[j].mins, self.grids[j].maxs,
                )
                if overlap:
                    raise ValueError(
                        "LintSampler._set_grids: " \
                        f"Grids {i} and {j} spatially overlapping."
                    )

        # TODO: test if grids have different dim
