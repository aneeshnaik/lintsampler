import numpy as np
import warnings
from scipy.stats.qmc import QMCEngine, Sobol
from .grid import DensityGrid
from .utils import _is_1D_iterable, _choice, _check_hyperbox_overlap, _all_are_instances
from .sampling import _grid_sample


class LintSampler:
    """Linear interpolant sampler for density function defined on grid(s).

    #TODO extended description
    #TODO update examples

    Parameters
    ----------
    cells : iterable or `DensityGrid`
        Coordinate grid(s) to draw samples over. Several forms are available. If
        using a single coordinate grid, then `cells` can be any of:
        - a 1D iterable (array, list, tuple) representing the cell edges of a 1D
        grid. So, if the grid has N cells, then the iterable should have N+1
        elements.
        - a tuple of 1D iterables, representing the cell edges of a kD grid. If
        the grid has (N1, N2, ..., Nk) cells, then the tuple should have length
        k, and the 1D arrays in the tuple should have lengths N1+1, N2+1, ...,
        Nk+1.
        - a DensityGrid instance, either with densities pre-evaluated or not. If
        densities are already evaluated, `pdf` parameter should not be set and
        vice versa.
        If using multiple (non-overlapping) coordinate grids, then `cells`
        should be a list of any of the above. See the examples below for the
        various usage patterns. 
    pdf : {None, function}, optional
        Probability density function from which to draw samples. Function should
        take coordinate vector (or batch of vectors if vectorized; see
        `vectorizedpdf` parameter) and return (unnormalised) density (or batch
        of densities). Additional arguments can be passed to the function via
        `pdf_args` and `pdf_kwargs` parameters. Default is None, in which case
        it is assumed that `cells` comprises one instance or several instances
        of `DensityGrid` already having the densities evaluated (i.e.,
        `densities_evaluated=True`).
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

    Attributes
    ----------
    pdf : {None, function}
        PDF function to evaluate on grid. None if densities pre-evaluated. See
        corresponding parameter above.
    vectorizedpdf : bool
        Whether PDF function is vectorized. See corresponding parameter above.
    pdf_args : tuple
        Additional positional arguments for PDF function. See corresponding
        parameter above.
    pdf_kwargs : dict
        Additional keyword arguments for PDF function. See corresponding
        parameter above.
    dim : int
        Dimensionality of PDF / coordinate space.
    grids : list
        List of `DensityGrid` instances corresponding to series of coordinate
        grids passed by the user in `cells` parameter. Single list element
        if only one grid passed.
    ngrids : int
        Number of coordinate grids to sample over (i.e., length of `grids`
        attribute).
    qmc : bool
        Whether to use Quasi-Monte Carlo sampling. See corresponding parameter
        above.
    rng : numpy.random.Generator
        `numpy` random generator used for generating samples. Used alongside
        `scipy` `QMCEngine` if `qmc` is True.
    qmc_engine : {None, scipy.stats.qmc.QMCEngine}
        `scipy` `QMCEngine` used for generating samples if `qmc` is True.
    
    Methods
    -------
    sample(N_samples=None)
        Draw samples from given PDF over given grid(s).
    reset_cells(cells)
        Reset sampling domain without changing PDF.
        
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
        self, cells,
        pdf=None, vectorizedpdf=False, pdf_args=(), pdf_kwargs={},
        seed=None, qmc=False, qmc_engine=None
    ):
        # set pdf-related parameters as attributes
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs

        # set up the sampling grid under the hood
        self._set_grids(cells=cells)
        
        # if given, evaluate PDF on grids, else check grids already evaluated
        if self.pdf:
            if not callable(self.pdf):
                raise TypeError(
                    "LintSampler.__init__: " \
                    f"Given PDF is not callable."
                )
            self._evaluate_pdf()
        else:
            if self.vectorizedpdf or self.pdf_args or self.pdf_kwargs:
                raise UserWarning(
                    "LintSampler.__init__: " \
                    f"PDF configuration setting given but no PDF provided."
                )
            if not self._check_grids_evaluated():
                raise ValueError(
                    "LintSampler.__init__: " \
                    f"No densities pre-evaluated on grids and no PDF provided."
                )
  
        # configure random state according to given random seed and QMC params
        self._set_random_state(seed, qmc, qmc_engine)

    def sample(self, N_samples=None):
        """Draw samples from the pdf on the constructed grid(s).
        
        This function draws a sample (or N samples) from the given PDF over the
        given grid(s). It first chooses a grid (or N grids with replacement),
        weighting them by their total mass, then similarly chooses a cell
        (or N cells), then samples from the k-linear interpolant within the
        chosen cell(s).
        
        Parameters
        ----------
        N_samples : {None, int}, optional
            Number of samples to draw. Default is None, in which case a single
            sample is drawn.

        Returns
        -------
        X : scalar, 1D array (length k OR N_samples) or 2D array (N_samples, k)
            Sample(s) from linear interpolant. Scalar if single sample (i.e.,
            N_samples is None) in 1D. 1D array if single sample in k-D OR 
            multiple samples in 1D. 2D array if multiple samples in k-D.
        
        """
        # check N_samples sensible
        if (N_samples is not None):
            if not isinstance(N_samples, int):
                raise TypeError(
                    "LintSampler.sample: "\
                    f"Expected int N_samples, got {type(N_samples)}"
                )
            elif N_samples <= 0:
                raise ValueError(
                    "LintSampler.sample: "\
                    f"Expected positive N_samples, got {N_samples}"
                )

        # generate uniform samples (N_samples, k+1) if N_samples, else (1, k+1)
        # first k dims used for lintsampling, last dim used for cell choice
        if N_samples:
            u = self._generate_usamples(N_samples)
        else:
            u = self._generate_usamples(1)
        
        # if single grid, gridsample on it
        if self.ngrids == 1:
            X = _grid_sample(self.grids[0], u)

        # if multiple grids, choose grids and rescale usamples for cell choice
        else:
            # get list of grid masses
            grid_masses = np.array([g.total_mass for g in self.grids])
            
            # normalise to probability array
            p = grid_masses / grid_masses.sum()

            # choose grids:
            # grid_choice is 1D array of grid indices, length N_samples
            # cdf is 1D array giving CDF over grids, length self.ngrids
            grid_choice, cdf = _choice(p, u[:, -1], return_cdf=True)
            
            # append 0 to start of PDF, (now 1D array, length self.ngrids+1)
            cdf = np.append(0, cdf)
            
            # extremes of CDF intervals (each len self.ngrids)
            starts = cdf[:-1]
            ends = cdf[1:]
            
            # loop over grids:
            # gridsample at each grid, rescaling usamples by CDF interval size
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

    def reset_cells(self, cells):
        """Reset the sampling grid(s) without changing the pdf.
        
        Parameters
        ----------
        cells : iterable or `DensityGrid`
            See `cells` entry in documentation for class constructor.
        
        Returns
        -------
        None
        """
        # reset grid(s)
        self._set_grids(cells=cells)
        
        # evaluate PDF on new grids / check grids already evaluated
        if self.pdf:
            self._evaluate_pdf()
        else:
            self._check_grids_evaluated()

    def _set_grids(self, cells):
        """Configure the grid(s) for sampling.

        Parameters
        ----------
        cells : iterable or `DensityGrid`
            See `cells` entry in documentation for class constructor.

        Returns
        -------
        None
        """
        # cells cases:
        # - cells is a single pre-made density grid
        # - cells is a list of ditto
        # - cells is a list of: 1D iterables or tuples of 1D iterables
        # - cells is a single 1D iterable / tuple of 1D iterables
        if isinstance(cells, DensityGrid):
            self.ngrids = 1
            self.grids = [cells]
        elif isinstance(cells, list) and _all_are_instances(cells, DensityGrid):
            self.ngrids = len(cells)
            self.grids = cells
        elif isinstance(cells, list) and not _is_1D_iterable(cells):
            self.ngrids = len(cells)
            self.grids = [DensityGrid(cells=ci) for ci in cells]
        else:
            self.ngrids = 1
            self.grids = [DensityGrid(cells=cells)]

        # get dimensionality of problem from first grid
        self.dim = self.grids[0].dim

        # check that all grids have same dimension 
        for grid in self.grids[1:]:
            if grid.dim != self.dim:
                d1 = grid.dim
                d2 = self.dim
                raise ValueError(
                    "LintSampler._set_grids: "\
                    f"Grids have mismatched dimensions: {d1} and {d2}"
                )
    
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

    def _evaluate_pdf(self):
        """Loop over DensityGrid instances and evaluate PDF on each.

        Returns
        -------
        None
        """
        for grid in self.grids:
            grid.evaluate(
                self.pdf, self.vectorizedpdf, self.pdf_args, self.pdf_kwargs
            )
            
    def _check_grids_evaluated(self):
        """Loop over grids and check densities already evaluated.

        Returns
        -------
        grids_evaluated : bool
            Whether all grids in self.grids have densities already evaluated.
        """
        for grid in self.grids:
            if not grid.densities_evaluated:
                return False
        return True

    def _set_random_state(self, seed, qmc, qmc_engine):
        """Configure random number generators and set as attributes.
        
        Parameters
        ----------        
        seed : {None, int, ``numpy.random.Generator``}
            See documentation for `seed` in class constructor.
        qmc : bool
            See documentation for `qmc` in class constructor.    
        qmc_engine : {None, scipy.stats.qmc.QMCEngine}
            See documentation for `qmc_engine` in class constructor.

        Returns
        -------
        None
        """
        # store QMC flag as attribute
        self.qmc = qmc

        # set up numpy RNG; store as attribute
        self.rng = np.random.default_rng(seed)
        
        # if using quasi-MC, configure qmc engine
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

        # if qmc engine provided but QMC flag off, warn
        else:
            if qmc_engine is not None:
                warnings.warn(
                    "LintSampler.__init__: " \
                    "provided qmc_engine won't be used as qmc switched off."
                )

        # set attribute
        self.qmc_engine = qmc_engine

    def _generate_usamples(self, N):
        """Generate array of uniform samples ~U(0,1).
        
        Parameters
        ----------        
        N : int
            Number of uniform samples to draw.
        
        Returns
        -------
        u : array
            2D array of uniform samples shaped (N, dim+1), where `dim` is
            dimensionality of grid (see self.grid attribute).
        """
        if self.qmc:
            u = self.qmc_engine.random(N)
        else:
            u = self.rng.random((N, self.dim + 1))
        return u
