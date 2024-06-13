import numpy as np
from warnings import warn
from scipy.stats.qmc import QMCEngine, Sobol
from .density_structures.grid import DensityGrid
from .density_structures.base import DensityStructure
from .utils import _is_1D_iterable, _choice, _check_hyperbox_overlap, _all_are_instances
from .sampling import _grid_sample


class LintSampler:
    """Linear interpolant sampler for arbitrary probability density function.

    ``LintSampler`` takes a primary argument, ``domain``, which is the region
    within which sampling takes place. After instantiation, the ``sample``
    method realises the random sampling of the given ``pdf`` on the domain. The
    sampling may either take place at random (default), or in a low-discrepancy
    sequence (with ``qmc=True``).
 
    See the parameters below for additional control options, and further
    examples below for the various usage patterns.

    Parameters
    ----------
    
    domain : iterable or ``DensityStructure``
        Coordinate grid(s) to draw samples over. Several forms are available. If
        using a single coordinate grid, then ``domain`` can be any of:
        
        - a 1D iterable (array, list, tuple) representing the cell edges of a 1D
          grid. So, if the grid has N cells, then the iterable should have N+1
          elements.
        - a tuple of 1D iterables, representing the cell edges of a kD grid. If
          the grid has (N1, N2, ..., Nk) cells, then the tuple should have
          length k, and the 1D arrays in the tuple should have lengths 
          N1+1, N2+1, ..., Nk+1.
        - a ``DensityStructure`` subclass instance. In this case, ``pdf``
          parameter should not be set, as a PDF has already been evaluated over
          the ``DensityStructure``.
        
        If using multiple (non-overlapping) coordinate grids, then ``domain``
        should be a list of any of the above. See the examples below for the
        various usage patterns.
    
    pdf : {``None``, function}, optional
        Probability density function from which to draw samples. Function should
        take coordinate vector (or batch of vectors if vectorized; see
        ``vectorizedpdf`` parameter) and return (unnormalised) density (or batch
        of densities if vectorized). Additional arguments can be passed to the
        function via ``pdf_args`` and ``pdf_kwargs`` parameters. Default is
        ``None``, in which case it is assumed that ``domain`` parameter 
        comprises one instance or a list of several instances of 
        ``DensityStructure``.
    
    vectorizedpdf : bool, optional
        if ``True``, assumes that the pdf function is vectorized, i.e., it
        accepts  (..., ``dim``)-shaped batches of coordinate vectors and returns
        (...)-shaped batches of densities. If ``False``, assumes that the pdf
        function simply accepts (``dim``,)-shaped coordinate vectors (or floats
        in the univariate case) and returns single densities. Default is
        ``False``.
    
    pdf_args : tuple, optional
        Additional positional arguments to pass to pdf function; function
        call is ``pdf(position, *pdf_args, **pdf_kwargs)``. Default is empty
        tuple (no additional positional arguments).
    
    pdf_kwargs : dict, optional
        Additional keyword arguments to pass to pdf function; function call
        is ``pdf(position, *pdf_args, **pdf_kwargs)``. Default is empty dict
        (no additional keyword arguments).
    
    seed : {``None``, int, ``numpy.random.Generator``}, optional
        Seed for ``numpy`` random generator. Can be random generator itself,
        in which case it is left unchanged. Can also be an integer seed for a
        generator instance. Default is ``None``, in which case new default
        generator is created. See ``numpy`` random generator docs for more
        information.
    
    qmc : bool, optional
        Whether to use Quasi-Monte Carlo sampling. Default is ``False``.
    
    qmc_engine : {``None``, ``scipy.stats.qmc.QMCEngine``}, optional
        Quasi-Monte Carlo engine to use if ``qmc`` flag above is True. Should be
        subclass of ``scipy`` ``QMCEngine``, e.g. ``qmc.Sobol``. Should have
        dimensionality ``dim``+1, because first ``dim`` dimensions are used for
        lintsampling, while last dimension is used for cell choice (this happens
        even if only one cell is given). Default is ``None``. In that case, if
        qmc is True, then a scrambled Sobol sequence is used.

    
    Attributes
    ----------
    
    pdf : {``None``, function}
        PDF function to evaluate on grid. ``None`` if densities pre-evaluated.
        See corresponding parameter above.
    
    vectorizedpdf : bool
        Whether ``pdf`` function is vectorized. See corresponding parameter
        above.
    
    pdf_args : tuple
        Additional positional arguments for ``pdf`` function. See corresponding
        parameter above.
    
    pdf_kwargs : dict
        Additional keyword arguments for ``pdf`` function. See corresponding
        parameter above.
    
    dim : int
        Dimensionality of PDF / coordinate space.
    
    grids : list
        List of ``DensityStructure`` instances corresponding to series of
        sampling domains passed in ``domain`` parameter. Single list element if
        only one domain passed.
    
    ngrids : int
        Number of domains to sample over (i.e., length of ``grids``
        attribute).
    
    qmc : bool
        Whether to use Quasi-Monte Carlo sampling. See corresponding parameter
        above.
    
    rng : ``numpy.random.Generator``
        Random generator used for generating samples. Used alongside Quasi-Monte
        Carlo engine if ``qmc`` is True.
    
    qmc_engine : {``None``, ``scipy.stats.qmc.QMCEngine``}
        Quasi-Monte Carlo engine used for generating samples if ``qmc`` is True.

        
    Examples
    --------
    
    These examples demonstrate the multiple ways to use ``LintSampler``. In
    each case, we'll just generate densities from a uniform distribution, but
    in general they might come from any arbitrary density function.

    1. A single sample from a 1D grid. The grid spans x=0 to x=10, and has 32
    cells (so 33 edges). 
    
    >>> cells = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform()
    >>> LintSampler(cells,pdf=pdfrandom).sample()
    6.984134639227398

    This returns a single scalar: the sampling point within the grid.
    
    2. Multiple samples from a 1D grid (same grid as previous example). Now
    also demonstrating a vectorized pdf for efficiency when N becomes large.
    
    >>> cells = np.linspace(0, 10, 33)
    >>> def pdfrandom(X): return np.random.uniform(size=X.shape[0])
    >>> LintSampler(cells,pdf=pdfrandom,vectorizedpdf=True).sample(N=4)
    array([8.16447008, 5.30536088, 8.96135879, 7.73572977])

    This returns a 1D array: the ``N`` sampling points within the grid.

    3. Single sample from a k-D grid. In this case we'll take a 2D grid, with
    32 x 64 cells (so 33 gridlines along one axis and 65 along the other).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> cells = (x,y)
    >>> def pdfrandom(X): return np.random.uniform()
    >>> LintSampler(cells,pdf=pdfrandom).sample()
    array([  7.67294632, 190.45302915])

    This returns a 1D array: the single k-D sampling point within the grid.

    4. Multiple samples from a k-D grid (same grid as previous example).
    
    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> cells = (x,y)
    >>> def pdfrandom(X): return np.random.uniform()
    >>> LintSampler(cells,pdf=pdfrandom).sample(N=5)
    array([[1.35963966e-01, 1.38182930e+02],
           [6.52704300e+00, 1.63109912e+02],
           [4.35226761e+00, 1.49753235e+02],
           [3.56093155e+00, 1.48548481e+02],
           [1.31163401e+00, 1.59335676e+02]])

    This returns a 2D array, shape (``N``, k): the ``N`` k-D
    samples within the grid.

    5. A ``DensityGrid`` instance may also be passed to any of the above examples.
    See the ``DensityGrid`` documentation for details. In this case, one need not
    pass a pdf.

    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> def pdfrandom(X): return np.random.uniform()
    >>> g = DensityGrid((x, y), pdfrandom)
    >>> LintSampler(g).sample()
    array([  1.417842  , 139.40070095])

    """
    def __init__(
        self, domain,
        pdf=None, vectorizedpdf=False, pdf_args=(), pdf_kwargs={},
        seed=None, qmc=False, qmc_engine=None
    ):
        
        # check given PDF is callable
        if pdf:
            if not callable(pdf):
                raise TypeError(
                    "LintSampler.__init__: " \
                    f"Given PDF is not callable."
                )
        else:
            if vectorizedpdf or pdf_args or pdf_kwargs:
                warn(
                    "LintSampler.__init__: " \
                    f"PDF configuration setting given but no PDF provided."
                )

        # set pdf-related parameters as attributes
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs

        # set up the sampling grids under the hood
        self._set_grids(domain=domain)

        # configure random state according to given random seed and QMC params
        self._set_random_state(seed, qmc, qmc_engine)

    def sample(self, N=None):
        """Draw samples from the pdf on the constructed grid(s).
        
        This function draws a sample (or ``N`` samples) from the given PDF over
        the given grid(s). It first chooses a grid (or ``N`` grids with
        replacement), weighting them by their total mass, then similarly chooses
        a cell (or ``N`` cells), then samples from the k-linear interpolant
        within the chosen cell(s).
        
        Parameters
        ----------
        N : {None, int}, optional
            Number of samples to draw. Default is ``None``, in which case a
            single sample is drawn.

        Returns
        -------
        X : scalar, 1D array (length k OR ``N``) or 2D array (``N``, k)
            Sample(s) from linear interpolant. Scalar if single sample (i.e.,
            ``N`` is None) in 1D. 1D array if single sample in k-D OR 
            multiple samples in 1D. 2D array if multiple samples in k-D.
        
        """
        # check N sensible
        if (N is not None):
            if not isinstance(N, int):
                raise TypeError(
                    "LintSampler.sample: "\
                    f"Expected int N, got {type(N)}"
                )
            elif N <= 0:
                raise ValueError(
                    "LintSampler.sample: "\
                    f"Expected positive N, got {N}"
                )

        # generate uniform samples (N, k+1) if N, else (1, k+1)
        # first k dims used for lintsampling, last dim used for cell choice
        if N:
            u = self._generate_usamples(N)
        else:
            u = self._generate_usamples(1)
        
        # if single grid, gridsample on it
        if self.ngrids == 1:
            X = _grid_sample(self.grids[0], u)

        # if multiple grids, choose grids and rescale usamples for cell choice
        else:
            # get list of grid masses
            grid_masses = np.array([g.total_mass for g in self.grids])
            
            # normalise to probability array, 1D array len self.ngrids
            p = grid_masses / grid_masses.sum()

            # choose grids:
            # grid_choice is 1D array of grid indices, length N            
            grid_choice = _choice(p, u[:, -1])

            # cdf is 1D array giving CDF over grids, length self.ngrids + 1
            cdf = np.append(0, p.cumsum())
            
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
        if N:
            self.rng.shuffle(X, axis=0)

        # squeeze down to scalar / 1D if appropriate
        if not N and (self.dim == 1):
            X = X.item()
        elif not N:
            X = X[0]
        elif (self.dim == 1):
            X = X.squeeze()

        return X

    def reset_domain(self, domain):
        """Reset the sampling grid(s) without changing the pdf.
        
        Parameters
        ----------
        domain : iterable or `DensityGrid`
            See ``domain`` entry in documentation for class constructor.
        
        Returns
        -------
        None
        """
        # reset grid(s)
        self._set_grids(domain=domain)


    def _set_grids(self, domain):
        """Configure the grid(s) for sampling.

        Parameters
        ----------
        domain : iterable or `DensityGrid`
            See ``domain`` entry in documentation for class constructor.

        Returns
        -------
        None
        """
        # domain cases:
        # - domain is a single pre-made density grid
        # - domain is a list of ditto
        # - domain is a list of: 1D iterables or tuples of 1D iterables
        # - domain is a single 1D iterable / tuple of 1D iterables
        if isinstance(domain, DensityStructure):
            self.ngrids = 1
            self.grids = [domain]
            if self.pdf:
                warn(
                        "LintSampler.__set_grids: " \
                        "Pre-constructed DensityStructure provided, so `pdf` "\
                        "parameter is redundant."
                    )
        elif isinstance(domain, list) and _all_are_instances(domain, DensityStructure):
            self.ngrids = len(domain)
            self.grids = domain
            # check all list items are same sort of thing
            if not _all_are_instances(domain, type(domain[0])):
                raise TypeError(
                    "LintSampler._set_grids: "\
                    f"List members of different types"
                )
            if self.pdf:
                warn(
                        "LintSampler.__set_grids: " \
                        "Pre-constructed DensityStructure provided, so `pdf` "\
                        "parameter is redundant."
                    )
        elif isinstance(domain, list) and not _is_1D_iterable(domain):
            # check PDF provided
            if not self.pdf:
                raise ValueError(
                    "LintSampler._set_grids: "\
                    f"No PDF provided and no pre-constructed DensityStructure."
                )
            # check all list items are same sort of thing
            if not _all_are_instances(domain, type(domain[0])):
                raise TypeError(
                    "LintSampler._set_grids: "\
                    f"List members of different types"
                )
            self.ngrids = len(domain)
            gargs = dict(
                pdf=self.pdf, vectorizedpdf=self.vectorizedpdf,
                pdf_args=self.pdf_args, pdf_kwargs=self.pdf_kwargs
            )
            self.grids = [DensityGrid(edges=ci, **gargs) for ci in domain]
        else:
            # check PDF provided
            if not self.pdf:
                raise ValueError(
                    "LintSampler._set_grids: "\
                    f"No PDF provided and no pre-constructed DensityStructure."
                )
            self.ngrids = 1
            self.grids = [
                DensityGrid(edges=domain, pdf=self.pdf, vectorizedpdf=self.vectorizedpdf,
                            pdf_args=self.pdf_args, pdf_kwargs=self.pdf_kwargs)
            ]

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

    def _set_random_state(self, seed, qmc, qmc_engine):
        """Configure random number generators and set as attributes.
        
        Parameters
        ----------        
        seed : {``None``, int, ``numpy.random.Generator``}
            See documentation for ``seed`` in class constructor.
        qmc : bool
            See documentation for ``qmc`` in class constructor.    
        qmc_engine : {``None``, ``scipy.stats.qmc.QMCEngine``}
            See documentation for ``qmc_engine`` in class constructor.

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
                    warn(
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
                warn(
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
            2D array of uniform samples shaped (``N``, ``dim``+1), where ``dim``
            is dimensionality of grid (``self.dim`` attribute).
        """
        if self.qmc:
            u = self.qmc_engine.random(N)
        else:
            u = self.rng.random((N, self.dim + 1))
        return u
