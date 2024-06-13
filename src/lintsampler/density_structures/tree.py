import numpy as np
from functools import reduce
from warnings import warn
from .base import DensityStructure
from ..utils import _choice, _get_hypercube_corners, _get_grid_origin_from_cell_idx


class DensityTree(DensityStructure):
    """Tree-like object over which density function is evaluated.

    ``DensityTree`` uses the parameters ``mins`` and ``maxes`` to construct the
    k-dimensional root cell of the tree, evaluating densities on the 2^k corners
    of the cell with the given ``pdf`` function. If ``min_openings`` is
    non-zero, then the root is opened into a series of children, and the
    children are successively opened, each time with the ``pdf`` function being
    evaluated on the corners. After construction, the ``refine`` method
    can be used to further open the tree. During all of these cell openings,
    the tree uses a cache to ensure that a density evaluated on a parent is
    not re-evaluated on a child (although this functionality can be turned off
    with the ``usecache`` flag).

    See the examples below for the various usage patterns.
    
    Parameters
    ----------
    mins : scalar or 1D iterable
        For k-dimensional structure, length-k array giving coordinate minima
        along all axes (e.g., bottom-left corner in 2D). In one dimension, can
        simply provide single number.
    maxs : scalar or 1D iterable
        For k-dimensional structure, length-k array giving coordinate maxima
        along all axes (e.g., top-right corner in 2D). In one dimension, can
        simply provide single number.
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
    min_openings : int, optional
        Number of full tree openings to perform on initialisation. This is
        distinct from any further openings that happen on refining. Default is
        1.
    usecache : bool, optional
        Whether to use the cache to store density evaluations, so that densities
        evaluated on a parent are not later re-evaluated on a child. It is
        generally recommended to use this, unless a PDF function is so fast that
        that re-evaluations are cheaper than cache lookups. The cache is
        not used if ``batch=True`` (see below). Default is True.
    batch : bool, optional
        When creating the 2^k child cells of a given parent cell, whether to
        evaluate their various vertex densities in a single batch. This is
        incompatible with the density cache, so is switched off by default.
        However, there might be circumstances where ``batch=True`` plus
        ``usecache=False`` is faster than vice versa. Default is False.

    Attributes
    ----------
    mins : 1D array-like
        See corresponding parameter in constructor. Enforced by base class
        ``DensityStructure``.
    maxs : 1D array-like
        See corresponding parameter in constructor. Enforced by base class
        ``DensityStructure``.
    dim : int
        Dimensionality of structure. Enforced by base class
        ``DensityStructure``.
    total_mass : float
        Total probability mass of structure, summed over all leaves in tree.
        Enforced by base class ``DensityStructure``.
    root : ``TreeCell``
        Root cell of tree.
    leaves : list
        List of ``TreeCell`` instances representing leaves of tree.
    usecache : bool
        See corresponding parameter in constructor.
    batch : bool
        See corresponding parameter in constructor.

    Examples
    --------
    
    These examples demonstrate the various ways to set up and use an instance
    of ``DensityTree``.
    
    - A basic tree in 1D
    
      In the simplest case, we only need three parameters to construct a tree:
      the minimum coordinate bound(s), the maximum coordinate bound(s), and the
      pdf function to evaluate on the tree. In the 1D case, we can just use
      single numbers for the coordinate bounds, and in this example we'll
      use the ``scipy`` implementation of the standard normal distribution for
      the PDF.
      
      >>> from scipy.stats import norm
      >>> tree = DensityTree(-10, 10, pdf=norm.pdf)
      
      This sets up a one-dimensional tree, bounded by [-10, 10]. Let's
      interrogate some of the attributes describing the geometry of the tree:
      
      >>> tree.mins
      array([-10])
      >>> tree.maxs
      array([10])
      >>> tree.dim
      1
      
      ``mins`` and ``maxs`` are the same as the input parameters, but cast to
      arrays. ``dim`` is an integer describing the dimensionality of the space.

      The attribute `root` gives a reference to the root cell of the tree:
      
      >>> tree.root
      <lintsampler.density_structures.tree._TreeCell at 0x7fd1b2b34740>
      
      This is an instance of the private _TreeCell class.
      
      By default, ``min_openings`` is 1, so the root cell will be opened into
      (in one dimension) 2 children. These can be accessed via the ``leaves``
      attribute:
      
      >>> tree.leaves
      [<lintsampler.density_structures.tree._TreeCell at 0x7fc6f2d6f560>,
       <lintsampler.density_structures.tree._TreeCell at 0x7fc6f3cbf320>]

      We can get the total probability mass of the tree via the ``total_mass``
      attribute:
      
      >>> tree.total_mass
      3.989422804014327
      
      Because we're using a properly normalised Gaussian PDF, this should 
      actually be unity. However, our estimate of the integral is quite bad
      because we are essentially using the trapezoid approximation to the
      integral with only two trapezoids.
    
    - 1D tree with more full openings
    
      In the example above, we saw that the tree was not able to give a good
      approximation for the integral under the normal distribution with only
      two leaves, because the resolution is too low. One way to improve this
      is to increase the ``min_openings`` parameter, which sets how many
      times the initial root cell is opened. 
      
      >>> from scipy.stats import norm
      >>> tree = DensityTree(-10, 10, pdf=norm.pdf, min_openings=3)
      >>> len(tree.leaves)
      8
      >>> tree.total_mass
      1.0850046370702153
      
      Now, we have done 3 full openings, giving us 2^3=8 leaves. As a
      consequence, the estimate of the total probability mass is greatly
      improved.
      
    - 3D tree
    
      We can go beyond one dimension by providing ``mins`` and ``maxs``
      parameters which are iterables rather than scalars. For the PDF here, we
      can use the multivariate normal implementation in `scipy`:
      
      >>> from scipy.stats import multivariate_normal
      >>> pdf = multivariate_normal(np.zeros(3), np.eye(3)).pdf
      >>> tree = DensityTree([10, 100, 1000], [20, 200, 2000], pdf=pdf)
      >>> tree.dim
      3
      
    - Refinement
    
      As well as ``min_openings``, another way to increase the resolution of the
      tree is the ``refine`` method. Instead of opening all the leaves in the
      tree, this uses a Romberg integration method to try to find the leaves
      which would be most helpful to open (i.e., some combination of the most
      massive and most erroneous leaves). The only required parameter of
      ``refine`` is ``tree_tol``: a fractional tolerance level, which can be
      roughly understood as the tolerance level on the total mass error of the
      tree. See the docstring of that method for more details about its various
      parameter options and settings.
      
      >>> from scipy.stats import norm
      >>> tree = DensityTree(-10, 10, pdf=norm.pdf)
      >>> tree.refine(1e-3)
      >>> tree.total_mass
      1.0020293093918315
    
    - Vectorized PDF calls
    
      By default, it is assumed that the PDF function takes single k-vectors and
      returns scalar densities. However, if a PDF function is vectorized such
      that it takes a batch of positions shaped (N, k) and returns a batch of
      densities shaped (N,) then we can let ``DensityTree`` know via the
      ``vectorizedpdf`` flag, which will hopefully speed up tree construction
      and refinement.
      
      As it happens, the ``scipy`` PDF functions we were using above are all
      vectorized, so:
      
      >>> DensityTree(-10, 10, pdf=norm.pdf, vectorizedpdf=True)

    - ``get_leaf_at_pos``
    
      The method ``get_leaf_at_pos`` is a useful method which can find the leaf
      on the tree that contains a given position.
      
      >>> from scipy.stats import multivariate_normal
      >>> pdf = multivariate_normal(mean=np.ones(2), cov=np.eye(2)).pdf
      >>> tree = DensityTree([-5, -5], [5, 5], pdf=pdf, vectorizedpdf=True)
      >>> tree.refine(1e-2)
      >>> tree.get_leaf_at_pos(np.array([1.02, -3.74]))
      <lintsampler.density_structures.tree._TreeCell at 0x7f9003228680>
      
      This returns a reference to the relevant leaf.
    """
    def __init__(
        self, mins, maxs, pdf,
        vectorizedpdf=False, pdf_args=(), pdf_kwargs={},
        min_openings=1, usecache=True, batch=False
    ):
        # cast to arrays
        if not hasattr(mins, "__len__"):
            mins = np.array([mins])
            maxs = np.array([maxs])
        else:
            mins = np.array(mins)
            maxs = np.array(maxs)

        # check same length
        if len(maxs) != len(mins):
            raise ValueError(
                "DensityTree.__init__: mins and maxs have different lengths."
            )

        # check mins/maxs all monotonic, finite-valued, 1D
        for arr in [mins, maxs]:
            if arr.ndim != 1:
                raise ValueError(
                    "DensityTree.__init__: Need 1D array of minima/maxima."
                )
            if np.any((maxs - mins) <= 0):
                raise ValueError(
                    "DensityTree.__init__: "\
                    "Coordinates not monotically increasing."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "DensityTree.__init__: "\
                    "Coordinates not finite-valued."
                )
        
        # warn user if cache and batch both True
        if batch and usecache:
            warn(
                "DensityTree.__init__: " \
                "`usecache` set to True but cache not used if `batch=True`"
            )

        # save mins/maxs/dim as private attrs (made public via properties)
        self._mins = np.array(mins)
        self._maxs = np.array(maxs)
        self._dim = len(mins)
        self.usecache = usecache
        self.batch = batch

        # construct density cache
        gc = _GridCache(mins, maxs, pdf, vectorizedpdf, pdf_args, pdf_kwargs)

        # set root cell
        self.root = _TreeCell(parent=None, idx=0, level=0, grid=gc, usecache=self.usecache)
        
        # full openings
        leaves = [self.root]
        for _ in range(min_openings):
            new_leaves = []
            for leaf in leaves:
                leaf.create_children(batch=self.batch)
                new_leaves += list(leaf.children)
            leaves = new_leaves        
        self.leaves = leaves

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
        return np.sum(self._leaf_masses)
    
    @property
    def _leaf_masses(self):
        return np.array([leaf.mass_raw for leaf in self.leaves])

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
        # flatten leaf masses into probability array
        p = self._leaf_masses / self.total_mass
        
        # choose leaf indices; idx: 1D (N,)
        idx = _choice(p, u)
        
        # loop over leaves, get mins/maxes corners
        N = len(u)
        mins = np.zeros((N, self._dim))
        maxs = np.zeros((N, self._dim))
        corners = np.zeros((2**self._dim, N))
        for i in range(N):
            leaf = self.leaves[idx[i]]
            mins[i] = leaf.x
            maxs[i] = leaf.x + leaf.dx
            for j in range(2**self._dim):
                corners[j][i] = leaf.corner_densities[j]
                
        # cast corner array into tuple
        corners = tuple(corners)

        return mins, maxs, corners

    def refine(self, tree_tol, leaf_tol=0.01, leaf_mass_contribution_tol=0.01, verbose=False):
        """Refine the tree by opening leaves with large estimated mass errors.

        Refinement uses a strategy based on Romberg integration, and happens
        in two stages. In the first stage, the tree leaves are repeatedly looped
        over and opened if their individual 'errors' (estimated from the
        difference between the last two Romberg-integrated masses) are above the
        given tolerance threshold. In the second stage, the most erroneous leaf
        on the tree is repeatedly found and opened until the total error on the
        tree is below the given threshold. 

        Parameters
        ----------
        tree_tol : float
            Total tree error tolerance level used in the second refinement
            loop. This is a *fractional* error, i.e., a leaf is opened if
            the tree's total mass error divided by the tree's total mass is
            above this threshold.
        leaf_tol : float, optional
            Individual leaf error tolerance level used in the first refinement
            loop. This is a *fractional* error, i.e., a leaf is opened if
            the leaf's mass error divided by the leaf's mass is above this
            threshold. Default: 0.01, or 1% fractional mass error per leaf.
        leaf_mass_contribution_tol: float, optional
            Individual leaf fractional mass below which the leav will not be 
            refined. The parameter acts as a prefactor to compare leaves to the
            mean mass of all leaves in the tree. Default: 0.01, or leaves with
            1% of the mass of the mean leaf.
        verbose : bool, optional
            If True, print messages at every loop iteration. Default: False. 

        Returns
        -------
        None
        """
        # raise error if only leaf is root (i.e. no full openings on init)
        if len(self.leaves) == 1:
            raise RuntimeError(
                "DensityTree.refine_by_error: "\
                "Romberg refinement strategy needs at least two tree levels "\
                "to estimate mass errors. Instantiate tree with min_openings>0."
            )
            
        if verbose:
            print(f"Pre-loop: {len(self.leaves)} leaves on tree. Total mass={self.total_mass}")
        
        # LEAF LOOP:
        # repeatedly loop over leaves until each leaf has converged Romberg mass
        # Initialize the convergence flag and counter
        leaves_converged = False
        counter = 0
        
        # Initialize arrays for masses and errors
        masses_romberg = np.array([leaf.mass_romberg for leaf in self.leaves])
        errors = np.array([np.abs(np.diff(leaf.romberg_estimates)[-1]) for leaf in self.leaves])

        while not leaves_converged:
            counter += 1
            new_leaves = []
            leaves_converged = True
            
            # Check if any leaf does not meet the convergence criterion
            non_converged_indices = np.where( (errors > leaf_tol * masses_romberg) &
                                              (masses_romberg > leaf_mass_contribution_tol * np.nanmean(masses_romberg)))[0]

            if len(non_converged_indices) > 0:
                leaves_converged = False
                # Iterate over non-converged leaves in reverse order to avoid index issues when deleting
                for i in reversed(non_converged_indices):
                    leaf = self.leaves.pop(i)
                    leaf.create_children(self.batch)
                    new_leaves += list(leaf.children)
                    # Delete the non-converged leaf's mass and error
                    masses_romberg = np.delete(masses_romberg, i)
                    errors = np.delete(errors, i)
                # Add new leaves to the tree
                self.leaves += new_leaves
                # Append new leaves' masses and errors
                new_masses_romberg = np.array([leaf.mass_romberg for leaf in new_leaves])
                new_errors = np.array([np.abs(np.diff(leaf.romberg_estimates)[-1]) for leaf in new_leaves])
                masses_romberg = np.concatenate((masses_romberg, new_masses_romberg))
                errors = np.concatenate((errors, new_errors))
            
            if verbose:
                print(f"End of leaf iteration {counter}: {len(self.leaves)} leaves on tree. Total mass={np.sum(masses_romberg)}, with mean leaf mass={np.nanmean(masses_romberg)}")

        # TREE LOOP:
        # repeatedly open most erroneous leaf in tree until whole tree converged
        tree_converged = False
        counter = 0

        # Precompute initial values
        masses_romberg = np.array([leaf.mass_romberg for leaf in self.leaves])
        masses_raw = np.array([leaf.mass_raw for leaf in self.leaves])

        while not tree_converged:
            counter += 1

            # total romberg mass
            m_tot = np.sum(masses_romberg)

            # leaf errors
            errs_sq = (masses_romberg - masses_raw) ** 2
            err_tot = np.sqrt(np.sum(errs_sq))
            if err_tot / m_tot < tree_tol:
                tree_converged = True
                break

            # open most erroneous leaf
            ind = np.argmax(errs_sq)
            leaf = self.leaves.pop(ind)
            leaf.create_children(self.batch)

            # Update masses_romberg and masses_raw lists
            masses_romberg = np.delete(masses_romberg, ind)
            masses_raw = np.delete(masses_raw, ind)

            children_masses_romberg = np.array([child.mass_romberg for child in leaf.children])
            children_masses_raw = np.array([child.mass_raw for child in leaf.children])

            masses_romberg = np.concatenate((masses_romberg, children_masses_romberg))
            masses_raw = np.concatenate((masses_raw, children_masses_raw))

            self.leaves += list(leaf.children)

            if verbose:
                print(
                    f"End of tree iteration {counter}: "
                    f"{len(self.leaves)} leaves on tree. "
                    f"Total mass={m_tot}. "
                    f"Fractional error={err_tot / m_tot}."
                )

    def get_leaf_at_pos(self, pos):
        """Find leaf cell which contains given position.
        
        Walk down the tree until at leaf cell which contains given position.

        Parameters
        ----------
        pos : scalar or 1D iterable
            Position at which to get leaf cell. Can be scalar or length-1
            iterable in 1D, otherwise should be iterable with length equal to
            dimensionality of tree (``tree.dim``).

        Returns
        -------
        cell : TreeCell
            Leaf cell containing given position.        
        """
        # cast position to numpy array
        if not hasattr(pos, "__len__"):
            pos = np.array([pos])
        else:        
           pos = np.array(pos)

        # check shape makes sense
        if pos.shape != (self._dim,):
            raise ValueError(
                "DensityTree.get_leaf_at_pos:"\
                f"Shape of input pos: {pos.shape} does not make sense."
            )

        # rescale position to unit cube, check falls inside tree        
        rpos = (pos - self.mins) / (self.maxs - self.mins)
        if not np.all((rpos >= 0) & (rpos <= 1)):
            raise ValueError(
                "DensityTree.get_leaf_at_pos: "\
                "Requested pos falls outside tree."
            )

        # start at root, walk down tree until at leaf
        cell = self.root
        mids = 0.5 * np.ones(self._dim)
        while cell.children:
            
            # get orthant of pos, convert to child number, set cell to child
            b = (rpos > mids).astype(int)
            ind = b.dot(2**np.arange(b.size)[::-1])
            cell = cell.children[ind]
            
            # rescale pos again to dimensions of child
            rpos = 2 * rpos - b
        
        return cell


class _TreeCell:
    
    def __init__(self, parent, idx, level, grid, hold_eval=False, usecache=True):
        
        # check cell_idx in appropriate range for level
        msg = f"lintsampler.tree._TreeCell: "\
              "Index {idx} is out of range for level {level} "\
              "with grid dimension {grid.dim}."
        assert (0 <= idx < 2**(level * grid.dim)), msg
        
        # save arguments as attrs
        self.parent = parent
        self.idx = idx
        self.level = level
        self.grid = grid
        self.usecache = usecache
        
        # store *real* (not grid) origin and real span of cell, calculate volume
        go = _get_grid_origin_from_cell_idx(self.idx, self.level, self.grid.dim)
        self.x = grid.convert_corners_to_pos(go, self.level)
        self.dx = (grid.maxs - grid.mins) / 2**self.level
        self.vol = np.prod(self.dx)
        
        if hold_eval:
            # get the *real* (not grid) vertices
            self.positions = grid.convert_corners_to_pos(self._get_grid_corners(), self.level)
            
        else:
            self._evaluate_cell()
            
        # no children yet!
        self.children = None
        
        return
    

    def distribute_evaluations(self, corner_densities):

        self.corner_densities = corner_densities

        # trapezoid mass = volume * average(density)
        self.mass_raw = self.vol * np.average(self.corner_densities)
        
        # romberg estimates
        self.romberg_estimates = self._generate_romberg_estimates()
        self.mass_romberg = self.romberg_estimates[-1]
        return

    
    def _evaluate_cell(self):
        # calculate corner densities
        self.corner_densities = self.grid.eval(self._get_grid_corners(), self.level, self.usecache)
        
        # check densities all non-negative and finite
        if np.any(self.corner_densities < 0):
            raise ValueError(
                "_TreeCell._evaluate_cell: Densities can't be negative"
            )
        if not np.all(np.isfinite(self.corner_densities)):
            raise ValueError(
                "_TreeCell._evaluate_cell: Detected non-finite density"
            )
        
        # trapezoid mass = volume * average(density)
        self.mass_raw = self.vol * np.average(self.corner_densities)
        
        # romberg estimates
        self.romberg_estimates = self._generate_romberg_estimates()
        self.mass_romberg = self.romberg_estimates[-1]
        return
    
    def create_children(self, batch=False):

        children = []

        if batch:
            # create structure for the verticies to be stored
            _nverticies = 2**(self.grid.dim)
            allverticies = np.zeros([_nverticies**2,self.grid.dim])

            # create cells; get positions to batch evaluate
            for cellnumber,idx in enumerate(self._get_child_ids()):

                # create cells, but don't evaluate yet (usecache is untouched in this case)
                childcell = _TreeCell(self, idx, self.level + 1, self.grid, hold_eval=True)
                children.append(childcell)

                # get the positions to evaluate for all children at once and store in array
                allverticies[cellnumber*_nverticies:(cellnumber+1)*_nverticies,:] = childcell.positions

            # batch evaluate the densities
            densities = self.grid.eval_positions(allverticies)

            # send evaluations back to children
            for cellnumber in range(_nverticies):
                children[cellnumber].distribute_evaluations(densities[cellnumber*_nverticies:(cellnumber+1)*_nverticies])
                        
        else:
            # default behaviour: do all child evaluations while creating children.
            for idx in self._get_child_ids():

                children.append(_TreeCell(self, idx, self.level + 1, self.grid, usecache=self.usecache))
        
        self.children = tuple(children)
    
    def estimate_descendant_mass(self, idx, level):
        midpt = self._get_descendant_midpoint(idx, level)
        fmid = self._interpolate_unitcube(midpt)
        return fmid * self.vol / 2**((level - self.level) * self.grid.dim)
    
    def _interpolate_unitcube(self, rescaled_pos):
        d1 = np.ones(self.grid.dim) - rescaled_pos
        d0 = rescaled_pos
        dx = reduce(np.outer, list(np.stack((d1, d0), axis=-1))).flatten()
        return np.sum(self.corner_densities * dx)
    
    def _get_grid_corners(self):
        mins = _get_grid_origin_from_cell_idx(self.idx, self.level, self.grid.dim)
        maxs = mins + 1
        return _get_hypercube_corners(mins, maxs)
    
    def _get_child_ids(self):
        ndim = self.grid.dim
        idx = self.idx
        return np.arange(idx * 2**ndim, (idx + 1) * 2**ndim)
    
    def _get_descendant_midpoint(self, idx, level):
        cur_level = level
        cur_idx = idx
        ndim = self.grid.dim
        origin = np.zeros(ndim)
        while cur_level > self.level:
            orthant = np.unravel_index(cur_idx % 2**ndim, [2] * ndim)
            pos_wrt_parent = 0.5 * np.array(orthant)
            origin = pos_wrt_parent + origin / 2
            cur_level -= 1
            cur_idx = cur_idx // 2**ndim
        midpt = origin + 1 / 2**(level - self.level + 1)
        return midpt
    
    def _generate_romberg_estimates(self):

        # Store frequently accessed attributes in local variables
        level = self.level
        idx = self.idx

        raws = [self.mass_raw]
        ancestor = self
        for i in range(self.level):
            ancestor = ancestor.parent
            raws.append(ancestor.estimate_descendant_mass(idx, level))
        raws = np.array(raws)

        estimates = [raws[0]]
        level_estimates = np.copy(raws)
        for i in range(level):
            divisor = 4**(i + 1) - 1
            level_estimates = level_estimates[:-1] - np.diff(level_estimates) / divisor
            estimates.append(level_estimates[0])
        return estimates



class _GridCache:
    """Manager and cache for grid-based evaluations of a given PDF function.

    Parameters
    ----------
    mins : 1D array-like
        For k-dimensional structure, length-k array giving coordinate minima
        along all axes (e.g., bottom-left corner in 2D).
    maxs : 1D array-like
        For k-dimensional structure, length-k array giving coordinate maxima
        along all axes (e.g., top-right corner in 2D).
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
    mins : ndarray
        See corresponding parameter in constructor.
    maxs : ndarray
        See corresponding parameter in constructor.
    pdf : callable
        See corresponding parameter in constructor.
    vectorizedpdf : bool
        See corresponding parameter in constructor.
    pdf_args : tuple
        See corresponding parameter in constructor.
    pdf_kwargs : dict
        See corresponding parameter in constructor.
    dim : int
        Dimensionality of the grid.
    
    Notes
    -----
    _GridCache actually stores a series of caches in the private attribute
    _levelcaches. This is a dict with integer keys, corresponding to tree levels
    (i.e., grid refinement levels). So, _levelcaches[0] is the cache for the
    single-cell root grid, _levelcaches[1] is the cache for the 2^k-cell first
    level, etc.
    
    Each of these caches is itself a dict, with tuple keys and float values.
    The keys give the grid coordinates of the corners on the given level, and
    the values are the cached PDF evaluations at the given corners. The grid
    coordinates are integer grid indices, so in 2D the root level will have
    corners (0, 0), (0, 1), (1, 0) and (1, 1), while the first level will
    have (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1) and 
    (2, 2). However, several of these latter corners will appear in the
    root level, namely (0, 0), (0, 2), (2, 0), and (2, 2), and so won't be
    repeated in the level-1 cache. In general, at any level other than root,
    if a coordinate tuple is all even-valued (including zero), then one must
    instead consider the previous level.
    """
    def __init__(self, mins, maxs, pdf, vectorizedpdf, pdf_args, pdf_kwargs):
        # save arguments as attrs
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs
        
        # infer dimensionality
        self.dim = len(mins)
        
        # set up (currently empty) dict to store density caches
        # see Notes in class docstring for how levelcaches work
        self._levelcaches = {}

    def eval_positions(self, pos):
        
        return self.pdf(pos,*self.pdf_args,**self.pdf_kwargs)
    

    def eval(self, corners, level, usecache):
        
        if usecache:
            # check which corners already in cache
            m_cached = np.zeros(len(corners), dtype=bool)
            densities = np.zeros(len(corners), dtype=np.float64)
            for i, corner in enumerate(corners):
                f = self._retrieve_from_cache(corner, level)
                if f is not None:
                    m_cached[i] = True
                    densities[i] = f
            
            # if any corners not in cache, evaluate density fn and cache
            if not m_cached.all():
                pos = self.convert_corners_to_pos(corners[~m_cached], level)
                
                # if 1D, squeeze (N, 1) -> (N)
                if self.dim == 1:
                    pos = pos[..., 0]

                if self.vectorizedpdf:
                    try:
                        densities[~m_cached] = self.pdf(pos,*self.pdf_args,**self.pdf_kwargs)
                    except TypeError:
                        raise ValueError(
                            "_GridCache._eval: "\
                            "pdf function does not return appropriate shape."
                        )

                else:
                    new_densities = []
                    for xi in pos:
                        new_densities.append(self.pdf(xi,*self.pdf_args,**self.pdf_kwargs))
                    try:
                        densities[~m_cached] = np.array(new_densities)
                    except TypeError:
                        raise ValueError(
                            "_GridCache._eval: "\
                            "pdf function does not return appropriate shape."
                        )

                for i, corner in enumerate(corners[~m_cached]):
                    self._cache(corner, level, densities[~m_cached][i])

        else:
            densities = self.pdf(self.convert_corners_to_pos(corners, level),*self.pdf_args,**self.pdf_kwargs)
        return densities

    def convert_corners_to_pos(self, corners, level):
        """Convert grid corners to physical positions based on the grid level.

        Parameters
        ----------
        corners : array-like
            2D array of ints, shape (N, k), where N is the number of points to
            convert and k is the dimensionality of the grid. Each row of
            corners gives the coordinates of the given point in the grid
            coordinate system, i.e. the integers spanning 0 to 2^level.
        level : int
            The grid refinement level at which the corners are to be evaluated.

        Returns
        -------
        positions : ``numpy`` array
            2D array of floats, same shape as input corners, giving the physical
            positions corresponding to the given grid corners.
        """
        dx = (self.maxs - self.mins) / 2**level
        return self.mins + corners * dx
    
    def _retrieve_from_cache(self, corner, level):
        """
        Retrieve a cached value for a grid corner at a given grid level.

        Parameters
        ----------
        corner : iterable
            Sequence of ints giving the grid coordinates at which to retrieve
            an evaluation from the cache.
        level : int
            The level of refinement of the grid.

        Returns
        -------
        value : float or None
            The cached density value or None if not found.
        """
        # convert to tuple
        corner = tuple(corner)
        
        # if corner is all even (and not at root level), search parent level
        # else search this level
        if level != 0 and all(x % 2 == 0 for x in corner):
            return self._retrieve_from_cache(tuple(i // 2 for i in corner), level-1)
        else:
            if level in self._levelcaches:
                try:
                    val = self._levelcaches[level][corner]
                    return val
                except KeyError:
                    return None
            else:
                return None
    
    def _cache(self, corner, level, value):
        """Caches a PDF evaluation for a given corner at a given level.

        Parameters
        ----------
        corner : iterable
            Sequence of ints giving the grid coordinates of the corner at which
            to cache the PDF evaluation.
        level : int
            The level of refinement of the grid.
        value : float
            The density value to cache.

        Returns
        -------
        None
        """
        # cast to tuple
        corner = tuple(corner)

        # except on root, can't be even vertex (would've been cached on parent)
        msg = "_GridCache._cache: "\
              f"Corner {corner} on level {level} should've cached on parent"
        assert not (level != 0 and all(x % 2 == 0 for x in corner)), msg
        
        # add to cache
        if level not in self._levelcaches:
            self._levelcaches[level] = {}
        self._levelcaches[level][corner] = value
