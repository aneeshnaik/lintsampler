import numpy as np
from functools import reduce
from .base import DensityStructure
from ..utils import _choice, _get_hypercube_corners

#TODO refine by location
class DensityTree(DensityStructure):
    """Tree-like object over which density function is evaluated.

    #TODO extended description
    
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
    min_openings : int, optional
        Number of full tree openings to perform on initialisation. This is
        distinct from any further openings that happen on calling a refinement
        method.

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
    
    Examples
    --------
    #TODO
    """
    def __init__(
        self, mins, maxs, pdf,
        vectorizedpdf=False, pdf_args=(), pdf_kwargs={},
        min_openings=0, usecache=True, batch=False
    ):
        # check mins/maxs shapes make sense
        if mins.ndim != 1:
            raise ValueError(
                "DensityTree.__init__: Need one-dimensional array of minima."
            )
        if maxs.ndim != 1:
            raise ValueError(
                "DensityTree.__init__: Need one-dimensional array of maxima."
            )
        if len(maxs) != len(mins):
            raise ValueError(
                "DensityTree.__init__: mins and maxs have different lengths."
            )

        # save mins/maxs/dim as private attrs (made public via properties)
        self._mins = np.array(mins)
        self._maxs = np.array(maxs)
        self._dim = len(mins)
        self._usecache = usecache
        self.batch = batch

        # construct density cache
        self._cache = _GridCache(mins, maxs, pdf, vectorizedpdf, pdf_args, pdf_kwargs)

        # set root cell
        self.root = _TreeCell(parent=None, idx=0, level=0, grid=self._cache, usecache=self._usecache)
        
        # full openings
        leaves = [self.root]
        for _ in range(min_openings):
            new_leaves = []
            for leaf in leaves:
                leaf.create_children(batch=self.batch)
                new_leaves += list(leaf.children)
            leaves = new_leaves        
        self.leaves = leaves

        self.densities_evaluated = True
        return

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

    def refine_by_error(self, leaf_tol, tree_tol, verbose=False):
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
        leaf_tol : float
            Individual leaf error tolerance level used in the first refinement
            loop. This is a *fractional* error, i.e., a leaf is opened if
            the leaf's mass error divided by the leaf's mass is above this
            threshold.
        tree_tol : float
            Total tree error tolerance level used in the second refinement
            loop. This is a *fractional* error, i.e., a leaf is opened if
            the tree's total mass error divided by the tree's total mass is
            above this threshold.
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
        leaves_converged = True
        counter = 0
        while not leaves_converged:
            counter += 1
            new_leaves = []
            leaves_converged = True
            for leaf in self.leaves:
                err = np.abs(np.diff(leaf.romberg_estimates)[-1])
                if err > leaf_tol * leaf.mass_romberg:
                    leaves_converged = False
                    leaf.create_children(self.batch)
                    new_leaves += list(leaf.children)
                else:
                    new_leaves.append(leaf)
            self.leaves = new_leaves
            if verbose:
                print(f"End of leaf iteration {counter}: {len(self.leaves)} leaves on tree. Total mass={self.total_mass}")

        # TREE LOOP:
        # repeatedly open most erroneous leaf in tree until whole tree converged
        tree_converged = False
        counter = 0
        while not tree_converged:
            counter += 1

            # total romberg mass
            m_tot = np.sum([leaf.mass_romberg for leaf in self.leaves])
            
            # leaf errors            
            errs_sq = np.array([(leaf.mass_romberg - leaf.mass_raw)**2 for leaf in self.leaves])
            err_tot = np.sqrt(np.sum(errs_sq))
            if err_tot / m_tot < tree_tol:
                tree_converged = True
            
            # open most erroneous leaf
            ind = np.argmax(errs_sq)
            leaf = self.leaves.pop(ind)
            leaf.create_children(self.batch)
            self.leaves += list(leaf.children)
            if verbose:
                print(
                    f"End of tree iteration {counter}: "\
                    f"{len(self.leaves)} leaves on tree. "\
                    f"Total mass={self.total_mass}. "\
                    f"Fractional error={err_tot / m_tot}."
                )
    
        return

    def get_leaf_at_pos(self, pos):
        """Find leaf cell which contains given position.
        
        Walk down the tree until at leaf cell which contains given position.

        Parameters
        ----------
        pos : 1D array-like
            Position at which to get leaf cell. Length should be tree.ndim.

        Returns
        -------
        cell : TreeCell
            Leaf cell containing given position.        
        """
        # cast position to numpy array
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
            raise ValueError("Requested pos falls outside tree.")

        # start at root, walk down tree until at leaf
        cell = self._root
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
        assert idx in range(2**(level * grid.ndim))
        
        # save arguments as attrs
        self.parent = parent
        self.idx = idx
        self.level = level
        self.grid = grid
        self.usecache = usecache
        
        # store *real* (not grid) origin and real span of cell, calculate volume
        self.x = grid._convert_corners_to_pos(self._get_origin(), self.level)
        self.dx = (grid.maxs - grid.mins) / 2**self.level
        self.vol = np.prod(self.dx)
        
        if hold_eval:
            # create a list of corners we need
            self.positions = self.grid._convert_corners_to_pos(self._get_corners(), self.level)
            return
            
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
        self.corner_densities = self.grid.eval(self._get_corners(), self.level, self.usecache)
        
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
            _nverticies = 2**(self.grid.ndim)
            allverticies = np.zeros([_nverticies**2,self.grid.ndim])

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
        return
    
    def interpolate_pos(self, pos):
        return self._interpolate_unitcube((pos - self.x) / self.dx)
    
    def estimate_descendant_mass(self, idx, level):
        midpt = self._get_descendant_midpoint(idx, level)
        fmid = self._interpolate_unitcube(midpt)
        return fmid * self.vol / 2**((level - self.level) * self.grid.ndim)
    
    def _interpolate_unitcube(self, rescaled_pos):
        d1 = np.ones(self.grid.ndim) - rescaled_pos
        d0 = rescaled_pos
        dx = reduce(np.outer, list(np.stack((d1, d0), axis=-1))).flatten()
        return np.sum(self.corner_densities * dx)
    
    def _get_origin(self):
        return self.grid.get_origin_from_cell_idx(self.idx, self.level)
    
    def _get_corners(self):
        mins = self._get_origin()
        maxs = mins + 1
        return _get_hypercube_corners(mins, maxs)
    
    def _get_child_ids(self):
        ndim = self.grid.ndim
        idx = self.idx
        return np.arange(idx * 2**ndim, (idx + 1) * 2**ndim)
    
    def _get_descendant_midpoint(self, idx, level):
        cur_level = level
        cur_idx = idx
        ndim = self.grid.ndim
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
        raws = [self.mass_raw]
        ancestor = self
        for i in range(self.level):
            ancestor = ancestor.parent
            raws.append(ancestor.estimate_descendant_mass(self.idx, self.level))
        raws = np.array(raws)

        estimates = [raws[0]]
        level_estimates = np.copy(raws)
        for i in range(self.level):
            divisor = 4**(i + 1) - 1
            level_estimates = level_estimates[:-1] - np.diff(level_estimates) / divisor
            estimates.append(level_estimates[0])
        return estimates


class _GridCache:
    def __init__(self, mins, maxs, pdf, vectorizedpdf, pdf_args, pdf_kwargs):
        # save arguments as attrs
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs
        
        # infer dimensionality
        self.ndim = len(mins)
        
        # set up (currently empty) caches to save density evaluations
        self.caches = {}
        
        # number of calls to density function: are we using this somewhere?
        self.fn_calls = 0
        return

    def eval_positions(self, pos):
        
        return self.pdf(pos,*self.pdf_args,**self.pdf_kwargs)
    

    def eval(self, corners, level, usecache):
        
        if usecache:
            # check which corners already in cache
            m_cached = np.zeros(len(corners), dtype=bool)
            densities = np.zeros(len(corners), dtype=np.float64)
            for i, corner in enumerate(corners):
                f = self._retrieve_from_cache(corner, level)
                if f:
                    m_cached[i] = True
                    densities[i] = f
            
            # if any corners not in cache, evaluate density fn and cache
            if not m_cached.all():
                pos = self._convert_corners_to_pos(corners[~m_cached], level)
                

                if self.vectorizedpdf:
                    densities[~m_cached] = self.pdf(pos,*self.pdf_args,**self.pdf_kwargs)
                    self.fn_calls += pos.size

                else:
                    new_densities = []
                    for xi in pos:
                        new_densities.append(self.pdf(xi,*self.pdf_args,**self.pdf_kwargs))
                        self.fn_calls += 1
                    densities[~m_cached] = np.array(new_densities)

                for i, corner in enumerate(corners[~m_cached]):
                    self._cache(corner, level, densities[~m_cached][i])

        else:
            densities = self.pdf(self._convert_corners_to_pos(corners, level),*self.pdf_args,**self.pdf_kwargs)

        return densities

    def get_origin_from_cell_idx(self, idx, level):
        assert idx in range(2**(level * self.ndim))
        if level == 0:
            return np.zeros(self.ndim, dtype=np.int64)
        else:
            parent_idx = idx // 2**self.ndim
            orthant = np.unravel_index(idx % 2**self.ndim, [2] * self.ndim)
            return 2 * self.get_origin_from_cell_idx(parent_idx, level-1) + orthant

    def _convert_corners_to_pos(self, corners, level):
        dx = (self.maxs - self.mins) / 2**level
        return self.mins + corners * dx
    
    def _retrieve_from_cache(self, corner, level):
        corner = tuple(corner)
        if level != 0 and all(x % 2 == 0 for x in corner):
            return self._retrieve_from_cache(tuple(i // 2 for i in corner), level-1)
        else:
            if level in self.caches:
                try:
                    val = self.caches[level][corner]
                    return val
                except KeyError:
                    return None
            else:
                return None
    
    def _cache(self, corner, level, value):
        corner = tuple(corner)
        if level != 0 and all(x % 2 == 0 for x in corner):
            self._cache(tuple(i // 2 for i in corner), level-1, value)
        else:
            if level not in self.caches:
                self.caches[level] = {}
            self.caches[level][corner] = value
        return

