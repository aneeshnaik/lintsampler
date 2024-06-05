import numpy as np
from functools import reduce
from .base import DensityStructure
from ..utils import _choice, _get_hypercube_corners

#TODO class docstring
#TODO method docstrings
#TODO refine by location
#TODO refine by mass
class DensityTree(DensityStructure):
    """

    ***LONGER DESCRIPTION TO GO HERE***
    
    Parameters
    ----------
    mins: 1D array-like, length ndim
        Coordinate minima along all axes (e.g., bottom-left corner in 2D).
    maxs: 1D array-like, length ndim
        Coordinate maxima along all axes (e.g., top-right corner in 2D).
    density_fn: callable
        Function to integrate. Should take in coord vector (1D, length ndim) and
        output density at that point (float).
    """
    def __init__(self, mins, maxs, density_fn, min_openings=0):
        # save arguments as attrs
        self._mins = np.array(mins)
        self._maxs = np.array(maxs)
        self.density_fn = density_fn

        # infer dimensionality and check mins/maxs shapes make sense
        self._dim = len(mins)
        if self.mins.ndim != 1:
            raise ValueError("Need one-dimensional array of minima.")
        if len(maxs) != self._dim:
            raise ValueError("mins and maxs have different lengths.")

        # construct density cache grid
        self._grid = _CacheGrid(mins, maxs, density_fn)

        # set root cell
        self._root = _TreeCell(None, 0, 0, self._grid)
        
        # full openings
        leaves = [self._root]
        for i in range(min_openings):
            new_leaves = []
            for leaf in leaves:
                leaf.create_children()
                new_leaves += list(leaf.children)
            leaves = new_leaves
            leaf_masses = [leaf.mass_raw for leaf in leaves]
            m0 = np.sum(leaf_masses)
            print(f"Full opening {i}. {len(leaves)} leaves on tree. Total raw={m0}")        
        self._leaves = leaves
        
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
        return np.array([leaf.mass_raw for leaf in self._leaves])

    def choose_cells(self, u):
        
        p = self._leaf_masses / self.total_mass
        idx = _choice(p, u)
        
        N = len(u)
        
        mins = np.zeros((N, self._dim))
        maxs = np.zeros((N, self._dim))
        corners = np.zeros((2**self._dim, N))
        for i in range(N):
            leaf = self._leaves[idx[i]]
            mins[i] = leaf.x
            maxs[i] = leaf.x + leaf.dx
            for j in range(2**self._dim):
                corners[j][i] = leaf.corner_densities[j]
        corners = tuple(corners)
        return mins, maxs, corners

    def refine_by_error(self, leaf_tol, tree_tol):
        
        leaves = self._leaves
    
        leaves_converged = False
        iter_counter = 0
        while not leaves_converged:
            iter_counter += 1
            m_tot = np.sum([leaf.mass_romberg for leaf in leaves])
            print(f"Leaf iteration {iter_counter}: {len(leaves)} leaves on tree. Total mass={m_tot}")
            new_leaves = []
            leaves_converged = True
            for leaf in leaves:
                if np.abs(np.diff(leaf.romberg_estimates)[-1]) > leaf_tol * leaf.mass_romberg:
                    leaves_converged = False
                    leaf.create_children()
                    new_leaves += list(leaf.children)
                    m_tot -= leaf.mass_romberg
                    m_tot += np.sum([child.mass_romberg for child in leaf.children])
                else:
                    new_leaves.append(leaf)
            leaves = new_leaves      

        tree_converged = False
        iter_counter = 0
        while not tree_converged:
            iter_counter += 1
            m_tot = np.sum([leaf.mass_romberg for leaf in leaves])
            errors_sq = np.array([(leaf.mass_romberg - leaf.mass_raw)**2 for leaf in leaves])
            err_rms = np.sqrt(np.average(errors_sq))
            err_tot = np.sqrt(np.sum(errors_sq))
            if err_tot / m_tot < tree_tol:
                tree_converged = True
            print(f"Tree iteration {iter_counter}: {len(leaves)} leaves on tree. Total mass={m_tot}. RMS error={err_rms}. Total error={err_tot}")
            
            ind = np.argmax(errors_sq)
            leaf = leaves.pop(ind)
            leaf.create_children()
            leaves += list(leaf.children)
        
        self._leaves = leaves
        return

    def _get_leaf_at_pos(self, pos):
        """Find leaf cell which contains given position.
        
        Walk down the tree until at leaf cell which contains given position.

        Parameters
        ----------
        pos : 1D array-like
            Position at which to get leaf cell. Length should be tree.ndim.

        Returns
        -------
        cell : TreeCell instance
            Leaf cell containing given position.        
        """
        # cast position to numpy array, check shape makes sense
        pos = np.array(pos)
        if pos.shape != (self._dim,):
            s = pos.shape
            raise TypeError(f"Shape of input pos: {s} does not make sense.")

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
    
    def __init__(self, parent, idx, level, grid):
        
        # check cell_idx in appropriate range for level
        assert idx in range(2**(level * grid.ndim))
        
        # save arguments as attrs
        self.parent = parent
        self.idx = idx
        self.level = level
        self.grid = grid
        
        # store *real* (not grid) origin and real span of cell, calculate volume
        self.x = grid._convert_corners_to_pos(self._get_origin(), self.level)
        self.dx = (grid.maxs - grid.mins) / 2**self.level
        self.vol = np.prod(self.dx)
        
        # calculate corner densities
        self.corner_densities = self.grid.eval(self._get_corners(), self.level)
        
        # trapezoid mass = volume * average(density)
        self.mass_raw = self.vol * np.average(self.corner_densities)
        
        # romberg estimates
        self.romberg_estimates = self._generate_romberg_estimates()
        self.mass_romberg = self.romberg_estimates[-1]
        
        # no children yet!
        self.children = None
        return
    
    def create_children(self):
        children = []
        for idx in self._get_child_ids():
            children.append(_TreeCell(self, idx, self.level + 1, self.grid))
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


class _CacheGrid:
    def __init__(self, mins, maxs, density_fn):
        # save arguments as attrs
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.density_fn = density_fn
        
        # infer dimensionality
        self.ndim = len(mins)
        
        # set up (currently empty) caches to save density evaluations
        self.caches = {}
        
        # number of calls to density function
        self.fn_calls = 0
        return

    def eval(self, corners, level):
        
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
            new_densities = []
            for xi in pos:
                new_densities.append(self.density_fn(xi))
                self.fn_calls += 1
            densities[~m_cached] = np.array(new_densities)
            for i, corner in enumerate(corners[~m_cached]):
                self._cache(corner, level, densities[~m_cached][i])
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
