# TODO: Grid documentation
# TODO: Grid test suite
import numpy as np
from functools import reduce
from .utils import _is_1D_iterable, _choice
from .unitsample_kd import _unitsample_kd


class Grid:
    def __init__(self, pdf, cells, vectorizedpdf=False, pdf_args=(), pdf_kwargs={}):

        # store PDF and related parameters as attributes
        self.pdf = pdf
        self.vectorizedpdf = vectorizedpdf
        self.pdf_args = pdf_args
        self.pdf_kwargs = pdf_kwargs
        
        # 1D case: cells is 1D iterable (array, tuple, list)
        # e.g. cells = np.linspace(-4,4,50)
        if _is_1D_iterable(cells):

            # store dimensionality (1) and single edgearray and dims
            self.dim = 1
            self.edgearrays = [np.array(cells)]
            self.edgedims = (len(cells),)

        # kD case: cells is tuple of 1D iterables
        # e.g. cells = (np.linspace(-12,12,100),np.linspace(-4,4,50))
        elif isinstance(cells, tuple) and _is_1D_iterable(cells[0]):
            
            # infer dimensionality
            self.dim = len(cells)

            # loop over dimensions, store edge arrays and dims
            self.edgearrays = []
            self.edgedims = ()
            for d in range(0,self.dim):
                self.edgearrays.append(np.array(cells[d]))
                self.edgedims += (len(cells[d]),)
        
        # cells type not recognised
        else:
            raise TypeError(
                "Grid.__init__: "\
                "you must specify an evaluation domain with a single 1D "\
                "iterable or a tuple of 1D iterables. "\
                "See documentation for details."
            )
        
        # check edge arrays all monotonically increasing and finite-valued
        for arr in self.edgearrays:
            if np.any(np.diff(arr) <= 0):
                raise ValueError(
                    "Grid.__init__: Edges not monotically increasing."
                )
            if not np.all(np.isfinite(arr)):
                raise ValueError(
                    "Grid.__init__: Edges not finite-valued."
                )

        self.mins = np.array([arr[0] for arr in self.edgearrays])
        self.maxs = np.array([arr[-1] for arr in self.edgearrays])
        self.f = self._evaluate_pdf()
        self.masses = self._calculate_faverages() * self._calculate_volumes()
        self.total_mass = np.sum(self.masses)
    
    def sample(self, u):
        # TODO: docstring
        
        # randomly choose grid cell(s)
        cells = self._choose(u=u[..., -1])

        # get 2^k-tuple of densities at cell corners
        corners = self._corners(cells)

        # sample on unit hypercube
        z = _unitsample_kd(*corners, u=u[..., :-1])

        # rescale coordinates (loop over dimensions)
        for d in range(self.dim):
            e = self.edgearrays[d]
            c = cells[:, d]
            z[:, d] = e[c] + np.diff(e)[c] * z[:, d]
        
        return z

    def _evaluate_pdf(self):
        """Evaluate the pdf on a grid, handling the flag for vectorized.
        
        Parameters
        ------------
        self : LintSampler
            The LintSampler instance.

        Returns
        -----------
        evalf: np.ndarray
            The values of the pdf at the edge points of the grid.

            
        """
        # create the flattened grid for evaluation in k>1 case
        if self.dim > 1: 
            edgegrid = np.stack(np.meshgrid(*self.edgearrays, indexing='ij'), axis=-1).reshape(np.prod(self.edgedims), self.dim)
        else:
            edgegrid = self.edgearrays[0]

        # evaluate PDF over edge grid (iterate over grid if not vectorized)
        if self.vectorizedpdf:
            evalf = np.array(self.pdf(edgegrid, *self.pdf_args, **self.pdf_kwargs), dtype=np.float64)         
        else:
            npts = np.prod(self.edgedims)
            evalf = np.zeros(npts, dtype=np.float64)
            for pt in range(npts):
                evalf[pt] = self.pdf(edgegrid[pt], *self.pdf_args, **self.pdf_kwargs)

        if np.any(evalf < 0):
            raise ValueError("Grid._evaluate_pdf: Densities can't be negative")
        if not np.all(np.isfinite(evalf)):
            raise ValueError("Grid._evaluate_pdf: Detected non-finite density")

        # reshape densities to grid
        evalf = evalf.reshape(*self.edgedims)

        return evalf
    
    def _calculate_faverages(self):
        # TODO: update docstring
        """Given grid of densities, evaluated at corners, calculate cell averages.

        Parameters
        ----------
        f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
            Grid of densities evaluated at corners of k-dimensional grid.

        Returns
        -------
        average : k-dim numpy array, shape (N0 x N1 x ... x N{k-1})
            Density in each grid cell, averaging over 2**k corners. Shape is shape
            of grid.
        
        """
        # infer dimensionality and grid shape
        shape = tuple([s - 1 for s in self.f.shape])

        # loop over corners, add density contribution to sum
        # at each corner construct slice tuple, e.g. a[:-1,:-1,etc] for 1st corner
        sum = np.zeros(shape)
        slice0 = slice(-1)
        slice1 = slice(1, None)
        for i in range(2**self.dim):
            n = np.binary_repr(i, width=self.dim)
            t = ()
            for d in range(self.dim):
                t += ([slice0, slice1][int(n[d])],)
            sum += self.f[t]
        
        # div. by no. corners for average
        average = sum / 2**self.dim
        return average


    def _calculate_volumes(self):
        #TODO: update docstring
        """From a sequence of arrays of edge lines, calculate grid cell volumes.
        
        Calculates difference arrays with numpy.diff, then volumes with outer
        product.

        Parameters
        ----------
        *edgearrays : 1 or more 1D numpy arrays
            k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
            is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
            (N1+1,), (N2+1,) respectively.
        
        Returns
        -------
        vols : numpy array, k-dimensional, shape (N0 x N1 x ... x N{k-1})
            k-dimensional array containing volumes of grid cells. Shape is shape of
            grid.
        """
        diffarrays = []
        for edgearray in self.edgearrays:
            diffarrays.append(np.diff(edgearray))
        shape = tuple([d.size for d in diffarrays])
        vols = reduce(np.outer, diffarrays).reshape(shape)
        return vols
    

    
    def _choose(self, u):
        """From k-dimensional grid of densities, choose mass-weighted cell(s).

        Given a k-dimensional grid, shaped (N0 x N1 x ... x N{k-1}), the user
        specifies a sequence of k 1D arrays (lengths N0+1, N1+1, etc.) representing
        the (not necessarily evenly spaced) gridlines along each dimension, and a
        kD array (shape N0+1 x N1+1 x ...) representing the densities at the grid
        corners. This function then calculates the mass of each grid cell according
        to the trapezoid rule (i.e., the average density over all 2**k corners times
        the volume of the cell), then randomly chooses a cell (or several cell) from
        the set of cells, weighting each cell by its mass in this choice.

        Parameters
        ----------
        *edgearrays : 1 or more 1D numpy arrays
            k arrays representing 'edge lines' of k-dimensional grid. E.g., if grid
            is 3D and shaped N0 x N1 x N2, then provide 3 1D arrays, shaped (N0+1,),
            (N1+1,), (N2+1,) respectively. The edges do *not* need to be evenly
            spaced.
        f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
            Grid of densities evaluated at corners of k-dimensional grid.
        u : 1D numpy array, length N_cells
            Array of uniform samples, length equal to number of desired cells.

        Returns
        -------
        idx : 2D numpy array (N_cells x k)
            Indices along each dimension of randomly sampled cells.
        """
        # TODO: docstring
        # normalise mass and flatten into probability array
        m_norm = self.masses / self.masses.sum()
        p = m_norm.flatten()

        # choose cells
        cells = _choice(p=p, u=u)

        # unravel 1D cell indices into k-D grid indices
        idx = np.stack(np.unravel_index(cells, m_norm.shape), axis=-1)
        return idx
    
    def _corners(self, cells):
        #TODO fix docstring
        """From gridded densities, get densities on 2^k corners of given cells.

        Parameters
        ----------
        f : k-dim numpy array, shape (N0+1 x N1+1 x ... x N{k-1}+1)
            Grid of densities evaluated at corners of k-dimensional grid.
        cells : 2-d numpy array, shape (N, k)
            Grid indices of N chosen cells along the k dimensions of the grid.
            
        Returns
        -------
        corners : 2^k-tuple of 1D numpy arrays, each length N
            Densities at corners of given cells. Conventional ordering applies,
            e.g., in 3D: corners = (f000, f001, f010, f011, f100, f101, f110, f111).

        """
        # loop over 2^k corners, get densities at each
        corners = []
        for i in range(2**self.dim):
            
            # binary representation of corner, e.g. [0,0,...,0] is first corner
            n = np.binary_repr(i, width=self.dim)
            n = np.array([int(c) for c in n], dtype=int)
    
            # get densities on given corners
            idx = cells + n
            idx = np.split(idx.T, self.dim)
            idx = tuple([idxi.squeeze() for idxi in idx])
            corners.append(self.f[idx])
        return tuple(corners)
