from abc import ABC, abstractmethod


class DensityStructure(ABC):
    """Abstract base class for density structures.
    
    This class provides an interface for density structures that can be used
    to choose cells based on a given input and to retrieve various properties
    related to the structure's dimensions, boundaries, and total mass.
    """    
    @abstractmethod
    def choose_cells(self, u):
        """Choose cells given 1D array of uniform samples.
        
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
        pass
    
    @property
    @abstractmethod
    def dim(self):
        """Dimension of the density structure.
        
        Returns
        -------
        int
            The dimension of the structure.
        """
        pass
    
    @property
    @abstractmethod
    def mins(self):
        """Minimum boundary values (i.e., first corner) of the structure.
        
        Returns
        -------
        array_like
            1D array, length-k, giving the coordinates of the first corner of
            the k-dimensional structure.
        """
        pass
    
    @property
    @abstractmethod
    def maxs(self):
        """Maximum boundary values (i.e., last corner) of the structure.
        
        Returns
        -------
        array_like
            1D array, length-k, giving the coordinates of the last corner of
            the k-dimensional structure.
        """
        pass
    
    @property
    @abstractmethod
    def total_mass(self):
        """Total probability mass of the structure.
        
        Returns
        -------
        float
            The total probability mass, summed over all the cells of the
            structure.
        """
        pass