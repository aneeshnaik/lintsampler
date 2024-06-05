from abc import ABC, abstractmethod

#TODO docstring
class DensityStructure(ABC):
    
    @abstractmethod
    def choose_cells(self, u):
        pass
    
    @property
    @abstractmethod
    def dim(self):
        pass
    
    @property
    @abstractmethod
    def mins(self):
        pass
    
    @property
    @abstractmethod
    def maxs(self):
        pass
    
    @property
    @abstractmethod
    def total_mass(self):
        pass