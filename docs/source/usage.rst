Usage
=====

``lintsampler`` has two public functions: 

- ``sample``
  
  This is a very general function to draw one or several samples from one or several cells, when the densities are known on the cell corners.

  If only one cell is being considered, then the user provides the coordinate vectors (or scalars in 1D) of the first and last corners, and the known densities on the :math:`2^k` corners. The function will then draw a single sample within the cell, or a series of samples if ``N_samples`` is set.

  If several cells are being considered, then the function works by first estimating the probabilities of all cells then randomly choosing a cell (or ``N_samples`` cells, with replacement) from the probability-weighted list, then performing the linear interpolant sampling within the cell(s). Here, the user provides a batch of coordinate vectors for the first and last vertices of all cells, and a batch of densities on each of the :math:`2^k` corners of the cells. No assumptions are made about the layout or sizes of the cells: they can be overlapping, disconnected, or even concentric.

  See the :doc:`function docstring <sample>` or the example notebooks for further details and examples. 

- ``gridsample``

  This is a slightly friendlier function than ``sample``, which deals with the special case where one has a series of cells laid out in a regular (albeit not necessarily fixed) grid, and the densities are known on the grid intersections. Here, the user need only provide arrays giving the coordinates of the gridlines, and another array giving the grid of densities. The function then estimates the probabilities of all the cells in the grid, then randomly chooses a cell (or ``N_samples`` cells, with replacement) then performs linear interpolant sampling within the cell(s) to draw samples.

  See the :doc:`function docstring <gridsample>` or the example notebooks for further details and examples. 
