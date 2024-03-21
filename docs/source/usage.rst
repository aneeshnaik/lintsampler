Usage
=====

``lintsampler`` has two public functions: 

- ``sample``
  
  This is a very general function to draw one or several samples from one or several cells, when the densities are known on the cell corners.

  If only one cell is being considered, then the user provides the coordinate vectors (or scalars in 1D) of the first and last corners, and the known densities on the :math:`2^k` corners. The function will then draw a single sample within the cell, or a series of samples if ``N_samples`` is set.
  
  For example, to draw six samples within a single 3D cell, with first and last corners at :math:`(x, y, z) = (10, 100, 1000)` and :math:`(20, 200, 2000)` respectively:
  
  .. code-block:: python

    >>> x0 = np.array([10, 100, 1000])
    >>> x1 = np.array([20, 200, 2000])
    >>> f = np.random.uniform(size=8)
    >>> sample(x0, x1, *f, N_samples=6)
    array([[  12.63103673,  186.7514952 , 1716.6187807 ],
           [  14.67375968,  116.20984414, 1557.59629547],
           [  11.47055697,  178.41650558, 1592.18260186],
           [  12.41780309,  105.28009531, 1436.39525998],
           [  13.44764381,  152.57623376, 1880.55963378],
           [  18.5522151 ,  133.87092063, 1558.85620176]])

  If several cells are being considered, then the function works by first estimating the probabilities of all cells then randomly choosing a cell (or ``N_samples`` cells, with replacement) from the probability-weighted list, then performing the linear interpolant sampling within the cell(s). Here, the user provides a batch of coordinate vectors for the first and last vertices of all cells, and a batch of densities on each of the :math:`2^k` corners of the cells. No assumptions are made about the layout or sizes of the cells: they can be overlapping, disconnected, or even concentric.
  
  For example, to draw 7 samples from a series of 5 disconnected (albeit touching at corners) cells:
  
  .. code-block:: python

    >>> x0 = np.array([[10, 100], [20, 200], [30, 300], [40, 400], [50, 500]])
    >>> x1 = np.array([[20, 200], [30, 300], [40, 400], [50, 500], [60, 600]])
    >>> f = tuple(np.random.uniform(size=5) for i in range(4))
    >>> sample(x0, x1, *f, N_samples=7)
    array([[ 59.8448691 , 598.90797448],
           [ 11.74636295, 145.18978709],
           [ 30.22930825, 343.08761123],
           [ 40.11901342, 470.0128441 ],
           [ 37.06187148, 304.1873569 ],
           [ 58.81175529, 506.3814689 ],
           [ 37.5928607 , 303.74784732]])

  See the :doc:`function docstring <sample>` or the example notebooks for further details and examples. 

- ``gridsample``

  This is a slightly friendlier function than ``sample``, which deals with the special case where one has a series of cells laid out in a regular (albeit not necessarily fixed) grid, and the densities are known on the grid intersections. Here, the user need only provide arrays giving the coordinates of the gridlines, and another array giving the grid of densities. The function then estimates the probabilities of all the cells in the grid, then randomly chooses a cell (or ``N_samples`` cells, with replacement) then performs linear interpolant sampling within the cell(s) to draw samples.

  For example, to draw 5 samples from a 2D grid with 32 x 64 cells (so 33 and 65 grid lines along the two dimensions):  
  
  .. code-block:: python

    >>> x = np.linspace(0, 10, 33)
    >>> y = np.linspace(100, 200, 65)
    >>> f = np.random.uniform(size=(33, 65))
    >>> gridsample(x, y, f=f, N_samples=5)
    array([[1.35963966e-01, 1.38182930e+02],
           [6.52704300e+00, 1.63109912e+02],
           [4.35226761e+00, 1.49753235e+02],
           [3.56093155e+00, 1.48548481e+02],
           [1.31163401e+00, 1.59335676e+02]])

  See the :doc:`function docstring <gridsample>` or the example notebooks for further details and examples. 
