===========
lintsampler
===========



When you know densities on the :math:`2^k` vertices of a :math:`k`-dimensional hyperbox (or a series of such hyperboxes, e.g., the cells of a :math:`k`-dimensional grid), linear interpolant sampling provides a technique to draw samples within the hyperbox. 

``lintsampler`` provides a Python implementation of linear interpolation sampling, using only ``numpy``.

These pages describe how linear interpolation sampling works and how to use ``lintsampler``. For a more detailed introduction to the theory, see the paper (LINK).

The source code is a public repository on `GitHub <https://github.com/aneeshnaik/lintsampler>`_.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: User Guide

   installation
   usage
   sample
   gridsample
   attribution
   license

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Examples

   examples/1_gmm
   examples/2_doughnuts
