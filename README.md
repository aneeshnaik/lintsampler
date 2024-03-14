# lintsampler

**Efficient random sampling via linear interpolation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/aneeshnaik/lintsampler/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/lintsampler/badge/?version=latest)](https://lintsampler.readthedocs.io/en/latest/?badge=latest)

When you know densities on the 2 endpoints of 1D interval, or the 4 corners of a 2D rectangle, or generally the $2^k$ vertices of a $k$-dimensional hyperbox (or a series of such hyperboxes, e.g., the cells of a $k$-dimensional grid), linear interpolant sampling provides a technique to draw random samples within the hyperbox. `lintsampler` provides a Python implementation of this.

See the documentation or the linear interpolant sampling paper for further details. 

## Documentation

The documentation is available at [https://lintsampler.readthedocs.io/](http://lintsampler.readthedocs.io/).

## Attribution

If using `lintsampler` for a research publication, please cite our paper: link to come.

## License

`lintsampler` is available under the MIT license. See the LICENSE file for specifics.
