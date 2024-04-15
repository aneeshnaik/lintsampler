# lintsampler

When you know densities on the $2^k$ vertices of a $k$-dimensional hyperbox (or a series of such hyperboxes, e.g., the cells of a $k$-dimensional grid), linear interpolant sampling provides a technique to draw samples within the hyperbox.

`lintsampler` provides a Python implementation of linear interpolation sampling, using only `numpy`.

These pages describe how linear interpolation sampling works and how to use `lintsampler`. For a more detailed introduction to the theory, see the paper (LINK).

The source code is a public repository on [GitHub](https://github.com/aneeshnaik/lintsampler).

```{toctree}
:caption: User Guide
:hidden: true
:maxdepth: 1

installation
usage
sample
gridsample
attribution
license
```

```{toctree}
:caption: Theory
:hidden: true
:maxdepth: 1

theory/preamble.md
theory/inverse_sampling
theory/linear_interpolant.md
```

```{toctree}
:caption: Examples
:hidden: true
:maxdepth: 1

examples/1_gmm
examples/2_doughnuts
examples/3_dark_matter
```
