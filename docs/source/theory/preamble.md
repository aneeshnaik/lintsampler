# Preamble

The main purpose of `lintsampler` is drawing samples from a $k$-dimensional density function $f(\mathbf{x})$, where the values of the density function are only known at the corners of a $k$-dimensional rectilinear cell (i.e., a *hyperbox*), or a series of such cells. It achieves this by *linear interpolant sampling,* i.e., it makes the approximation that the density function in the interior of the cell(s) is given by the $k$-linear interpolant. Then, it is straightforward to obtain samples very efficiently via inverse transform sampling.

A note about the density: `lintsampler`'s algorithm does not require that $f(\mathbf{x})$ be a properly normalised probability density; it is sufficient to specify the density function up to a normalising constant. Throughout the rest of this Theory section of the documentation, we use the symbol $f$ to denote an unnormalised density function, and $p$ to denote a true PDF.

The [next section](./inverse_sampling) describes how inverse transform sampling works in general, then the [section after that](./linear_interpolant) describes the linear interpolant sampling algorithm. It concludes with a self-contained summary, so a reader in a hurry can [skip there](./linear_interpolant.md#summary). The [final section](./worked_example.md) gives a worked example.

