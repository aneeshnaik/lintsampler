# When to Use `lintsampler`

This page closely mirrors the 'Statement of Need' section of our [JOSS paper](https://joss.theoj.org/papers/133f9681a79ab47b916ecee9e183be9e).

Below is a (non-exhaustive) list of 'use cases', i.e., situations where a user might find `lintsampler` (and/or the the linear interpolant sampling algorithm underpinning it) to be preferable over random sampling techniques such as importance sampling, rejection sampling or Markov chain Monte Carlo (MCMC). MCMC in particular is a powerful class of methods with many excellent Python implementations.[^emcee] In certain use cases as described below, `lintsampler` can offer more convenient and/or more efficient sampling compared with these various techniques.

[^emcee]: For example, [emcee](https://emcee.readthedocs.io/en/stable/).

We'll assume that the target PDF the user wishes to sample from does not have its own exact sampling algorithm (such as the Box-Muller transform for a Gaussian PDF). The power of `lintsampler` lies in its applicability to arbitrary PDFs for which tailor-made sampling algorithms are not available.


## Use Cases

### 1. Expensive PDF
   
If the PDF being sampled from is expensive to evaluate and a large number of samples is desired, then `lintsampler` might be the most cost-effective option. This is because `lintsampler` does not evaluate the PDF for each sample (as would be the case for other random sampling techniques), but on the nodes of the user-chosen grid. Particularly in a low-dimensional setting where the grid does not have too many nodes, this can mean far fewer PDF evaluations. This point is demonstrated in the [first example notebook](./example_notebooks/1_gmm.ipynb).[^1]

[^1]: Similarly, there might be situations where the user is not so concerned about strict statistical representativeness but wants to generate a huge number of samples from a target PDF with the least possible computational cost (such as e.g., generating realistic point cloud distributions in video game graphics). They can use `lintsampler` with a coarse grid (so minimal PDF evaluations), then `sample()` to their heart's content.


### 2. Multimodal PDF

If the target PDF has a highly complex structure with multiple, well-separated modes, then `lintsampler` might be the *easiest* option (in terms of user configuration). In such scenarios, MCMC might struggle unless the walkers are carefully preconfigured to start near the modes. Similarly, rejection sampling or importance sampling would be highly suboptimal unless the proposal distribution is carefully chosen to match the structure of the target PDF. With `lintsampler`, the user need only ensure that the resolution of their chosen grid is sufficient to resolve the PDF structure, and so the setup remains straightforward. This is demonstrated in the [second example notebook](./example_notebooks/2_doughnuts.ipynb) in the `lintsampler` docs.[^2]

[^2]: It is worth noting that in these kinds of complex, multimodal problems, a single fixed grid might not be the most cost-effective sampling domain. For this reason, `lintsampler` also provides simple functionality for sampling over multiple disconnected grids.


### 3. PDF with large dynamic range

If the target PDF has a very large dynamic range, then the `DensityTree` object provided by `lintsampler` might be an effective solution. Here, the PDF is evaluated not on a fixed grid, but on the leaves of a tree. The tree is refined such that regions of concentrated probability are more finely resolved, based on accuracy criteria. Such an example is shown in the [third example notebook](./example_notebooks/3_dark_matter.ipynb) in the `lintsampler` docs.

### 4. Noise needs to be minimised

In Quasi-Monte Carlo (QMC) sampling, one purposefully generates more 'balanced' (and thus less random) draws from a target PDF, so that sampling noise decreases faster than $N^{-1/2}$. `lintsampler` allows easy QMC sampling with arbitrary PDFs. We are not aware of such capabilities with any other package. We give an example of using `lintsampler` for QMC in the [fourth example notebook](./example_notebooks/4_qmc.ipynb) in the `lintsampler` docs.


## 'Real World' Example

Any one of the four use cases above would serve by itself as a sufficient case for choosing `lintsampler`, but here we give an example of a real-world scenario that combines all of the use cases. It is drawn from our own primary research interests in computational astrophysics.

Much of computational astrophysics consists of large-scale high performance computational simulations of gravitating systems. For example, simulations of planets evolving and interacting within a solar system, simulations of stars interacting within a galaxy, or vast cosmological simulations in which a whole universe is grown __in silico__. 

There exists a myriad of codes used to run these simulations, each using different algorithms to solve the governing equations. One class of simulation code that has gained much attention in recent years is the class of code employing basis function expansions.[^exp] In these codes, the matter density at any point in space is represented by a sum over basis functions (not unlike a Fourier series), with the coefficients in the sum changing over space and time. As such, the matter comprising the system is represented everywhere as a smooth, continous fluid, but for many applications and/or downstream analyses of the simulated system, one needs to instead represent the system as a set of discrete particles. These particles can be obtained by drawing samples from the continuous density field.

[^exp]: For example, [EXP](https://arxiv.org/abs/2104.14577)

This is a scenario that satisfies all four of the use cases list above. To explain further:

- The PDF we are sampling from (i.e., the basis expansion representation of the matter density field) can be expensive to evaluate if a large number of terms are included in the sum.
- The PDF can be highly multimodal when the system we are simulating comprises many distinct gravitating substructures, such as stellar clusters.
- The PDF can have a large dynamic range. Astrophysical structures such as galaxies and dark matter 'haloes' often have power-law density profiles, such as the [Navarro-Frenk-White profile](https://en.wikipedia.org/wiki/Navarro%E2%80%93Frenk%E2%80%93White_profile). Further complicating the issue is that a typical dark matter halo will host several 'subhaloes', which in turn might host 'subsubhaloes', and so on. In short, a range of spatial scales needs to be resolved.
- If the particle set being sampled is to be used for further simulation, it can be helpful to draw as 'noiseless' a sample as possible for reasons of numerical stability.

For these reasons, this kind of astrophysical simulation code provides an excellent example of a 'real world' application for `lintsampler`. Here, one would cover the simulation domain with a `DensityTree` instance (or several instances, one for each primary structure), call the `refine` method to better resolve the high-density regions, then feed the tree to a `LintSampler` instance and call `sample` to generate particles. The `qmc` flag can be passed to the sampler in order to employ Quasi-Monte Carlo sampling.

## Caveats

In all use cases listed above, it is assumed that the dimension of the problem is not too high. `lintsampler` works by evaluating a given PDF on the nodes of a grid (or grid-like structure, such as a tree), so the number of evaluations (and memory occupancy) grows exponentially with the number of dimensions. As a consequence, many of the efficiency arguments given for `lintsampler` below don't apply to higher dimensional problems. We probably wouldn't use `lintsampler` in more than 6 dimensions, but there is no hard limit here: the question of how many dimensions is too many will depend on the problem at hand. 
