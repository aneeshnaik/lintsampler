# When to Use `lintsampler`

Below is a (non-exhaustive) list of situations where a user might find `lintsampler` (and/or the the linear interpolant sampling algorithm underpinning it) to be preferable over other random sampling techniques.

In all of these use cases, it is assumed that the dimension of the problem is not too high. `lintsampler` works by evaluating a given PDF on the nodes of a grid (or grid-like structure, such as a tree), so the number of evaluations (and memory occupancy) grows exponentially with the number of dimensions. As a consequence, many of the efficiency arguments given for `lintsampler` below don't apply to higher dimensional problems. We probably wouldn't use `lintsampler` in more than 6 dimensions, but there is no hard limit here: the question of how many dimensions is too many will depend on the problem at hand. 

A second assumption made below is that the target PDF the user wishes to sample from does not have its own exact sampling algorithm (such as the [Box-Muller transform](https://en.wikipedia.org/wiki/Box-Muller_transform) for a Gaussian PDF). If it does, use that instead.



1. Expensive PDF
   
   If the PDF being sampling from is expensive to evaluate and large number of samples is desired, then `lintsampler` might be the most cost-effective option. This is because `lintsampler` does not evaluate the PDF for each sample (as would be the case for importance sampling, rejection sampling or MCMC), but on the nodes of the user-chosen grid. Particularly in a low-dimensional setting where the grid does not have too many nodes, this can mean far fewer PDF evaluations. This point is demonstrated in the [first example notebook](./example_notebooks/1_gmm.ipynb).

   Similarly, there might be situations where the user is not so concerned about strict statistical representativeness but wants to generate a huge number of samples from a target PDF with the least possible computational cost (such as e.g., generating realistic point cloud distributions in video game graphics), they can use `lintsampler` with a coarse grid (so minimal PDF evaluations), then `sample()` to their heart's content.

2. Multimodal PDF

   If the PDF being sampled from has a highly complex structure with multiple, well-separated modes, then `lintsampler` might be the *easiest* option (in terms of user configuration). In such scenarios, MCMC might struggle unless the walkers are carefully preconfigured to start near the modes. Similarly, rejection sampling or importance sampling would be highly suboptimal unless the proposal distribution is carefully chosen to match well the structure of the target PDF. With `lintsampler`, the user need only ensure that the resolution of their chosen grid is sufficient to resolve the PDF structure, and so the setup remains straightforward. This is demonstrated in the [second example notebook](./example_notebooks/2_doughnuts.ipynb).

   It is worth noting that in these kinds of complex, multimodal problems, a single fixed grid might not be the most cost-effective sampling domain. For this reason, `lintsampler` also provides simple functionality for sampling over multiple disconnected grids.

3. PDF with large dynamic range

   If the target PDF has a very large dynamic range, such as the power law density profiles frequently encountered in astronomical problems, then the `DensityTree` object provided by `lintsampler` might be an effective solution. Here, the PDF is evaluated not on a fixed grid, but on the leaves of a tree which is refined such that regions of concentrated probability are more finely resolved. Such an example is shown in the [third example notebook](./example_notebooks/3_dark_matter.ipynb).

4. Quasi-Monte Carlo

   In Quasi-Monte Carlo (QMC) sampling, one purposefully generates more 'balanced' (and thus less random) draws from a target PDF, so that sampling noise decreases faster than $N^{-1/2}$. `lintsampler` allows quasi-Monte Carlo sampling with arbitrary PDFs. We are not aware of such capabilities with any other package. We give an example of using `lintsampler` for QMC in the [fourth example notebook](./example_notebooks/4_qmc.ipynb).
