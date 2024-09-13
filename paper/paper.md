---
title: 'lintsampler: Easy random sampling via linear interpolation'
tags:
  - Python
  - statistics
  - numpy
  - random variates
  - random sampling
  - low discrepancy sequence
authors:
  - name: Aneesh P. Naik
    corresponding: true
    orcid: 0000-0001-6841-1496
    affiliation: 1
  - name: Michael S. Petersen
    orcid: 0000-0003-1517-3935
    affiliation: 1
affiliations:
 - name: Institute for Astronomy, University of Edinburgh, UK
   index: 1
date: 14 June 2024
bibliography: paper.bib
---


# Summary

`lintsampler` provides a Python implementation of a technique we term 'linear interpolant sampling': An algorithm to efficiently draw pseudo-random samples from an arbitrary probability density function (PDF). First, the PDF is evaluated on a grid-like structure. Then, it is assumed that the PDF can be approximated between grid vertices by the (multidimensional) linear interpolant. With this assumption, random samples can be efficiently drawn via inverse transform sampling [@devroye.book]. 

`lintsampler` is primarily written with `numpy`, drawing some additional functionality from `scipy`. Under the most basic usage of `lintsampler`, the user provides a PDF function and some parameters describing a grid-like structure to the `LintSampler` class, and is then be able to draw samples via the `sample` method. Additionally, there is the functionality for the user to set the random seed, employ quasi-Monte Carlo sampling, or sample within a premade grid (`DensityGrid`) or tree (`DensityTree`) structure.


# Statement of need

For a small number of well-studied probability distributions, optimised algorithms exist to draw samples with low computational overhead. However, one often wishes to draw samples from an arbitrary PDF for which no such an algorithm is available. In such situations, the method of choice is typically some flavour of Markov Chain Monte Carlo (MCMC), a powerful class of methods with many excellent Python implementations [@emcee.paper;@pymc.paper;@sgmcmcjax.paper;@pxmcmc.paper]. One of the drawbacks of MCMC techniques is that they typically require a degree of tuning during the setup (e.g. choice of the proposal distribution, initial walker positions, etc.), and a degree of inspection afterward to check for convergence. This additional work is a price worth paying for many use cases, but can be excessive in scenarios where the user is less concerned with strict sampling accuracy or minimising PDF evaluations, and would prefer a simpler means to generate an approximate sample.

`lintsampler` was designed with such situations in mind. In the simplest use case, the user needs only to provide a PDF function and some one-dimensional arrays representing a grid, and a set of samples will be generated. Compared with MCMC, there is rather less work involved on the part of the user, but there are some compensating disadvantages. First, some care needs to be taken to ensure the grid has sufficient resolution for the use case. Second, in high dimensional scenarios with finely resolved grids, the PDF might be evaluated many more times than the one with MCMC.

We anticipate `lintsampler` finding a place in many applications in scientific research and other areas underpinned by statistics. In such fields, pseudo-random sampling fulfils a myriad of purposes, such as Monte Carlo integration, Bayesian inference, or the generation of initial conditions for numerical simulations. The linear interpolant sampling algorithm underpinning `lintsampler` is a simple and effective alternative to existing techniques, and has no publicly available implementation at present.

# Features

Although `lintsampler` is written in pure Python, making the code highly readable, the methods make extensive use of `numpy` functionality to provide rapid sampling. After the structure spanning the domain has been constructed, sampling proceeds with computational effort scaling linearly with number of sample points. 

We provide two methods to define the domain, both optimised with `numpy` functionality for efficient construction. The `DensityGrid` class takes highly flexible inputs for defining a grid. In particular, the grid need not be evenly spaced (or even continuous) in any dimensions; the user can preferentially place grid elements near high-density regions. The `DensityTree` class takes error tolerance parameters and constructs an adaptive structure to achieve the specified tolerance. We also provide a base class (`DensityStructure`) such that the user could extend the methods for spanning the domain.

Documentation for `lintsampler`, including example notebooks demonstrating a range of problems, is available via a [readthedocs page](https://lintsampler.readthedocs.io). The documentation also has an extensive explanation of the interfaces, including optimisation parameters for increasing the efficiency in sampling.

# Acknowledgements

We would like to thank Sergey Koposov for useful discussions. APN acknowledges funding support from an Early Career Fellowship from the Leverhulme Trust. MSP acknowledges funding support from a UKRI Stephen Hawking Fellowship.


# References
