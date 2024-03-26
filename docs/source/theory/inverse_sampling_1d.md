# Inverse Transform Sampling: 1D

Linear interpolant sampling is based on the idea of  *inverse transform sampling*,[^invaka] which is a standard technique for drawing samples from probability distributions.

This section and the next give a brief description of inverse transform sampling, but fuller descriptions can be found elsewhere, such as the [Wikipedia article](https://en.wikipedia.org/wiki/Inverse_transform_sampling)  or various standard reference texts.[^devroye]

Here's how the process works in 1D. Given a univariate probability density function (PDF) $p(x)$, to draw a sample $X\sim p(x)$:
1. Integrate $p$ to obtain the cumulative density function (CDF), $F(x) = \int_{-\infty}^x p(x') dx'$.
2. Invert the CDF to give the quantile function, $Q(z) = F^{-1}(z)$.
3. Draw a random sample $U$ from the uniform distribution on the unit interval.
4. Apply the quantile function to the uniform sample to obtain $X = Q(U)$. The point $X$ is a random sample from $p(x)$.

Let's visualise how this process works. First, let's take a simple example of a univariate PDF. We'll use the standard normal (or *Gaussian*) distribution.[^1DGaussian] Here is a plot of the PDF:
```{figure} ../assets/1D_pdf.png
:align: center
```
Of course, the famous bell curve.

We can integrate the PDF to get the CDF:[^GaussianCDF]
```{figure} ../assets/1D_cdf.png
:align: center
```
As expected, this approaches 0 at large, negative $x$ and approaches 1 at large, positive $x$.

The idea now is to invert the CDF into its quantile function $Q$,[^probit] then apply $Q$ to a series of uniform samples. This process is shown in the animation below. Here, we draw a series of uniform samples, place them on the vertical axis of the CDF plot, trace them across to the point where they meet the CDF curve, then take the horizontal coordinate of this point as the desired sample. After doing this a number of times, the bell curve shape clearly begins to emerge.
```{figure} ../assets/1D_animation.gif
:align: center
```

To summarise, in this 1D scenario, the key ingredient for generating random samples from a given distribution is the quantile function (i.e., inverse CDF) of the distribution.[^uniform] If we are in a situation where we know the quantile function in closed form, inverse transform sampling is the method of choice, as a large number of samples can be drawn almost instantaneously. Even when an analytic expression for the quantile function is not available, this technique can still prove highly efficient, e.g., if one pre-computes a lookup table for the quantile function via numerical integration, then interpolates within the table to get $Q(U)$.

The next section describes how the process works in more than one dimension (i.e., with a multivariate probability distribution), which requires adding a few steps to the procedure.

[^invaka]: Also known by other names, such as simply the *inverse method*.
[^devroye]: For example, the classic 1986 textbook by Devroye: *Non-Uniform Random Variate Generation*. DOI: [10.1007/978-1-4613-8643-8](https://doi.org/10.1007/978-1-4613-8643-8)
[^1DGaussian]: $p(x) \propto e^{-x^2/2}$.
[^GaussianCDF]: Formally, the CDF of the standard normal distribution is $F(x) = \frac{1}{2}\left[1 + \mathrm{erf}(\frac{x}{\sqrt{2}})\right],$ where $\mathrm{erf}$ is the [error function](https://en.wikipedia.org/wiki/Error_function).
[^probit]: The quantile function for the standard normal distribution is the [probit function](https://en.wikipedia.org/wiki/Probit).
[^uniform]: Actually, there is a second key ingredient: the ability to draw uniform (pseudo-)random samples, which we have taken as given.