# Inverse Transform Sampling

Linear interpolant sampling is based on the idea of  *inverse transform sampling*,[^invaka] which is a standard technique for drawing samples from probability distributions.

This section gives a brief description of inverse transform sampling, but fuller descriptions can be found elsewhere, such as the [Wikipedia article](https://en.wikipedia.org/wiki/Inverse_transform_sampling)  or various standard reference texts.[^devroye]

## Univariate PDF

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


## Multivariate PDF

Now, we can consider what happens in a multi-dimensional scenario, where we have a multivariate PDF $p(\mathbf{x}); \mathbf{x}=(x_0, x_1, ..., x_{k-1})$. Now, a few steps are added to the procedure. We first integrate successively over all of the dimensions to obtain a hierarchy of marginal distributions,
\begin{align}
\begin{split}
   p(x_0, x_1, ..., x_{k-2}) &= \int p(\mathbf{x}) dx_{k-1}; \\
   p(x_0, x_1, ..., x_{k-3}) &= \int p(x_0, x_1, ..., x_{k-2}) dx_{k-2};\\
   &\vdots\\
   p(x_0, x_1) &= \int p(x_0, x_1, x_2) dx_2; \\
   p(x_0) &= \int p(x_0, x_1) dx_1.
\end{split}
\end{align}

Looping back through these distributions, we can derive a cascade of conditional distributions by dividing each marginal distribution by the distribution before it,
\begin{align}
\begin{split}
   p(x_1 | x_0) &=  \frac{p(x_0, x_1)}{p(x_0)}; \\
   p(x_2 | x_0, x_1) &=  \frac{p(x_0, x_1, x_2) }{ p(x_0, x_1)}; \\
   &\vdots\\
   p(x_{k-1}|x_0, x_1, ..., x_{k-2}) &= \frac{p(\mathbf{x}) }{ p(x_0, x_1, ..., x_{k-2})}.
\end{split}
\end{align}

Then, to draw a sample $\mathbf{X}\sim p(\mathbf{x})$, we first draw a 1D sample along the first dimension $X_0 \sim p(x_0)$ using the univariate sampling procedure described above. Next, we draw a second 1D sample $X_1 \sim p(x_1 | x_0=X_0)$, then $X_2 \sim p(x_2 | x_0=X_0, x_1=X_1)$, and so on until the dimensions are exhausted.[^dimdep] This process is visualised below.

**ANIMATION GOES HERE**

As in the univariate case, when the quantile functions for all of the conditional distributions are known, this procedure is an extremely rapid way of generating multivariate samples. However, when this is not the case and the quantile function is to be obtained numerically, the computational expense of the $k$-dimensional numerical integration grows rapidly with the number of dimensions.

[^invaka]: Also known by other names, such as simply the *inverse method*.
[^devroye]: For example, the classic 1986 textbook by Devroye: *Non-Uniform Random Variate Generation*. DOI: [10.1007/978-1-4613-8643-8](https://doi.org/10.1007/978-1-4613-8643-8)
[^1DGaussian]: $p(x) \propto e^{-x^2/2}$.
[^GaussianCDF]: Formally, the CDF of the standard normal distribution is $F(x) = \frac{1}{2}\left[1 + \mathrm{erf}(\frac{x}{\sqrt{2}})\right],$ where $\mathrm{erf}$ is the [error function](https://en.wikipedia.org/wiki/Error_function).
[^probit]: The quantile function for the standard normal distribution is the [probit function](https://en.wikipedia.org/wiki/Probit).
[^uniform]: Actually, there is a second key ingredient: the ability to draw uniform (pseudo-)random samples, which we have taken as given.
[^dimdep]: The key thing to note here is that the sampling distribution along each dimension in the loop is *conditioned on the sample values obtained on the earlier dimensions in the loop*.