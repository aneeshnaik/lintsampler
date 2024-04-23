# Inverse Transform Sampling

Linear interpolant sampling is based on the idea of  *inverse transform sampling*,[^invaka] which is a standard technique for drawing samples from probability distributions.

This section and gives a brief description of inverse transform sampling, but fuller descriptions can be found elsewhere, such as the [Wikipedia article](https://en.wikipedia.org/wiki/Inverse_transform_sampling)  or various standard reference texts.[^devroye]


## One Dimension

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


## $k$ Dimensions

Now, we can consider what happens in a multi-dimensional scenario, where we have a multivariate PDF $p(\mathbf{x})$, where $\mathbf{x}=(x_0, x_1, ..., x_{k-1})$ is a vector of random variables. Compared with the 1D procedure, a few additional steps are introduced. We first integrate successively over all of the dimensions to obtain a hierarchy of marginal distributions,

$$
\begin{split}
   p(x_0, x_1, ..., x_{k-2}) &= \int p(\mathbf{x}) dx_{k-1}; \\
   p(x_0, x_1, ..., x_{k-3}) &= \int p(x_0, x_1, ..., x_{k-2}) dx_{k-2};\\
   &\vdots\\
   p(x_0, x_1) &= \int p(x_0, x_1, x_2) dx_2; \\
   p(x_0) &= \int p(x_0, x_1) dx_1.
\end{split}
$$

Looping back through these distributions, we can derive a cascade of conditional distributions by dividing each marginal distribution by the distribution before it,

$$
\begin{split}
   p(x_1 | x_0) &=  \frac{p(x_0, x_1)}{p(x_0)}; \\
   p(x_2 | x_0, x_1) &=  \frac{p(x_0, x_1, x_2) }{ p(x_0, x_1)}; \\
   &\vdots\\
   p(x_{k-1}|x_0, x_1, ..., x_{k-2}) &= \frac{p(\mathbf{x}) }{ p(x_0, x_1, ..., x_{k-2})}.
\end{split}
$$

Then, to draw a sample $\mathbf{X}\sim p(\mathbf{x})$, we first draw a 1D sample along the first dimension $X_0 \sim p(x_0)$ using the univariate sampling procedure described in the previous section. Next, we draw a second 1D sample $X_1 \sim p(x_1 | x_0=X_0)$, then $X_2 \sim p(x_2 | x_0=X_0, x_1=X_1)$, and so on until the dimensions are exhausted.[^dimdep] 

As in the univariate case, when the quantile functions for all of the conditional distributions are known, this procedure is an extremely rapid way of generating multivariate samples. However, when this is not the case and the quantile function is to be obtained numerically, the computational expense of the $k$-dimensional numerical integration grows rapidly with the number of dimensions.


## 2D Example: Setup

We'll illustrate this process with an example. For simplicity of visualisation we'll take a 2D example, but the process works in any number of dimensions.

For our 2D PDF over $x$ and $y$, we'll take a 2-dimensional Gaussian with mean $\mu \equiv (\mu_x, \mu_y) = (1.5, -0.5)$ and a covariance matrix given by

$$
   \Sigma \equiv
      \begin{pmatrix}
      \Sigma_{xx} & \Sigma_{xy}\\
      \Sigma_{yx} & \Sigma_{yy}
      \end{pmatrix} =
      \begin{pmatrix}
      1.0 & 1.2\\
      1.2 & 1.8
      \end{pmatrix}.
$$

We can visualise this PDF with a contour plot:
```{figure} ../assets/2D_pdf.png
:align: center
```
Walking outwards from the centre, the four contour lines here are the 1-, 2-, 3-, and 4-$\sigma$ contours.


## 2D Example: $p(x)$ and $p(y|x)$

According to the procedure outlined at the beginning of this section, we're going to need to derive two one-dimensional PDFs from this 2D PDF: $p(x)$ and $p(y|x)$.[^axswap] For $p(x)$, we can use the useful fact that marginal distributions of the multivariate Gaussian are also themselves Gaussian. In other words, $p(x)$ is a 1D Gaussian (normal) distribution with mean $\mu_x$ and standard deviation $\Sigma_{xx}$.[^marginalparams] This is illustrated in the plot below, where the lower panel shows the original 2D PDF $p(x, y)$ while the upper panel shows the marginal PDF $p(x)$.

```{figure} ../assets/2D_marginal.png
:align: center
```

Next, we need to derive the conditional PDF $p(y | x)$. Like marginal distributions, it turns out that conditional distributions of the multivariate Gaussian are also Gaussian,[^projections] with mean and variance given by[^matrixeqs]

$$
\begin{split}
   \mu_{y|x} &=  \mu_y + \Sigma_{yx}\Sigma_{xx}^{-1}(x - \mu_x); \\
   \Sigma_{y|x} &=  \Sigma_{yy} - \Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}. \\
\end{split}
$$

As one might expect, unlike the marginal distribution $p(x)$, the conditional distribution $p(y|x)$ has a functional dependence on $x$ as well as $y$. To visualise it, we can take a single $x$ value and see the distribution over $y$. In particular, in the figure below we'll take $x=3.5$, so the mean $\mu_{y|x}=1.9$ and the variance $\Sigma_{y|x}=0.36$. The left-hand panel shows the original 2D PDF $p(x, y)$ while the right-hand panel shows $p(y | x=3.5)$. 

```{figure} ../assets/2D_conditional.png
:align: center
```

## 2D Example: Sampling

We now have everything we need to draw samples. The whole procedure is illustrated on this (somewhat busy) figure:

```{figure} ../assets/2D_sampling.png
:align: center
```

There are five steps annotated on the figure. These are:
1. Draw a uniform sample $U_1$.
2. Given the marginal distribution $p(x)$, apply the quantile function $F^{-1}(U_1)$, to give a sample $X\sim p(x)$.[^step2]
3. Given the sample $X$, calculate the parameters of the conditional PDF $p(y | x=X)$.
4. Draw a second uniform sample $U_2$.
5. Again using the 1D procedure, convert $U_2$ into a sample $Y\sim p(y|x=X)$.

Then, $X$ and $Y$ together form a joint sample from $p(x, y)$. In the figure above, the sample we get by carrying out the procedure once is indicated by the golden circle. 

To get multiple samples, we can simply repeat these steps multiple times. This is shown in the animation below.

```{figure} ../assets/2D_animation.gif
:align: center
```

[^invaka]: Also known by other names, such as simply the *inverse method*.
[^devroye]: For example, the classic 1986 textbook by Devroye: *Non-Uniform Random Variate Generation*. DOI: [10.1007/978-1-4613-8643-8](https://doi.org/10.1007/978-1-4613-8643-8)
[^1DGaussian]: $p(x) \propto e^{-x^2/2}$.
[^GaussianCDF]: Formally, the CDF of the standard normal distribution is $F(x) = \frac{1}{2}\left[1 + \mathrm{erf}(\frac{x}{\sqrt{2}})\right],$ where $\mathrm{erf}$ is the [error function](https://en.wikipedia.org/wiki/Error_function).
[^probit]: The quantile function for the standard normal distribution is the [probit function](https://en.wikipedia.org/wiki/Probit).
[^uniform]: Actually, there is a second key ingredient: the ability to draw uniform (pseudo-)random samples, which we have taken as given.
[^dimdep]: The key thing to note here is that the sampling distribution along each dimension in the loop is *conditioned on the sample values obtained on the earlier dimensions in the loop*.
[^axswap]: We could equally do this the other way round, i.e., derive $p(y)$ and $p(x | y)$.
[^marginalparams]: In this case, $\mu_x=1.5$ and $\Sigma_{xx}=1.0$.
[^projections]: These facts regarding marginal and conditional distributions can be understood as saying that both *projections* and *slices* of a multivariate Gaussian give Gaussian images.
[^matrixeqs]: These equations hold in general for an $m$-dimensional conditional slice of an $n$-dimensional Gaussian distribution. In such a case, $\mu_x, \mu_y$ are vectors (subvectors of the parent Gaussian's mean vector) and the various $\Sigma_{(...)}$ objects are matrices (submatrices of the parent Gaussian's covariance matrix). Here, however, these are all simply scalars.
[^step2]: This is precisely the 1D inverse transform sampling procedure described in the previous section. As we saw in the example there, because the marginal PDF $p(x)$ in this case is a Gaussian, the quantile function is the probit function.

