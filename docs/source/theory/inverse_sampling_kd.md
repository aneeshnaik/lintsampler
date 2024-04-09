# Inverse Transform Sampling: kD

The previous section described how inverse transform sampling works in one dimension. Now, we can consider what happens in a multi-dimensional scenario, where we have a multivariate PDF $p(\mathbf{x})$, where $\mathbf{x}=(x_0, x_1, ..., x_{k-1})$ is a vector of random variables.

## Mathematical Description

Compared with the 1D procedure, a few additional steps are introduced. We first integrate successively over all of the dimensions to obtain a hierarchy of marginal distributions,
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

Then, to draw a sample $\mathbf{X}\sim p(\mathbf{x})$, we first draw a 1D sample along the first dimension $X_0 \sim p(x_0)$ using the univariate sampling procedure described in the previous section. Next, we draw a second 1D sample $X_1 \sim p(x_1 | x_0=X_0)$, then $X_2 \sim p(x_2 | x_0=X_0, x_1=X_1)$, and so on until the dimensions are exhausted.[^dimdep] 


## Example: Setup

We'll illustrate this process with an example. For simplicity of visualisation we'll take a 2D example, but the process works in any number of dimensions.

For our 2D PDF over $x$ and $y$, we'll take a 2-dimensional Gaussian with mean $\mu \equiv (\mu_x, \mu_y) = (1.5, -0.5)$ and a covariance matrix given by
\begin{equation}
   \Sigma \equiv
      \begin{pmatrix}
      \Sigma_{xx} & \Sigma_{xy}\\
      \Sigma_{yx} & \Sigma_{yy}
      \end{pmatrix} =
      \begin{pmatrix}
      1.0 & 1.2\\
      1.2 & 1.8
      \end{pmatrix}.
\end{equation}
We can visualise this PDF with a contour plot:
```{figure} ../assets/2D_pdf.png
:align: center
```
Walking outwards from the centre, the four contour lines here are the 1-, 2-, 3-, and 4-$\sigma$ contours.


## Example: $p(x)$ and $p(y|x)$

According to the procedure outlined at the beginning of this section, we're going to need to derive two one-dimensional PDFs from this 2D PDF: $p(x)$ and $p(y|x)$.[^axswap] For $p(x)$, we can use the useful fact that marginal distributions of the multivariate Gaussian are also themselves Gaussian. In other words, $p(x)$ is a 1D Gaussian (normal) distribution with mean $\mu_x$ and standard deviation $\Sigma_{xx}$.[^marginalparams] This is illustrated in the plot below, where the lower panel shows the original 2D PDF $p(x, y)$ while the upper panel shows the marginal PDF $p(x)$.

```{figure} ../assets/2D_marginal.png
:align: center
```

Next, we need to derive the conditional PDF $p(y | x)$. Like marginal distributions, it turns out that conditional distributions of the multivariate Gaussian are also Gaussian,[^projections] with mean and variance given by[^matrixeqs]
\begin{align}
\begin{split}
   \mu_{y|x} &=  \mu_y + \Sigma_{yx}\Sigma_{xx}^{-1}(x - \mu_x); \\
   \Sigma_{y|x} &=  \Sigma_{yy} - \Sigma_{yx}\Sigma_{xx}^{-1}\Sigma_{xy}. \\
\end{split}
\end{align}
As one might expect, unlike the marginal distribution $p(x)$, the conditional distribution $p(y|x)$ has a functional dependence on $x$ as well as $y$. To visualise it, we can take a single $x$ value and see the distribution over $y$. In particular, in the figure below we'll take $x=3.5$, so the mean $\mu_{y|x}=1.9$ and the variance $\Sigma_{y|x}=0.36$. The left-hand panel shows the original 2D PDF $p(x, y)$ while the right-hand panel shows $p(y | x=3.5)$. 

```{figure} ../assets/2D_conditional.png
:align: center
```

## Example: Sampling

We now have everything we need to draw samples. The procedure runs as follows:

1. Draw a uniform sample $U_1$.
2. Use the 1D inverse transform sampling procedure to convert $U_1$ into a sample $X~p(x)$. Because the marginal PDF $p(x)$ is a Gaussian, this can be done analytically,[^probit] as we saw in the example accompanying the description of 1D inverse transform sampling.
3. Given the sample $X$, calculate the parameters of the conditional PDF $p(y | x=X)$.
4. Draw a second uniform sample $U_2$.
5. Again using the 1D procedure, convert $U_2$ into a sample $Y~p(y|x=X)$.

Then, $X$ and $Y$ together form a joint sample from $p(x, y)$.

This procedure is animated below.

```{figure} ../assets/2D_animation.gif
:align: center
```

As in the univariate case, when the quantile functions for all of the conditional distributions are known, this procedure is an extremely rapid way of generating multivariate samples. However, when this is not the case and the quantile function is to be obtained numerically, the computational expense of the $k$-dimensional numerical integration grows rapidly with the number of dimensions.

[^dimdep]: The key thing to note here is that the sampling distribution along each dimension in the loop is *conditioned on the sample values obtained on the earlier dimensions in the loop*.
[^axswap]: We could equally do this the other way round, i.e., derive $p(y)$ and $p(x | y)$.
[^marginalparams]: In this case, $\mu_x=1.5$ and $\Sigma_{xx}=1.0$.
[^projections]: These facts regarding marginal and conditional distributions can be understood as saying that both *projections* and *slices* of a multivariate Gaussian give Gaussian images.
[^matrixeqs]: These equations hold in general for an $m$-dimensional conditional slice of an $n$-dimensional Gaussian distribution. In such a case, $\mu_x, \mu_y$ are vectors (subvectors of the parent Gaussian's mean vector) and the various $\Sigma_{(...)}$ objects are matrices (submatrices of the parent Gaussian's covariance matrix). Here, however, these are all simply scalars.
[^probit]: $X = \mathrm{probit}(U_1)$