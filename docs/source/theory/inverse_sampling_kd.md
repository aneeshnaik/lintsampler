# Inverse Transform Sampling: kD

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

[^dimdep]: The key thing to note here is that the sampling distribution along each dimension in the loop is *conditioned on the sample values obtained on the earlier dimensions in the loop*.