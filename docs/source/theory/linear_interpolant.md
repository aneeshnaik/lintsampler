# Linear Interpolant Sampling

This page describes the linear interpolant sampling algorithm underpinning `lintsampler`. The [first section](#one-dimensional-unit-interval) outlines how to draw a sample on the 1D unit interval, i.e., when the density $f(x)$ is known at $x=0$ and $x=1$ only. The [second section](#k-dimensional-unit-hypercube) then generalises this to the unit hypercube over an arbitrary number of dimensions. Next, the [third section](#arbitrary-hyperbox) describes the procedure for drawing a sample in an arbitrary hyperbox, by applying a linear transformation to the unit hypercube sample. The [fourth section](#multiple-hyperboxes) discusses applying the technique in a context where there many such hyperboxes, e.g., when densities are known only on a grid.

At the end of the page is a [summary section](#summary) which summarises the whole procedure. All of the key equations and steps are restated there, so that it can serve as a self-contained description of the technique. Thus, readers seeking a quick reference can skip the preceding sections and head straight there.

The algebra on this page gets a little dense, and mostly concerns abstract probability distributions. The [next page](./worked_example) gives a worked example, which should make it all a little more concrete.


## One-dimensional unit interval

First, the problem of drawing a sample on the one-dimensional unit interval, where the (unnormalised) density is known only at the interval ends. In other words, the density function of interest is $f(x)$, with $x\in [0, 1]$ and densities known at the vertices: $f(0) = f_0$ and $f(1)=f_1$. 

The key step of linear interpolant sampling is to assume that between the vertices, we can approximate the true density function $f$ with the linear interpolant $\hat{f}$. In one dimension this is

$$
    \hat{f}(x) = f_0 (1 - x) + f_1 x.
$$ (1DInterpolant)

The figure below demonstrates this with an example. The 'true' density $f(x)$ here is shown by the light grey line. At $x=0$ and $x=1$ it takes the values 6 and 3 respectively, so that the linear interpolant (shown by the teal line) is $\hat{f}(x) = 6 - 3x$.

```{figure} ../assets/lint1Dunit_pdf.png
:align: center
```

To draw samples from $\hat{f}(x)$, we must first normalise it to a true PDF, leading to

$$
    p(x) = \frac{f_0 (1 - x) + f_1 x}{\tilde{f}},
$$ (1DInterpolantNormalised)

where $\tilde{f} = (f_0 + f_1) / 2$ is the average vertex density.

Now, the goal is to draw samples from this PDF. We can do this with the inverse transform sampling procedure described on the [previous page](./inverse_sampling). For this, we require an expression for the quantile distribution. The PDF is linear, so its corresponding CDF is quadratic,

$$
    F(x) = \left(\frac{f_1-f_0}{f_1 + f_0}\right)x^2 + \frac{2f_0}{f_0 + f_1}x.
$$ (1DCDF)

The CDF is then readily inverted by completing the square to obtain the quantile function,

$$
    Q(z) = \begin{cases}
    z, &f_0 = f_1;\\
    \dfrac{-f_0 + \sqrt{f_0^2 + (f_1^2 - f_0^2)z}}{f_1 - f_0}, &f_0 \neq f_1.
    \end{cases}
$$ (1DQuantile)

Thus, to draw a sample $X\sim p(x)$, one need only draw a uniform sample $U$, then pass it to the quantile function above to obtain a sample. In other words:

$$
    X = \begin{cases}
    U, &f_0 = f_1;\\
    \dfrac{-f_0 + \sqrt{f_0^2 + (f_1^2 - f_0^2)U}}{f_1 - f_0}, &f_0 \neq f_1.
    \end{cases}
$$ (1DSample)

In the special case that $f_0 = f_1$, the linear interpolant between $x=0$ and $1$ becomes a flat line, so sampling on that interval is equivalent to sampling uniformly ($X=U$).

The animation below demonstrates this in action, taking the same toy problem as in the figure above. An extra panel has been added to the figure showing the CDF.[^1DCDF]. The animation shows 1000 samples being generated one at a time: uniform samples are generated along the vertical axis of the top panel, then these are transformed into samples $X$ via the quantile function.[^1DQ] As more samples are increasingly generated, the distribution of samples shown in the lower panel increasingly comes to resemble the interpolant $\hat{f}$.

```{figure} ../assets/lint1Dunit_animation.gif
:align: center
```


## k-dimensional unit hypercube

Let's generalise the procedure from the previous section from the one-dimensional unit interval to the $k$-dimensional unit hypercube. The 'first' corner of the cube is at the coordinate origin, and the various sidelengths are unity. As in the 1D case, the density is known at the $2^k$ corners of the cube. We write $f_{i_0, i_1, ..., i_{k-1}}$ to indicate the density at the corner indexed $(i_0, i_1, ..., i_{k-1})$, $i_j \in \{0, 1\}$, so that e.g., $f_{0...0}$ is the density at the origin.

As in the 1D case, the key step here is to assume that inside the hypercube, we can approximate the density with the linear interpolant. Now, instead of the 1D linear interpolant, we have the multilinear interpolant:

$$
    \hat{f}(\mathbf{x}) = \sum_{i_0, i_1, ..., i_{k-1} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}),
$$ (kDInterpolant)

where

$$
    \Delta^{(0, 1)}_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}) \equiv \prod_{d=0}^{k-1}  \left\{\!\begin{aligned}
        &1-x_d &\text{ if } i_d = 0\\
        &x_d &\text{ if } i_d = 1
    \end{aligned}\right\}.
$$ (Delta01)

In other words, the value of the interpolant at a given interior point is a weighted sum over the corner densities, with each corner weighted by a geometrical factor $\Delta$ encoding its proximity to the interior point. The superscipt $(0, 1)$ accompanying $\Delta$ is to indicate that it is defined over the unit hypercube; a more general version of $\Delta$ will be used later.

In one dimension, this $k$-linear interpolant reduces to the 1D interpolant described earlier {eq}`1DInterpolant`. Similarly, in two dimensions $(x, y)$, it reduces to the [bilinear interpolant](https://en.wikipedia.org/wiki/Bilinear_interpolation)

$$
    \hat{f}(x, y) = f_{00}(1-x)(1-y) + f_{01}(1-x)y + f_{10}x(1-y) + f_{11}xy.
$$ (2DInterpolant)

The figure below illustrates this bilinear interpolant.
```{figure} ../assets/lintkDunit_interpolant.png
:align: center
```
The left-hand panel shows an example of a density field, $f(x, y) = 10 - 2x^2 - 2y^2$, on the 2D unit interval. The corner values are then $f_{00}=10$, $f_{01}=f_{10}=8$, and $f_{11}=6$. The right-hand panel then plots the bilinear interpolant. Visually similar, but obviously different.

As in the 1D case, we can renormalise the multilinear interpolant in order to treat it as a true probability density,

$$
    p(\mathbf{x}) = \frac{1}{\tilde{f}} \times  \sum_{i_0, i_1, ..., i_{k-1} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}} \Delta_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}),
$$ (kDInterpolantNormalised)

where $\tilde{f}$ is the mean corner density.[^faverage]

The goal now is to draw samples from this $k$-dimensional PDF. Once again, we can achieve this with inverse transform sampling. As we discussed on the [previous page](./inverse_sampling), to perform inverse transform sampling with a multivariate PDF, one needs to derive a sequence of one-dimensional PDFs $p(x_0)$, $p(x_1 | x_0)$, $p(x_2 | x_0, x_1)$, etc.

Integrating over all dimensions except the first, we obtain

$$
     p(x_0) = \frac{\tilde{f}_0 (1 - x_0) + \tilde{f}_1 x_0}{\tilde{f}},
$$ (1DMarginal)

where $\tilde{f}_0$ is the mean corner density of the $2^{k-1}$ corners at $x_0 = 0$,

$$
    \tilde{f}_0 \equiv \frac{1}{2^{k-1}} \times \sum_{i_1, ..., i_{k-1} \in \{0,1\}} f_{0, i_1, ..., i_{k-1}}.
$$ (f0)

Similarly, $\tilde{f}_1$ is the average density over the corners at $x_0 = 1$. In other words, the 1D marginal probability density function takes precisely the same expression as the probability density function we derived for the 1D linear interpolant above {eq}`1DInterpolantNormalised`, but with the endpoint densities $f_0, f_1$ replaced with averages $\tilde{f}_0, \tilde{f}_1$ taken over the two end facets of the hypercube along the relevant dimension. So, one can draw a sample $X_0 \sim p(x_0)$ via Eq. {eq}`1DSample`, replacing $f_0,f_1$ with $\tilde{f}_0, \tilde{f}_1$.

Proceeding to the second dimension ($x_1$), the conditional distribution $p(x_1 | x_0)$ is given by

$$
    p(x_1 | x_0) = \frac{\tilde{f}_{00}(1-x_0)(1-x_1) + \tilde{f}_{01}(1-x_0)x_1 + \tilde{f}_{10}x_0(1-x_1) + \tilde{f}_{11}x_0x_1}{\tilde{f}_0 (1 - x_0) + \tilde{f}_1 x_0},
$$ (2DConditionalUnarranged)

where, extending the notation above, $\tilde{f}_{ij}$ is the mean corner density of the $2^{k-2}$ corners at $x_0=i, x_1=j$. This distribution can be written in a more useful form

$$
    p(x_1 | x_0) = \frac{g^{(1)}_0 (1 - x_1) + g^{(1)}_1 x_1}{\tilde{g}^{(1)}},
$$ (2DConditional)

where the functional dependence on $x_0$ has been completely absorbed into the two newly defined densities $g^{(1)}_0$ and $g^{(1)}_1$ (and their average $\tilde{g}^{(1)} \equiv (g^{(1)}_0 + g^{(1)}_1) / 2$). These are

$$
    g^{(1)}_0(x_0) = \tilde{f}_{00}(1-x_0) + \tilde{f}_{10}x_0,
$$ (2Dg0)

$$
    g^{(1)}_1(x_0) = \tilde{f}_{01}(1-x_0) + \tilde{f}_{11}x_0.
$$ (2Dg1)

These can be readily understood as the linearly interpolated densities at the positions $(x_0, 0)$ and $(x_0, 1)$ (averaging away the higher dimensions). 

Once again, the distribution $p(x_1 | x_0)$ takes the form of a 1D linear interpolant, so we can draw a sample $X_1 \sim p(x_1|x_0)$ using Eq. {eq}`1DSample`, replacing $f_0,f_1$ with $g^{(1)}_0(x_0), g^{(1)}_1(x_0)$, *evaluating the latter densities at $x_0=X_0$*. Then, the pair $(X_0, X_1)$ represent a sample from the joint distribution $p(x_0, x_1)$.

This pattern continues into higher dimensions. For example, the conditional distribution along the third dimension $p(x_2 | x_0, x_1)$ takes the same form as $p(x_1 | x_0)$, but now with densities $g^{(2)}_0(x_0, x_1), g^{(2)}_1(x_0, x_1)$, which depend on $x_0$ and $x_1$ rather than just $x_0$. Then, a sample $X_2 \sim p(x_2 | x_0, x_1)$ is drawn using Eq. {eq}`1DSample`, replacing $f_0,f_1$ with $g^{(2)}_0, g^{(2)}_1$ evaluated at $(X_0, X_1)$. 

In general, along the $(j+1)$<sup>th</sup> dimension, the conditional distribution over $x_j$ is

$$
    p(x_j | x_0, x_1, ..., x_{j-1}) = \frac{g^{(j)}_0 (1 - x_j) + g^{(j)}_1 x_j}{\tilde{g}^{(j)}},
$$ (kDConditional)

where $g^{(j)}_0$ and $g^{(j)}_1$ are given by

$$
    g^{(j)}_0 = 
    \begin{cases}
        \tilde{f}_0, &j = 0;\\
        \sum_{i_0, i_1, ..., i_{j-1} \in \{0,1\}} \tilde{f}_{i_0, i_1, ..., i_{j-1}, 0} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{j-1}}(x_0, x_1, ..., x_{j-1}), &\text{otherwise}.
    \end{cases}
$$ (g0)

$$
    g^{(j)}_1 =
    \begin{cases}
        \tilde{f}_1, &j = 0;\\
        \sum_{i_0, i_1, ..., i_{j-1} \in \{0,1\}} \tilde{f}_{i_0, i_1, ..., i_{j-1}, 1} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{j-1}}(x_0, x_1, ..., x_{j-1}), &\text{otherwise}.
    \end{cases}
$$ (g1)

So, $g^{(j)}_0$ and $g^{(j)}_1$ are aggregate densities which depend on the sampling points chosen along the previous dimensions.

In summary, the procedure for drawing a sample from the $k$-linear interpolant density is to loop over the dimensions; at each dimension, evaluate $g_0$ and $g_1$ using the sample values from the dimensions before the current dimension, then sample a new point along the current dimension using Eq. {eq}`1DSample`, replacing $f_0,f_1$ with $g^{(j)}_0, g^{(j)}_1$.



## Arbitrary hyperbox

The sampling procedure outlined in the previous section concerned a unit hypercube with first vertex at the origin. However, it is easily generalised to arbitrary hyperboxes at arbitrary positions. Given a $k$-dimensional box with first vertex at $\mathbf{a}=(a_0, a_1, ..., a_{k-1})$ and final vertex at $\mathbf{b}=(b_0, b_1, ..., b_{k-1})$ (so that the box spans $[a_j, b_j]$ along dimension $j$), the k-linear interpolant within the box is given by

$$
    \hat{f}(\mathbf{x}) = \frac{1}{V} \times \sum_{i_0, i_1, ..., i_{k-1} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}} \Delta^{(\mathbf{a},\mathbf{b})}_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}),
$$ (kDArbitraryInterpolant)

which can be renormalised into a true probability density

$$
    p(\mathbf{x}) = \frac{1}{\tilde{f}V^2} \times  \sum_{i_0, i_1, ..., i_{k-1} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}} \Delta^{(\mathbf{a},\mathbf{b})}_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}),
$$ (kDArbitraryInterpolantNormalised)

where $V=\prod_i (b_i - a_i)$ is the volume of the box and the superscript $(\mathbf{a},\mathbf{b})$ accompanying the geometric factor $\Delta$ indicates that it now takes a slightly different definition compared with $\Delta^{(0,1)}$ before,

$$
    \Delta^{(\mathbf{a},\mathbf{b})}_{i_0, i_1, ..., i_{k-1}}(\mathbf{x}) \equiv \prod_{d=0}^{k-1}  \left\{\!\begin{aligned}
        &b_d-x_d &\text{ if } i_d = 0\\
        &x_d-a_d &\text{ if } i_d = 1
    \end{aligned}\right\}.
$$ (DeltaAB)

One can transform this box into a unit hypercube by defining a new coordinate system $\mathbf{z}$, related linearly to $\mathbf{x}$ via

$$
    z_i = \frac{x_i- a_i}{b_i - a_i}.
$$ (LinearTransform)

Thus, the point $\mathbf{a}$ in the original coordinate system corresponds to the origin in the new coordinate system, while the point $\mathbf{b}$ corresponds to $\mathbf{z} = (1, 1, ..., 1)$.

The probability distribution over $\mathbf{x}$ transforms as

$$
    p({\mathbf{z}}) = \frac{p({\mathbf{x}})}{|\mathcal{J}|},
$$ (PDFTransformJacobian)

where $|\mathcal{J}|$ is the determinant of the Jacobian matrix of the coordinate transformation. Because the transformation is diagonal, this determinant straightforwardly evaluates to $1/V$, while the geometric factor $\Delta^{(\mathbf{a},\mathbf{b})}(\mathbf{x}) = V \Delta^{(0, 1)}(\mathbf{z})$, so

$$
    p({\mathbf{z}}) = \frac{1}{\tilde{f}} \times  \sum_{i_0, i_1, ..., i_{k-1} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{k-1}}(\mathbf{z}).
$$ (PDFTransformExplicit)

This is identical to the probability distribution we drew samples from in the case of the unit hypercube. So, given a $k$-dimensional box with first and last vertices at $\mathbf{a}$ and $\mathbf{b}$, one can simply draw samples $\mathbf{Z}$ within the unit hypercube as demonstrated in the previous section, then transform them to the desired box according to $X_i=a_i + (b_i - a_i)Z_i$.

This is demonstrated in two dimensions in the figure below. 
```{figure} ../assets/lint_transformation.png
:align: center
```
In the left-hand panel we start off with a 2D rectilinear cell in ($x_0, x_1$): a rectangle bounded by $x_0=40,50$ and $x_1=180,200$. Within this rectangle is a (bilinearly) interpolated density function. The centre panel then shows this density transformed to the unit square. Within the unit square, we take a sample (shown by the blue point), then we can transform it back to the space of the original rectangle (right-hand panel).


## Multiple hyperboxes

We now have a procedure for drawing a random sample within a single $k$-dimensional hyperbox. It is straightforward to incorporate this procedure into an algorithm for drawing a single sample across a series of such hyperboxes. An example of such a scenario is a grid where densities are known only at the intersections of the gridlines.

For a given grid cell $C$, the unnormalised density function $f$ integrated over the cell volume $V_C$ gives a `mass' for the cell. As we only know the values of $f$ at the cell vertices, we can estimate this mass as

$$
    m_C = V_C \tilde{f}_C,
$$ (CellMass)

i.e., the cell volume multiplied by the mean of the densities at the $2^k$ vertices adjoining $C$. This estimate for the mass is consistent with the approximation that the density function in the interior of the cell is given by the $k$-linear interpolant, the integral of which over $V_C$ gives $V_C \tilde{f}_C$.

Having computed the masses of all cells according to this formula, the probability of cell $C$ can be computed by normalising,

$$
    p_C = \frac{m_C}{\sum_i m_i}.
$$ (CellProbability)

Having thus assigned a probability to each cell, one can then randomly choose a cell from the probability-weighted list of cells, then draw a sample within the chosen cell using the procedure described in the previous subsections.


## Summary

This section summarises the algorithm which has been described over the course of this page. The key steps and equations are all repeated here so that this section can be used as a self-contained reference. 

We start with a $k$-dimensional rectilinear cell. The $2^k$ vertices of this cell are indexed $i_0, i_1, ..., i_{k-1} \in \{0,1\}$, and the corresponding (known) densities at these vertices are labelled  $f_{i_0, i_1, ..., i_{k-1}}$. The spatial coordinates of the first vertex (i.e., the vertex indexed $i_0, i_1, ..., i_{k-1} = 0$) are encoded in the vector $\mathbf{a}$ and those of the last vertex ($i_0, i_1, ..., i_{k-1} = 1$) are $\mathbf{b}$.

To draw a single random sample from this hyperbox:
1. Sample a point $\mathbf{Z}$ within the unit $k$-dimensional hypercube. To do this, loop over the $k$ dimensions. At each dimension $j$:

    + Calculate aggregate densities $g^{(j)}_0$ and $g^{(j)}_1$. These are given by

      $$
        g^{(j)}_0 = 
        \begin{cases}
            \tilde{f}_0, &j = 0;\\
            \sum_{i_0, i_1, ..., i_{j-1} \in \{0,1\}} \tilde{f}_{i_0, i_1, ..., i_{j-1}, 0} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{j-1}}(Z_0, Z_1, ..., Z_{j-1}), &\text{otherwise}.
        \end{cases}
      $$ (g0Summary)

      $$
        g^{(j)}_1 = 
        \begin{cases}
            \tilde{f}_1, &j = 0;\\
            \sum_{i_0, i_1, ..., i_{j-1} \in \{0,1\}} \tilde{f}_{i_0, i_1, ..., i_{j-1}, 1} \Delta^{(0, 1)}_{i_0, i_1, ..., i_{j-1}}(Z_0, Z_1, ..., Z_{j-1}), &\text{otherwise}.
        \end{cases}
      $$ (g1Summary)

      In the notation here, $\tilde{f}_{...}$ indicates that all vertex densities in the dimensions higher than the indexed dimensions are averaged over. For example, in 3 dimensions, $\tilde{f}_{01} = (f_{010} + f_{011}) / 2$ and $\tilde{f}_{1} = (f_{100} + f_{101} + f_{110} + f_{111}) / 4$. For all dimensions except the first ($j=0$), the aggregate densities $g^{(j)}_0$ and $g^{(j)}_1$ depend on the sampling points drawn at the previous dimensions, $Z_0$ to $Z_{j-1}$. These appear in the geometrical factor $\Delta^{(0, 1)}$, which is given by
    
      $$
        \Delta^{(0, 1)}_{i_0, i_1, ..., i_{j-1}}(Z_0, Z_1, ..., Z_{j-1}) \equiv \prod_{d=0}^{j-1}  \left\{\!\begin{aligned}
            &1-Z_d &\text{ if } i_d = 0\\
            &Z_d &\text{ if } i_d = 1
        \end{aligned}\right\}.
      $$ (Delta01Summary)
    
    + Sample $Z_j$ according to
      
      $$
        Z_j = \begin{cases}
        U, &g_0 = g_1;\\
        \dfrac{-g_0 + \sqrt{g_0^2 + (g_1^2 - g_0^2)U}}{g_1 - g_0}, &g_0 \neq g_1,
        \end{cases}
      $$ (ZSampleSummary)

      where $U$ represents a sample from the uniform distribution on the unit interval, and the superscript has been omitted from $g_0^{(j)}$ and $g_1^{(j)}$ for brevity.

2. Having obtained a complete joint sample $\mathbf{Z} = (Z_0, Z_1, ..., Z_{k-1})$ within the unit hypercube, transform to a sample $\mathbf{X}$ in the actual coordinates of the cell via the element-wise linear transformation $X_i=a_i + (b_i - a_i)Z_i$.

These steps can be repeated $N$ times to draw $N$ independent samples within the cell.

The procedure above can also be incorporated into a wider algorithm for drawing samples in the scenario where one has a volume (or multiple disconnected volumes) comprising multiple such cells, for example when densities are known at the intersections of $k$-dimensional rectilinear grid. To draw a single sample within the volume:

1. Calculate the probabilities of all cells in the grid. The probability of cell $\alpha$ is given by
   
   $$
   p_\alpha = \frac{V_\alpha \tilde{f}_\alpha}{\sum_\beta V_\beta \tilde{f}_\beta},
   $$ (CellProbabilitySummary)

   where $V_\alpha$ is the geometric volume of the cell and $\tilde{f}_\alpha$ is the mean of the densities at the $2^k$ vertices of the cell.
2. Weighting each cell by its probability as calculated in the previous step, randomly choose a cell.
3. Within the chosen cell, draw a sample according to the single-cell procedure described above.


[^1DCDF]: Because $f_0=6$ and $f_1=3$ in this example, $F(x) = (4x - x^2) / 3$.
[^1DQ]: $X=2-\sqrt{4 - 3U}$
[^faverage]: $\tilde{f} \equiv \frac{1}{2^k} \sum_{\{i\} \in \{0,1\}} f_{i_0, i_1, ..., i_{k-1}}$