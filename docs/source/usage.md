# How to Use `lintsampler`

`lintsampler` has one public sampling class: `LintSampler`.

At the simplest, the constructor takes a probability density function (pdf) input, along with a set of cells in which to sample. Once the instance has been built, the `.sample()` function can be used to draw a number of samples from the pdf.

For example, to draw six samples within a single 3D cell, with first and last corners at $(x, y, z) = (10, 100, 1000)$ and $(20, 200, 2000)$ respectively:

```python
>>> import numpy as np
>>> from lintsampler import LintSampler
>>> x = np.linspace(10,20,2)
>>> y = np.linspace(100,200,2)
>>> z = np.linspace(1000,2000,2)
>>> def rndmpdf(X): return np.random.uniform()
>>> LintSampler((x,y,z),pdf=rndmpdf).sample(N=6)
array([[  12.63103673,  186.7514952 , 1716.6187807 ],
       [  14.67375968,  116.20984414, 1557.59629547],
       [  11.47055697,  178.41650558, 1592.18260186],
       [  12.41780309,  105.28009531, 1436.39525998],
       [  13.44764381,  152.57623376, 1880.55963378],
       [  18.5522151 ,  133.87092063, 1558.85620176]])
```

See the [function docstring](./lintsampler) or the example notebooks for further details and examples.

As well as `LintSampler`, the package contains a few other public classes which all represent pdf evaluation domains the user can provide as the first parameter in the `LintSampler` constructor in more advanced use cases. `DensityGrid` and `DensityTree` are respectively grid-like and tree-like structures, while `DensityStructure` is the abstract base class which can be easily extended to create custom structures as desired. To see these classes in action, see the example notebooks.
