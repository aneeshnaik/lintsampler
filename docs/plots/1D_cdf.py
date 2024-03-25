import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('figstyle.mplstyle')

if __name__ == "__main__":

    x = np.linspace(-6, 6, 500)
    y = norm.cdf(x)
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # plot CDF
    ax.plot(x, y, c='teal', lw=2)

    # set x/y ticks
    ax.set_xticks(np.arange(-6, 6+2, 2))
    ax.set_yticks([0, 0.5, 1])
    
    # remove left/top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set x/y spine extents
    ax.spines['bottom'].set_bounds(-6, 6)
    ax.spines['left'].set_bounds(0, 1)
    
    # save
    fig.savefig("../source/assets/1D_cdf.png", bbox_inches='tight')
