import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use('figstyle.mplstyle')

if __name__ == "__main__":

    x = np.linspace(-6, 6, 500)
    y = norm.pdf(x)
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    
    # plot PDF
    ax.plot(x, y, c='teal', lw=2)

    # set x ticks
    ax.set_xticks(np.arange(-6, 6+2, 2))

    # remove y axis labels/ticks
    ax.tick_params(left=False, labelleft=False)
    
    # remove left/top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # set x spine extent
    ax.spines['bottom'].set_bounds(-6, 6)
    
    # save
    fig.savefig("../source/assets/1D_pdf.png", bbox_inches='tight')
