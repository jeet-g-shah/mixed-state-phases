import matplotlib.pyplot as plt
import numpy as np


def heatmap(x, y, z, xlabel="", ylabel="", title="", zlabel="", cmap="RdBu"):
    # generate 2 2d grids for the x & y bounds
    y, x = np.meshgrid(y, x)

    fig, ax = plt.subplots()

    c = ax.pcolormesh(x, y, z, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    cbar = fig.colorbar(c, ax=ax)
    if zlabel:
        cbar.set_label(zlabel, rotation=270)

    return fig
