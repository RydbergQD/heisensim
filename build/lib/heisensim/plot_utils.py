from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm


def create_line_segment(x, y, z, norm="Normalize", cmap="coolwarm", linewidth=2):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    if norm == "Normalize":
        norm = plt.Normalize(z.min(), z.max())
    elif norm == "LogNorm":
        norm = LogNorm(z.min(), z.max())
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z)
    lc.set_linewidth(linewidth)
    return lc


def plot_line_colormap(
    x,
    y,
    z,
    norm="Normalize",
    cmap="coolwarm",
    linewidth=2,
    fig=None,
    ax=None,
    add_cbar=True,
    cbar_label: None | str = None,
):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()
    if np.ndim(y) == 1 and np.ndim(z) == 1:
        lc = create_line_segment(x, y, z, norm=norm, cmap=cmap, linewidth=linewidth)
        line = ax.add_collection(lc)
    if np.ndim(y) == 1 and np.ndim(z) == 2:
        for zz in z:
            lc = create_line_segment(x, y, zz, norm=norm, cmap=cmap, linewidth=linewidth)
            line = ax.add_collection(lc)
    if np.ndim(y) == 2 and np.ndim(z) == 1:
        for yy in y:
            lc = create_line_segment(x, yy, z, norm=norm, cmap=cmap, linewidth=linewidth)
            line = ax.add_collection(lc)
    if np.ndim(y) == 2 and np.ndim(z) == 2:
        for yy, zz in zip(y, z):
            lc = create_line_segment(x, yy, zz, norm=norm, cmap=cmap, linewidth=linewidth)
            line = ax.add_collection(lc)
    if add_cbar:
        cbar = fig.colorbar(line, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    return fig, ax
