from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def shortest_distances(
    distances: np.ndarray,
    expected_distances: Optional[Union[np.ndarray, Tuple[np.ndarray, float, float]]] = None,
    show: Optional[bool] = True,
    **kwargs,
) -> Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]:
    """Plots the results of shortest distances calculations in histogram format.

    Parameters
    ----------
    distances
        The shortest distances, as calculated by :func:`monkeybread.calc.shortest_distances`.
    expected_distances
        The expected distances, with optional threshold and p-value, as calculated by
        :func:`monkeybread.stat.shortest_distances`.
    show
        If true, displays the plot(s). If false, returns the Axes instead.
    kwargs
        Keyword arguments to pass to :func:`seaborn.histplot`.

    Returns
    -------
    If `show = False`, returns nothing. Otherwise, returns a single Axes object or a tuple of
    two Axes objects if expected_distances is provided.
    """
    # Set up plot structure
    ax = None
    axs: Tuple[Optional[plt.Axes], Optional[plt.Axes]] = (None, None)
    if expected_distances is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        ax = axs[0]

    # Plot actual distances on first axis
    ax = sns.histplot(list(map(float, np.transpose(distances)[1])), ax=ax, legend=None, stat="density", **kwargs)
    ax.set_title("Observed Shortest Distances")

    # Plot expected distances distribution on second axis if exists
    if expected_distances is not None:
        sns.histplot(expected_distances[0], ax=axs[1], stat="density", **kwargs)

        axs[1].set_title("Expected Shortest Distances")
        max_y = max(axs[0].get_ylim(), axs[1].get_ylim())
        axs[0].set_ylim(max_y)
        axs[1].set_ylim(max_y)
        x_bounds = min(axs[0].get_xlim()[0], axs[1].get_xlim()[0]), max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
        axs[0].set_xlim(x_bounds)
        axs[1].set_xlim(x_bounds)
        axs[0].set_xlabel("Distance")
        axs[1].set_xlabel("Distance")

        # If p-value and threshold included, add to plots
        if type(expected_distances) == tuple:
            axs[0].text(
                0.97,
                0.97,
                f"p-value: {expected_distances[2]:.2f}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=axs[0].transAxes,
            )
            threshold = expected_distances[1]
            threshold_line = axs[0].axvline(threshold, 0, 1.0, color="red", linestyle="--")
            axs[1].axvline(threshold, 0, 1.0, color="red", linestyle="--")
            axs[0].legend(handles=[threshold_line], labels=["Threshold"], loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

    if show:
        plt.show()
    else:
        return axs if axs is not None else ax
