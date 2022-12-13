from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def ligand_receptor_scatter(
    actual_scores: Dict[Tuple[str, str], float],
    stat_scores: Dict[Tuple[str, str], Tuple[np.ndarray, float]],
    lr_pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plots a scatterplot highlighting the observed ligand-receptor scores and significance.

    Uses the results of :func:`monkeybread.calc.ligand_receptor_score` and
    :func:`monkeybread.stat.ligand_receptor_score`.

    Creates a scatterplot where each point is one ligand-receptor pair, the x-axis displays the
    observed score, and the y-axis displays the p-value from the statistical test.

    Parameters
    ----------
    actual_scores
        The observed scores, as calculated by :func:`monkeybread.calc.ligand_receptor_score`.
    stat_scores
        The permuted scores, as calculated by :func:`monkeybread.stat.ligand_receptor_score`.
    lr_pairs
        A subset of ligand-receptor pairs from `actual_scores` and `stat_scores` to plot.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to `seaborn.histplot`.

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
    within.
    """
    if ax is None:
        ax = plt.axes()

    # Convert lr_pairs to list
    if lr_pairs is None:
        lr_pairs = list(actual_scores.keys())
    elif isinstance(lr_pairs, tuple):
        lr_pairs = [lr_pairs]

    x_score = [actual_scores[lr] for lr in lr_pairs]
    y_pval = -np.log10([stat_scores[lr][1] for lr in lr_pairs])
    labels = [f"{lig}/{rec}" for lig, rec in lr_pairs]

    sns.scatterplot(x=x_score, y=y_pval, ax=ax)

    for x, y, s in zip(x_score, y_pval, labels):
        ax.text(x, y, s, fontsize="xx-small", ha="left", va="top")

    ax.axhline(-np.log10(0.05), 0, 1, color="lightgray", zorder=-10)

    ax.set_ylabel("-log10(pval)")
    ax.set_xlabel("Interaction Score")
    ax.set_title("Ligand-Receptor Interaction Significance")

    if show:
        plt.show()
    else:
        return ax
