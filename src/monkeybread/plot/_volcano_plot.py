from typing import List, Optional, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns


def volcano_plot(
    adata: ad.AnnData,
    group: Union[str, List[str]],
    key: Optional[str] = "rank_genes_groups",
    title: Optional[str] = None,
    adjusted_pvals: Optional[bool] = False,
    show: Optional[bool] = True,
    filter_kwargs: Optional[dict] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plots the results of :func:`scanpy.tl.rank_genes_groups` in the form of a volcano plot.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group
        Which group (as in :func:`scanpy.tl.rank_genes_groups`’s groupby argument) to return
        results from. Can be a list. All groups are returned if `groups` is None.
    key
        Key differential expression groups were stored under.
    title
        Title of the resulting plot.
    adjusted_pvals
        Use adjusted p-values instead of raw p-values.
    show
        Whether to show the plot or return it
    filter_kwargs
        Keyword arguments to pass into :func:`scanpy.get.rank_genes_groups_df`.
    kwargs
        Keyword arguments to pass into :func:`seaborn.scatterplot`.

    Returns
    -------
    If `show = False` returns the current axes. If `show = True` returns nothing.
    """
    if filter_kwargs is None:
        filter_kwargs = {}
    if title is None:
        title = f"{key} {group}"

    # Pull dataframe from adata object, and select columns of interest
    de_df = sc.get.rank_genes_groups_df(adata, group=group, key=key, **filter_kwargs)
    logfold = de_df["logfoldchanges"]
    pvals = de_df["pvals_adj" if adjusted_pvals else "pvals"]

    # Plot logfold and -log pvals
    ax = sns.scatterplot(x=logfold, y=np.negative(np.log10(pvals)), legend=None, **kwargs)

    # Add significance line at p = 0.05 and set title and axis labels
    ax.axhline(-np.log10(0.05), 0, 1, color="lightgray", zorder=-10)
    ax.set(ylabel="-log10(pval)", xlabel="logfoldchange", title=title)

    if show:
        plt.show()
    else:
        return ax
