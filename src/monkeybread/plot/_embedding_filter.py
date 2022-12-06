from typing import List, Optional, Union

import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData


def embedding_filter(
    adata: AnnData,
    mask: Union[List[bool], List[str]],
    group: Optional[str] = None,
    basis: Optional[str] = "spatial",
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Shows a filtered embedding, allowing for examination of specific cells.

    Cells in the mask will be

    Parameters
    ----------
    adata
        Annotated data matrix.
    mask
        A mask to apply to `adata.obs.index`. Can be a list of cell indices or a boolean mask with
        the same length as the index.
    group
        Column in `adata.obs` to label cell contacts by.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to :func:`scanpy.pl.embedding`.

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
    within.
    """
    if ax is None:
        ax = plt.axes()

    # Subset adata to only include masked cells
    adata_sub = adata[mask]

    # Plot all cells in light gray
    sc.pl.embedding(
        adata, basis=basis, na_color="lightgrey", show=False, alpha=0.5, ax=ax, size=12000 / adata.shape[0], **kwargs
    )

    # Plot subset cells, optionally colored (otherwise red)
    sc.pl.embedding(
        adata_sub,
        basis=basis,
        show=False,
        ax=ax,
        color=group,
        na_color="red",
        size=(12000 / adata.shape[0]) * 5,
        **kwargs,
    )

    if show:
        plt.show()
    else:
        return ax
