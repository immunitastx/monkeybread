from typing import List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import scanpy as sc
from anndata import AnnData

import monkeybread as mb

def embedding_filter(
    adata: AnnData,
    mask: Union[List[bool], List[str]],
    color: Optional[str] = None,
    basis: Optional[str] = "spatial",
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    masked_color: Optional[str] = "lightgrey",
    masked_alpha: Optional[float] = 1.0,
    masked_dot_size: Optional[float] = None,
    unmasked_dot_size: Optional[float] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """Shows a filtered embedding, allowing for examination of specific cells.

    Cells in the mask will be larger and colored based on `color` if provided, otherwise red. Cells
    not in the mask will be smaller and colored gray.

    Parameters
    ----------
    adata
        Annotated data matrix.
    mask
        A mask to apply to `adata.obs.index`. Can be a list of cell indices or a boolean mask with
        the same length as the index.
    color
        Column in `adata.obs` or `adata.var_names` used to color cells.
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

    # Calculate default dot sizes
    if masked_dot_size is None:
        masked_dot_size = 12000 / adata.shape[0]
    if unmasked_dot_size is None:
        unmasked_dot_size = (12000 / adata.shape[0]) * 5
      
    # Plot all cells in light gray
    sc.pl.embedding(
        adata, 
        basis=basis, 
        na_color=masked_color, 
        show=False, 
        alpha=masked_alpha,
        ax=ax, 
        size=masked_dot_size,
        colorbar_loc=None,
        **kwargs
    )

    # Plot subset cells, optionally colored (otherwise red)
    sc.pl.embedding(
        adata_sub,
        basis=basis,
        show=False,
        ax=ax,
        color=color,
        na_color="red",
        size=unmasked_dot_size,
        **kwargs,
    )

    if show:
        plt.show()
    else:
        return ax


def embedding_zoom(
    adata: AnnData,
    left_pct: float = None,
    top_pct: float = None,
    width_pct: float = None,
    height_pct: float = None,
    color: Optional[str] = None,
    mask: Optional[Union[List[bool], List[str]]] = None,
    basis: Optional[str] = "spatial",
    show: Optional[bool] = True,
    unzoom_s: Optional[float] = None,
    zoom_s: Optional[float] = None, 
    axs: Optional[List[plt.Axes]] = None,
    fig: Optional[plt.Figure] = None,
    **kwargs,
) -> Optional[plt.Figure]:
    """Shows embeddings of cells neighboring other cells with zoomed focus.

    Plots the results of :func:`monkeybread.calc.cell_neighbors`, highlighting the cell neighbors.
    Zooms in on a specific rectangle in the plot, based on fractional coordinate space, for higher-resolution
    viewing.

    Parameters
    ----------
    adata
        Annotated data matrix.
    left_pct
        The fraction of the plot width to use as the left bound of the zoomed-in rectangle, e.g. 0.1
        represents 10% from the left of the plot.
    top_pct
        The fraction of the plot height to use as the upper bound of the zoomed-in rectangle, e.g.
        0.3 represents 30% from the top of the plot.
    width_pct
        The fraction of the plot width to use for the zoomed-in rectangle, e.g. 0.5 represents a
        width half of the original plot.
    height_pct
        The fraction of the plot height to use for the zoomed-in rectangle, e.g. 0.25 represents a
        height 25% of the original plot.
    color
        Column in `adata.obs` used to color cells.
    mask
        A mask to apply to `adata.obs.index`. Can be a list of cell indices or a boolean mask with
        the same length as the index, as described in :func:`monkeybread.plot.embedding_filter`.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    show
        Whether to show the plot or return the Axes object.
    unzoom_s
        Dot-size in the zoomed-out figure passed to :func:`scanpy.pl.embedding`.
    zoom_s
        Dot-size in the zoomed-in figure passed to :func:`scanpy.pl.embedding`.
    kwargs
        Keyword arguments that will be passed to :func:`scanpy.pl.embedding`.

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Figure object the plots are contained
    within.

    Example
    -------

    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/embedding_zoom.png
    """
    if not all([left_pct, top_pct, width_pct, height_pct]):
        raise ValueError("Must provide left_pct, top_pct, width_pct, height_pct")

    # Place in units of fractions rather than percents
    left_pct /= 100
    top_pct /= 100
    width_pct /= 100 
    height_pct /= 100

    # Add dot size to key-word arguments
    if unzoom_s:
        unzoom_kwargs = dict(kwargs, s=unzoom_s)
    else:
        unzoom_kwargs = kwargs

    if zoom_s:
        zoom_kwargs = dict(kwargs, s=zoom_s)
    else:
        zoom_kwargs = kwargs

    # Set up plot structure and plot original data
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2)
    if mask is not None:
        embedding_filter(
            adata, 
            mask, 
            basis=basis, 
            color=color, 
            show=False, 
            ax=axs[0], 
            **unzoom_kwargs
        )
    else:
        sc.pl.embedding(
            adata, 
            basis=basis, 
            color=color, 
            show=False, 
            ax=axs[0], 
            **unzoom_kwargs
        )
    try:
        axs[0].get_legend().remove()
    except:
        pass

    # Get scaling information and add rectangle
    left, right = axs[0].get_xlim()
    bottom, top = axs[0].get_ylim()
    tot_width = right - left
    tot_height = top - bottom
    left_bound, zoom_width = (left + left_pct * tot_width, width_pct * tot_width)
    top_bound, zoom_height = (top - top_pct * tot_height, height_pct * tot_height)
    rect = mpl.patches.Rectangle(
        (left_bound, top_bound - zoom_height), 
        zoom_width, 
        zoom_height, 
        linewidth=1, 
        edgecolor="black", 
        facecolor="none"
    )
    axs[0].add_patch(rect)

    # Subset anndata object
    zoom_adata = mb.util.subset_cells(
        adata,
        by="spatial",
        subset=[
            ("x", "gte", left_bound),
            ("x", "lte", left_bound + zoom_width),
            ("y", "lte", top_bound),
            ("y", "gte", top_bound - zoom_height),
        ],
    )

    # Plot cell contact
    if mask is not None:
        zoom_mask = list(set(adata[mask].obs.index).intersection(set(zoom_adata.obs.index)))
        embedding_filter(
            zoom_adata, 
            zoom_mask, 
            basis=basis, 
            color=color, 
            show=False, 
            ax=axs[1], 
            **zoom_kwargs
        )
    else:
        sc.pl.embedding(
            zoom_adata, 
            basis=basis, 
            color=color,
            show=False, 
            ax=axs[1], 
            **zoom_kwargs
        )

    # Return fig or show
    if show:
        plt.show()
        return zoom_adata
    else:
        return zoom_adata, fig



