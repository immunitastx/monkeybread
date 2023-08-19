from typing import Dict, Optional, Set, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from anndata import AnnData
from matplotlib.legend_handler import HandlerTuple

import monkeybread as mb


def cell_neighbor_embedding(
    adata: AnnData,
    cell_to_neighbors: Dict[str, Set[str]],
    group: Optional[str] = None,
    basis: Optional[str] = "X_spatial",
    group1_name: Optional[str] = "Group 1",
    group2_name: Optional[str] = "Group 2",
    dot_size_group1: Optional[float] = None,
    dot_size_group2: Optional[float] = None,
    dot_size_unselected: Optional[float] = None,
    palette: Optional[List[str]] = None,
    cell_color_unselected: Optional[str] = "lightgrey",
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Shows embeddings of cells with their neighbors.

    Plots the results of :func:`monkeybread.calc.cell_neighbors`, highlighting the cells and their neighbors
    within the tissue.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        Cells and their associated neighbors as calculated by :func:`monkeybread.calc.cell_neighbors`.
    group
        Column in `adata.obs` for which to color cells. If `None`, cells will be labeled according
        to the `group1_name` and `group2_name` arguments.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `X_spatial`.
    group1_name
        Name of the first group of cells. Used to label the cells in the first group
    group2_name
        Name of the second group of cells. Used to label the cells in the second group
    dot_size_group1
        Size of the dots denoting cells in the first group
    dot_size_group2
        Size of the dots denoting cells in the second group
    dot_size_unselected
        Size of the dots denoting cells in neither the first nor second group
    palette
        Color palette used to color the cells
    cell_color_unselected
        Color of cells that are neither in the first nor second group
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

    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/cell_neighbor_embedding.png
    """
    if ax is None:
        ax = plt.axes()

    # Subset adata to only include cells in first group
    cells_g1 = set(cell_to_neighbors.keys())
    adata_group1 = adata[list(cells_g1)]

    # Subset adata to only include cells in second group
    cells_g2 = set()
    for s in cell_to_neighbors.values():
        cells_g2 = cells_g2.union(s)
    adata_group2 = adata[list(cells_g2)]

    # Custom legend handler for circles
    class HandlerCircle(HandlerTuple):
        def create_artists(
            self, legend, orig_handle,
            xdescent, ydescent, width, height, fontsize, trans
        ):
            x = width / 2
            y = height / 2
            radius = min(width, height) / 2.5
            color = orig_handle[0]
            circle = plt.Circle((x, y), radius, color=color)
            return [circle]

    if palette is None:
        palette = sc.pl.palettes.godsnot_102

    if group is None:
        # Plot all cells in light gray
        if dot_size_unselected is None:
            dot_size_unselected = 12000 / adata.shape[0]
        sc.pl.embedding(
            adata, 
            basis=basis, 
            na_color=cell_color_unselected, 
            show=False, 
            alpha=0.5, 
            ax=ax, 
            size=12000 / adata.shape[0], 
            **kwargs
        )

        # Plot cells, optionally colored (otherwise red)
        if dot_size_group1 is None:
            dot_size_group1 = 12000 / adata.shape[0]
        sc.pl.embedding(
            adata_group1,
            basis=basis,
            show=False,
            ax=ax,
            color=group,
            na_color=palette[0],
            alpha=1.0,
            size=dot_size_group1,
            **kwargs,
        )

        if dot_size_group2 is None:
            dot_size_group2 = 12000 / adata.shape[0]
        sc.pl.embedding(
            adata_group2,
            basis=basis,
            show=False,
            ax=ax,
            color=group,
            na_color=palette[1],
            alpha=1.0,
            size=dot_size_group2,
            **kwargs,
        )

        labels = [group1_name, group2_name]
        colors = [palette[0], palette[1]]
    else:
        if dot_size_unselected is None:
            dot_size_unselected = 12000 / adata.shape[0]
        sc.pl.embedding(
            adata,
            basis=basis,
            na_color=cell_color_unselected,
            show=False,
            alpha=0.5,
            ax=ax,
            size=12000 / adata.shape[0],
            **kwargs
        )

        # Plot cells, optionally colored (otherwise red)
        if dot_size_group1 is None:
            dot_size_group1 = 12000 / adata.shape[0]

        num_cats_group1 = len(set(adata_group1.obs[group]))
        cats_group1 = sorted(set(adata_group1.obs[group]))
        adata_group1.obs[group] = adata_group1.obs[group].astype('category')
        adata_group1.obs[group].cat.reorder_categories(cats_group1)
        
        colors = list(palette[:num_cats_group1])
        sc.pl.embedding(
            adata_group1,
            basis=basis,
            show=False,
            ax=ax,
            color=group,
            palette=palette,
            #na_color=cell_color_group1,
            alpha=1.0,
            size=dot_size_group1,
            **kwargs,
        )

        num_cats_group2 = len(set(adata_group2.obs[group]))
        cats_group2 = sorted(set(adata_group2.obs[group]))
        adata_group2.obs[group] = adata_group2.obs[group].astype('category')
        adata_group2.obs[group].cat.reorder_categories(cats_group2)
        if dot_size_group2 is None:
            dot_size_group2 = 12000 / adata.shape[0]

        colors += palette[num_cats_group1:num_cats_group1+num_cats_group2]
        sc.pl.embedding(
            adata_group2,
            basis=basis,
            show=False,
            ax=ax,
            color=group,
            palette=palette[num_cats_group1:], # Use new colors
            #na_color=cell_color_group2,
            alpha=1.0,
            size=dot_size_group2,
            **kwargs,
        )

        labels=cats_group1+cats_group2

    # Create a custom legend with circle color patches
    handles = [(color,) for color in colors]
    ax.legend(
        handles=handles, 
        labels=labels,
        handler_map={tuple: HandlerCircle()},
        loc='center left', 
        bbox_to_anchor=(1, 0.5),
        frameon=False
    )

    if show:
        plt.show()
    else:
        return ax



