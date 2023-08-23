"""
Visualization of ligand-receptor co-expression score between neighboring cells.
"""

from typing import Dict, List, Optional, Tuple, Union, Set

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
import itertools
from collections import defaultdict
import matplotlib.colors as mcolors
from sklearn.preprocessing import scale

from ._embedding_other import embedding_filter, embedding_zoom


def _score_pairs(
        adata,
        lr_pair,
        cell_to_neighbor
    ):
    """
    Helper function compute ligand-receptor co-expression score between
    neighboring cells.

    The ligand-receptor score is described in :cite:p:`He2021.11.03.467020`.
    For a given pair of cells, it is expressed as `sqrt(l*r)` where `l` is the
    expression of the ligand and `r` is the expression of the receptor.
    """
    # Convert cell_to_neighbors dictionary to list of cell_id tuples
    neighbor_pairs = np.array(
        list(
            itertools.chain.from_iterable(
                itertools.product([cell], neighbors)
                for cell, neighbors in cell_to_neighbor.items()
            )
        )
    )

    # Score the ligand-receptor interaction for each pair
    pair_scores = np.sqrt(
        adata[neighbor_pairs.T[0], lr_pair[0]].X.toarray() \
        * adata[neighbor_pairs.T[1], lr_pair[1]].X.toarray()
    )
    pairs_w_scores = list(zip(
        [tuple(x) for x in neighbor_pairs],
        [x[0] for x in pair_scores]
    ))
    pairs_w_scores = sorted(
        pairs_w_scores,
        key=lambda x: x[1],
        reverse=True
    )

    return pairs_w_scores


def ligand_receptor_embedding(
    adata: AnnData,
    lr_pair: Tuple[str, str],
    cell_to_neighbor: Dict[str, Set[str]],
    color: str,
    basis: Optional[str] = 'X_spatial',
    palette: Optional[str] = None,
    cmap_name: Optional[str] = 'magma_r',
    encode_lw: Optional[bool] = True,
    max_lw: Optional[float] = 3.,
    lr_colorbar: Optional[bool]=True,
    ax: plt.Axes = None,
    show: Optional[bool] = True,
    **kwargs
) -> Dict[Tuple[str, str], float]:
    """Plot the ligand-receptor co-expression score between pairs of neighboring cells.
    A line is drawn between each pair of neighbors cells where that 
    line is colored according to the magnitude of their ligand-receptor co-expression 
    score.

    The ligand-receptor score is described in :cite:p:`He2021.11.03.467020`.
    For a given pair of cells, it is expressed as `sqrt(l*r)` where `l` is the 
    expression of the ligand and `r` is the expression of the receptor.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbor
        A mapping of each cell to its neighbors, as calculated by :func:`monkeybread.calc.cell_neighbors`.
    lr_pair
        A ligand-receptor pair where the ligand is expressed in the cells that 
        act as keys in the `cell_to_neighbor` dictonary argument and the receptor is 
        is expressed in the cells that are values in the `cell_to_neighbor` dictonary 
        argument.
    color
        Column in `adata.obs` used to color cells.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`. 
    palette
        Palette used to color cells
    cmap_name
        Colormap used to color lines connecting cells to their neighbors
    encode_lw
        Draw the width of each line connecting neighboring cells according to the magnitude of their 
        ligand-receptor score. Neighboring cells with a low ligand-receptor co-expression score will 
        be connected by a thinly drawn line whereas a pair of cells with a high ligand-receptor 
        co-expression score will be connected by a thickly drawn line. If `False`, all lines will be drawn 
        with equal thickness defined by the `max_lw` argument.
    max_lw
        If `encode_lw = True`, this is thickness of the line connecting the pair of cells with
        maximum ligand-receptor score. If `encode_lw = False`, this is the thickness connecting
        all pairs of neighboring cells.
    lr_colorbar
        If `True` draw a colorbar depicting the ligand-receptor scores. If `False`, hide this
        colorbar
    show
        Whether to show the plot or return the Axes object.
    kwargs
        Keyword arguments that will be passed to :func:`monkeybread.plot.embedding_zoom`.

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Figure and Axes containing
    the figure.

    Example
    -------

    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/ligand_receptor_embedding.png
    """
    # Score each neighbor pair for ligand-receptor expression
    pairs_w_scores = _score_pairs(
        adata,
        lr_pair,
        cell_to_neighbor
    )

    # Aggregate all cells being plotted
    sources = set(cell_to_neighbor.keys())
    targets = set()
    for vals in cell_to_neighbor.values():
        targets.update(vals)
    plot_cells = list(sources | targets)

    # Scale the ligand-receptor scores to be between zero and one
    min_score = min([x[1] for x in pairs_w_scores])
    max_score = max([x[1] for x in pairs_w_scores])
    scaled_scores = [
        (min([x[1], max_score]) - min_score) / (max_score-min_score)
        for x in pairs_w_scores
    ]

    # Get colormap, palette, and axes
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(5,5))
    else:
        fig = None
    if palette is None:
        palette = sns.color_palette()
    cmap = plt.cm.get_cmap(cmap_name)


    # Compute the line-widths
    if encode_lw:
        lws = [
            (min([x[1], max_score]) - min_score) / (max_score-min_score) * max_lw
            for x in pairs_w_scores
        ]
    else:
        lws = [max_lw for x in pairs_w_scores]

    # Plot the cell embedding
    embedding_filter(
        adata,
        mask=adata.obs.index.isin(plot_cells),
        color=color,
        palette=palette,
        show=False,
        ax=ax,
        **kwargs
    )

    # Plot the lines between neighboring cells colored by ligand-receptor score
    for (pair, score), scaled_score, lw in zip(pairs_w_scores, scaled_scores, lws):
        coords = adata[[pair[0], pair[1]],:].obsm[basis]
        ax.plot(
            [coords[0][0], coords[1][0]],
            [coords[0][1], coords[1][1]],
            color=cmap(scaled_score),
            ls='-',
            lw=lw
        )

    # Plot ligand-receptor colorbar
    if lr_colorbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(
                norm=plt.Normalize(min_score, max_score),
                cmap=cmap_name
            ),
            ax=ax,
            orientation="horizontal"
        )
        cbar.ax.set_title(f'{lr_pair[0]}, {lr_pair[1]} Co-expression Score')

    if show:
        plt.show() 
    else:
        return fig, ax




def ligand_receptor_embedding_zoom(
    adata: AnnData,
    lr_pair: Tuple[str, str],
    cell_to_neighbor: Dict[str, Set[str]],
    color: str,
    basis: Optional[str] = 'X_spatial',
    left_pct: Optional[float] = None,
    top_pct: Optional[float] = None,
    width_pct: Optional[float] = None,
    height_pct: Optional[float] = None,
    palette: Optional[str] = None,
    cmap_name: Optional[str] = 'magma_r',
    encode_lw: Optional[bool] = True,
    max_lw: Optional[float] = 3.,
    max_lw_scale_unzoom: Optional[float] = 0.1,
    lr_colorbar: Optional[bool]=True,
    axs: Optional[Union[plt.Axes, Tuple[plt.Axes, plt.Axes]]] = None,
    show: Optional[bool] = True,
    **kwargs
) -> Union[Tuple[plt.Figure, plt.Axes], None]:
    """Plot the ligand-receptor co-expression score between pairs of neighboring cells. 
    A line is drawn between each pair of neighbors cells where that 
    line is colored according to the magnitude of their ligand-receptor co-expression 
    score.

    The ligand-receptor score is described in :cite:p:`He2021.11.03.467020`.
    For a given pair of cells, it is expressed as `sqrt(l*r)` where `l` is the 
    expression of the ligand and `r` is the expression of the receptor.

    This function plots the full tissue slice as well as a specified zoomed-in 
    region of the tissue by calling :func:`monkeybread.plot.embedding_zoom`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbor
        A mapping of each cell to its neighbors, as calculated by :func:`monkeybread.calc.cell_neighbors`.
    lr_pair
        A ligand-receptor pair where the ligand is expressed in the cells that 
        act as keys in the `cell_to_neighbor` dictonary argument and the receptor is 
        is expressed in the cells that are values in the `cell_to_neighbor` dictonary 
        argument.
    color
        Column in `adata.obs` used to color cells.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.
    left_pct
        The fraction of the plot width to use as the left bound of the zoomed-in rectangle (e.g. 0.1
        represents 10% from the left of the plot)
    top_pct
        The fraction of the plot height to use as the upper bound of the zoomed-in rectangle (e.g.
        0.3 represents 30% from the top of the plot)
    width_pct
        The fraction of the plot width to use for the zoomed-in rectangle (e.g., 0.5 represents a
        width half of the original plot)
    height_pct
        The fraction of the plot height to use for the zoomed-in rectangle (e.g., 0.25 represents a
        height 25% of the original plot)
    palette
        Palette used to color cells
    cmap_name
        Colormap used to color lines connecting cells to their neighbors
    encode_lw
        Draw the width of each line connecting neighboring cells according to the magnitude of their 
        ligand-receptor score. Neighboring cells with a low ligand-receptor co-expression score will 
        be connected by a thinly drawn line whereas a pair of cells with a high ligand-receptor 
        co-expression score will be connected by a thickly drawn line. If `False`, all lines will be drawn 
        with equal thickness defined by the `max_lw` argument.
    max_lw
        If `encode_lw = True`, this is thickness of the line connecting the pair of cells with 
        maximum ligand-receptor score. If `encode_lw = False`, this is the thickness connecting
        all pairs of neighboring cells.
    lr_colorbar 
        If `True` draw a colorbar depicting the ligand-receptor scores. If `False`, hide this
        colorbar
    show
        Whether to show the plot or return the Axes object.
    kwargs
        Keyword arguments that will be passed to :func:`monkeybread.plot.embedding_zoom`.
        
    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Figure and Axes containing
    the figure

    Example
    -------
    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/ligand_receptor_score_embedding_zoom.png
    """
    # Score each neighbor pair for ligand-receptor expression
    pairs_w_scores = _score_pairs(
        adata,
        lr_pair,
        cell_to_neighbor
    )
 
    # Aggregate all cells being plotted
    sources = set(cell_to_neighbor.keys())
    targets = set()
    for vals in cell_to_neighbor.values():
        targets.update(vals)
    plot_cells = list(sources | targets)

    # Scale the ligand-receptor scores to be between zero and one
    min_score = min([x[1] for x in pairs_w_scores])
    max_score = max([x[1] for x in pairs_w_scores])
    scaled_scores = [
        (min([x[1], max_score]) - min_score) / (max_score-min_score)
        for x in pairs_w_scores
    ]

    # Compute the line-widths
    if encode_lw:
        lws = [
            (min([x[1], max_score]) - min_score) / (max_score-min_score) * max_lw
            for x in pairs_w_scores
        ]
    else:
        lws = [max_lw for x in pairs_w_scores]
   
    # Get axes, palette, and colormap
    if axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=2)
    if palette is None:
        palette = sns.color_palette()
    cmap = plt.cm.get_cmap(cmap_name)

    # Draw embedding
    zoom_adata, _ = embedding_zoom(
        adata,
        mask=adata.obs.index.isin(plot_cells),
        left_pct=left_pct,
        top_pct=top_pct,
        width_pct=width_pct,
        height_pct=height_pct,
        color=color,
        palette=palette,
        show=False,
        axs=axs,
        **kwargs
    )

    # Plot the lines between neighboring cells colored by ligand-receptor score
    for (pair, score), scaled_score, lw in zip(pairs_w_scores, scaled_scores, lws):
        coords = adata[[pair[0], pair[1]],:].obsm[basis]
        axs[0].plot(
            [coords[0][0], coords[1][0]], 
            [coords[0][1], coords[1][1]], 
            color=cmap(scaled_score), 
            ls='-', 
            lw=lw*max_lw_scale_unzoom
        )
        if pair[0] in zoom_adata.obs.index and pair[1] in  zoom_adata.obs.index:
            coords = zoom_adata[[pair[0], pair[1]],:].obsm[basis]
            axs[1].plot(
                [coords[0][0], coords[1][0]], 
                [coords[0][1], coords[1][1]], 
                color=cmap(scaled_score), 
                ls='-', 
                lw=lw
            )

    # Plot ligand-receptor colorbar
    if lr_colorbar:
        cbar = plt.colorbar(
            mpl.cm.ScalarMappable(
                norm=plt.Normalize(min_score, max_score), 
                cmap=cmap_name
            ), 
            ax=axs[1], 
            orientation="horizontal"
        )
        cbar.ax.set_title(
            f'{lr_pair[0]}, {lr_pair[1]} Co-expression Score'
        )

    if show:
        plt.show()
    else:
        return fig, axs


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

    Example
    -------
    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/ligand_receptor_scatter.png
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


def ligand_receptor_score_barplot_per_niche(
        niche_to_lr_to_score: Dict[str, Dict[Tuple[str, str], float]], 
        lr_pair: List[Tuple[str, str]], 
        plot_niches: Optional[List[str]]=None,
        ax: Optional[plt.Axes]=None,
        title: Optional[str]=None,
        show: Optional[bool]=True,
        **barplot_kwargs
    ) -> Union[None, plt.Axes]:
    """
    Create a barplot showing the ligand/receptor score for a specific ligand/receptor
    pair within each niche. This function plots the data returned by 
    :func:`monkeybread.calc.ligand_receptor_score_per_niche`.

    Parameters
    ----------
    niche_to_lr_to_score
        A dictionary mapping each niche to a sub-dictionary mapping each ligand/receptor 
        (represented as a tuple) to its score in that niche. This data structure is 
        returned by :func:`monkeybread.calc.ligand_receptor_score_per_niche`
    lr_pair
        Specific ligand/receptor pair to plot
    plot_niches
        Specific niches to plot. If `None`, all niches will be plotted
    ax
        Axes object to plot
    title
        Title of figure
    show
        If `True`, show the plot. If `False` return the Axes object the plot is contained
        within
    barplot_kwargs
        Arguments passed to :func:`seaborn.barplot`
    
    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
    within.

    Example
    -------
    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/ligand_receptor_score_barplot_per_niche.png
    """
    if ax is None:
        ax = plt.axes()

    da = []
    if not plot_niches:
        plot_niches = sorted(niche_to_lr_to_score.keys())
    for niche in plot_niches:
        lr_to_score = niche_to_lr_to_score[niche] 
        da.append((niche, lr_to_score[lr_pair]))
    plot_df = pd.DataFrame(
        data=da,
        columns=['Niche', 'Ligand/Receptor Score']
    )
    sns.barplot(
        plot_df, 
        x='Niche', 
        y='Ligand/Receptor Score', 
        ax=ax,
        order=plot_niches,
        **barplot_kwargs
    )
    if title:
        ax.set_title
    else:
        ax.set_title(
            f'{lr_pair[0]}/{lr_pair[1]} Co-expression Score'
        )
    if show:
        plt.show()
    else:
        return ax


def ligand_receptor_score_heatmap_per_niche(
        niche_to_lr_to_score: Dict[str, Dict[Tuple[str, str], float]],
        lr_pairs: Optional[List[Tuple[str,str]]]=None,
        plot_niches: Optional[List[str]]=None,
        z_score: Optional[List[str]]=True,
        clip_z: Optional[float]=2.,
        show: Optional[bool]=True,
        **clustermap_kwargs
    ) -> Union[None, sns.matrix.ClusterGrid]:
    """
    Create a heatmap showing the ligand/receptor scores for a specific set of 
    ligand/receptor pairs within each niche. The rows represent ligand/receptors
    and the columns represent niches. This function plots the data returned by 
    :func:`monkeybread.calc.ligand_receptor_score_per_niche`.

    Parameters
    ----------
    niche_to_lr_to_score
        A dictionary mapping each niche to a sub-dictionary mapping each ligand/receptor 
        (represented as a tuple) to its score in that niche. This data structure is 
        returned by :func:`monkeybread.calc.ligand_receptor_score_per_niche`
    lr_pairs
        Specific set of ligand/receptor pairs to plot. If `None`, all ligand/receptor
        pairs in the `niche_to_lr_to_score` data structure will be plotted
    plot_niches
        Specific niches to plot. If `None`, all niches will be plotted
    z_score
        If `True` plot the z-score normalized ligand-receptor scores, where the scores
        are normalized for each ligand/receptor pairs accross all niches
    clip_z
        If `z_score` is True, this clips the plotted z-score values at +/- `clip_z`
    show
        If `True`, show the plot. If `False` return the ClusterGrid object
        output by :func:`seaborn.clustermap`
    clustermap_kwargs
        Arguments passed to :func:`seaborn.clustermap`

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the ClusterGrid object
    output by :func:`seaborn.clustermap`

    Example
    -------
    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/ligand_receptor_score_heatmap_per_niche.png
    """
    da = []
    if not plot_niches:
        plot_niches = sorted(niche_to_lr_to_score.keys())
    if not lr_pairs:
        any_niche = sorted(niche_to_lr_to_score.keys())[0]
        lr_pairs = sorted(niche_to_lr_to_score[any_niche].keys())

    da = []
    for lr_pair in lr_pairs:
        row = []
        for niche in plot_niches:
            score = niche_to_lr_to_score[niche][lr_pair]
            row.append(score)
        da.append(row)
    
    if z_score:
        da=scale(da, axis=1)
    
    plot_df = pd.DataFrame(      
        data=da,
        columns=plot_niches,
        index=[', '.join(x) for x in lr_pairs]
    )

    if z_score:
        if clip_z:
            vmin=-clip_z
            vmax=clip_z
        else:
            vmin=None
            vmax=None
        cg = sns.clustermap(
            plot_df,
            vmin=vmin,
            vmax=vmax,
            **clustermap_kwargs
        )
    else:
        cg = sns.clustermap(
            plot_df,
            **clustermap_kwargs
        )

    if show:
        plt.show()
    else:
        return cg
