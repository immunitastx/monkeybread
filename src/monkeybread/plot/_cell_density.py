import math
from typing import Dict, Optional, Union, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import scanpy as sc
import numpy as np
from anndata import AnnData

import monkeybread as mb

def _create_colormap(color1, color2, num_steps=100, show=False):
    # Convert color strings to RGB values
    rgb1 = mcolors.hex2color(color1)
    rgb2 = mcolors.hex2color(color2)

    # Create a linearly spaced array between the two colors
    r = [rgb1[0], rgb2[0]]
    g = [rgb1[1], rgb2[1]]
    b = [rgb1[2], rgb2[2]]
    cmap_data = [list(x) for x in zip(r, g, b)]
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', cmap_data, num_steps)
    return cmap


def cell_density(
    adata: Union[AnnData, pd.Series, Dict[str, pd.Series]],
    key: Union[str, Dict[str, str]],
    spot_size: Optional[float] = None,
    alpha: Optional[float] = 1.,
    cmap: Optional[str] = None,
    legend_loc: Optional[str] = 'right margin',
    show: Optional[bool] = True,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[Union[plt.Figure, plt.Axes]]:
    """Plots the spatial density of cells across the tissue, as calculated by 
    :func:`monkeybread.calc.cell_density`.

    Parameters
    ----------
    adata
        Annotated data matrix
    key
        Either a key in `adata.obs` or a mapping of group names to keys in `adata.obs` corresponding
        to density columns
    spot_size
        The size of spots to plot
    alpha
        The alpha blending value, between 0 (transparent) and 1 (opaque).
    cmap
        Colormap to use for values 0 to 1
    show_legend
        If `True` show the legend. Otherwise, hide it.
    show
        Whether to show the plot or return it
    title
        Title of the plot
    ax
        An Axes object to add the plot to. Only works if `key` is a single key.

    Returns
    -------
    If `show = False` returns the current figure (if `key` is a mapping) or the current axes
    (if `key` is a string). If `show = True` returns nothing.

    Example
    -------
    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/kernel_density.png
    """
    if type(key) == dict:
        # Set up subplot dimensions (max columns 4)
        ncols = min(len(key), 4)
        nrows = math.ceil(len(key) / ncols)

        for (index, (category, column)) in enumerate(key.items()):
            # Plot recursively for each column calculated
            axs = plt.subplot(nrows, ncols, index + 1)
            cell_density(
                adata,
                key=column,
                spot_size=spot_size,
                cmap=cmap,
                show=False,
                title=category,
                ax=axs,
            )

        # Add some whitespace
        plt.subplots_adjust(
            left=0.1, 
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.4, 
            hspace=0.4
        )
        if show:
            plt.show()
        else:
            return plt.gcf()
    else:
        # Use scanpy's built in embedding, coloring based on density key added to adata
        axs = sc.pl.embedding(
            adata, 
            basis="spatial", 
            color=key, 
            s=spot_size, 
            cmap=cmap, 
            alpha=alpha,
            show=show, 
            legend_loc=legend_loc,
            title=title, 
            ax=ax, 
            vmin=0.0, 
            vmax=1.0
        )
        if not show:
            return axs


def location_and_density(
    adata: AnnData,
    groupby: str,
    groups: Union[str, List[str], List[List[str]]],
    groupnames: Optional[Union[str, List[str]]] = None,
    plot_density: Optional[bool] = True,
    resolution: Optional[bool] = 10,
    bandwsith: Optional[float] = 75,
    dot_size: Optional[Union[float, List[float]]] = 7,
    na_dot_size: Optional[float] = 1.5,
    title: Optional[str] = None,
    grid: Optional[bool] = True,
    n_grids: Optional[int] = 5,
    palette: Optional[List[str]] = None,
    delete_temp_columns: Optional[bool] = False,
    show: Optional[bool] = True
) -> Union[None, Tuple[plt.Figure, List[plt.Axes]]]:
    """
    A wrapper around both :func:`scanpy.pl.embedding` and :func:`monkeybread.calc.cell_density` that 
    creates multi-panel figures showing both the raw location of cells of a given cell type, by calling
    :func:`scanpy.pl.embedding` and their density across the tissue by calling 
    :func:`monkeybread.calc.cell_density`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A key in `adata.obs` that store the cell type labels to be plotted.
    groups
        Either a single cell type label in `adata.obs[{key}]`, a list of cell type labels in `adata.obs[{key}]`, 
        or a list of lists of cell type labels in `adata.obs[{key}]`. If provided a list of lists, then
        each sub-list is considered one cell type and a union of all labels in that sublist will be used to
        group the cells (e.g., [['CD4 memory T cell', 'CD4 naive T cell'], ['CD8 memory T cell', 'CD8 naive T cell']]
        encodes two groups of cells: CD4 T cells and CD8 T cells)
    groupnames
        If `groups` is a list of lists, then this list of groupnames stores a list of strings corresponding
        to each sublist and is used to label the cells in that group. (e.g., in the example above, such a list
        might be ['CD4 T cell', 'CD8 T cell'] that will label the CD4 and CD8 T cells)
    plot_density
        Plot the density plots alongside the embedding plot.
    resolution
        Resolution parameter to pass to :func:`monkeybread.calc.cell_density`
    bandwidth
        Bandwidth parameter to pass to :func:`monkeybread.calc.cell_density`
    dot_size
        The size of dots in the embedding for the cells of intests (specified in `groups`). To plot
        each group with a different dot size, a list of sizes can be provided.
    na_dot_size
        The size of dots in the embedding for cells that are not of interest (specified in `groups`)
    title
        Title of the first figure showing the embedding of all of the cells together in one plot.
    grid
        If `True`, draw grid lines in each plot
    n_grids
        Number of gridlines to be drawn if `grid` is `True`
    palette
        Color palette used to color each group
    delete_temp_columns
        Delete columns added to `adata.obs` that were added during the creation of these plots.
    show
        If `True` display the figure. Otherwise return the `plt.Figure` and `plt.Axes` objects.

    Returns
    -------
    If `show = False` returns the current figure and axes. If `show = True` returns nothing.

    Example
    -------
    The left-most figure displays the locations of B cells (red) and CD4 Tfh cells (blue) in the tissue. The center
    and right-most figures display the density of B cells and CD4 Tfh cells, respectively, across the tissue.

    .. image:: https://raw.githubusercontent.com/immunitastx/monkeybread/main/docs/_static/location_and_density.png
    """

    if type(groups) == str:
        groups = [[groups]]
        groupnames = [groupnames]
    elif type(groups) == list and type(groups[0]) == str:
        groups = [groups]

    if type(dot_size) == int or type(dot_size) == float:
        dot_size = [dot_size for i in range(len(groups))]

    if plot_density:
        fig, axarr = plt.subplots(1, len(groups)+1, figsize=(3*len(groups)+3,3))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(3,3))
        axarr = [ax]

    if palette is None:
        palette = [
            "#0652ff", #  electric blue
            "#e50000", #  red
            "#9a0eea", #  violet
            "#01b44c", #  shamrock
            "#fedf08", #  dandelion
            "#00ffff", #  cyan
            "#89fe05", #  lime green
            "#a2cffe", #  baby blue
            "#dbb40c", #  gold
            "#029386", #  teal
            "#ff9408", #  tangerine
            "#d8dcd6", #  light grey
            "#80f9ad", #  seafoam
            "#3d1c02", #  chocolate
            "#fffd74", #  butter yellow
            "#536267", #  gunmetal
            "#f6cefc", #  very light purple
            "#650021", #  maroon
            "#020035", #  midnight blue
            "#b0dd16", #  yellowish green
            "#9d7651", #  mocha
            "#c20078", #  magenta
            "#380282", #  indigo
        ]
    for group_i, (group, groupname, color, d_size) in enumerate(zip(groups, groupnames, palette, dot_size)):
        ct_to_in_group = {
            ct: str(ct in group)
            for ct in set(adata.obs[groupby])
        }
        adata.obs[f'is_{groupname}'] = [
            ct_to_in_group[ct]
            for ct in adata.obs[groupby]
        ]
        if group_i == 0:
            ax = sc.pl.embedding(
                adata,
                "spatial",
                color = f'is_{groupname}',
                groups = ['True'],
                palette=[color],
                na_color='#f2f2f2',
                size=[
                    d_size if val in set(['True'])
                    else na_dot_size
                    for val in adata.obs[f'is_{groupname}']
                ],
                na_in_legend=False,
                ax=axarr[0],
                show=False
            )
        else:
            ax = sc.pl.embedding(
                adata[adata.obs[f'is_{groupname}'] == 'True'],
                "spatial",
                color=f'is_{groupname}',
                palette=[color],
                na_color='#f2f2f2',
                size=d_size,
                ax=axarr[0],
                show=False
            )

    axarr[0].legend().remove()
    axarr[0].set_title(title)

    # Add grid lines
    if grid:
        x_lim = axarr[0].get_xlim()
        y_lim = axarr[0].get_ylim()

        x_grid_space = (x_lim[1] - x_lim[0])/(n_grids)
        y_grid_space = (y_lim[1] - y_lim[0])/(n_grids)
        axarr[0].set_xticks(
            np.arange(
                x_lim[0],
                x_lim[1],
                x_grid_space
            )
        )
        axarr[0].set_yticks(
            np.arange(
                y_lim[0],
                y_lim[1],
                y_grid_space
            )

        )
        axarr[0].grid(
            visible=True,
            which='major',
            linestyle='--',
            lw=0.25,
            color='black'
        )
        axarr[0].set_xticklabels([])
        axarr[0].set_yticklabels([])

    # Create density plots
    if plot_density:
        cmaps = [
            _create_colormap('#ffffff', palette[i], num_steps=100)
            for i in range(len(groups))
        ]

        for group_i, (group, groupname, cmap, ax) in enumerate(zip(groups, groupnames, cmaps, axarr[1:])):
            mb.calc.cell_density(
                adata,
                groupby=f'is_{groupname}',
                groups='True',
                resolution=resolution,
                bandwidth=bandwidth
            )
            alpha=1.0
            cell_density(
                adata,
                key=f'is_{groupname}_density_True',
                cmap=cmaps[group_i],
                legend_loc=None,
                ax=ax,
                alpha=alpha,
                show=False
            )

        for ax, groupname in zip(axarr[1:], groupnames):
            ax.legend().set_visible(False)
            ax.set_title(
                f'{groupname} (density)'
            )

            if grid:
                x_lim = ax.get_xlim()
                y_lim = ax.get_ylim()

                x_grid_space = (x_lim[1] - x_lim[0])/(n_grids)
                y_grid_space = (y_lim[1] - y_lim[0])/(n_grids)
                ax.set_xticks(
                    np.arange(
                        x_lim[0],
                        x_lim[1], # + x_grid_space,
                        x_grid_space
                    )
                )
                ax.set_yticks(
                    np.arange(
                        y_lim[0],
                        y_lim[1], #+y_grid_space,
                        y_grid_space
                    )

                )
                ax.grid(
                    visible=True,
                    which='major',
                    linestyle='--',
                    lw=0.25,
                    color='black'
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    if delete_temp_columns:
        adata.drop(
            labels=[
                f'is_{groupname}_density_True'
                for groupname in groupnames
            ],
            axis=1
        )

    if show:
        plt.show()
    else:
        return fig, axarr


