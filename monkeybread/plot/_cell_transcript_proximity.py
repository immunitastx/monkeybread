from anndata import AnnData
from typing import Optional, Union, List
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def cell_transcript_proximity(
    adata: AnnData,
    cells: List[str],
    transcripts: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    label_transcripts: Optional[bool] = False,
    pairwise: Optional[bool] = False,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Union[plt.Axes, plt.Figure]:
    """Plots cell boundaries and optionally transcripts in the surrounding area.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cells
        A list of cell ids to plot.
    transcripts
        Transcripts to plot, as calculated by :func:`~monkeybread.calc.cell_transcript_proximity`.
    label
        A column in `adata` to use to color cells.
    label_transcripts
        Whether to label each transcript by gene (in legend form).
    pairwise
        Whether to draw a matrix of subplots [i, j] where the plot at [i, j] contains only the genes
        demarcated by the row and column. Allows for more fine-tuned observation of specific genes.
    ax
        An axis object to use, only used if `pairwise = False`.
    kwargs
        Keyword arguments passed to :func:`~seaborn.scatterplot` for transcripts.

    Returns
    -------
    ax
        A matplotlib Axes containing the plot if `pairwise = False`, otherwise a matplotlib Figure
        containing all of the subplots.
    """
    genes = pd.Categorical(transcripts['gene']).categories
    if pairwise:
        fig, axs = plt.subplots(nrows = len(genes), ncols = len(genes),
                                figsize = (12, 8))
        for i, marker1 in enumerate(genes):
            for j, marker2 in enumerate(genes):
                cell_transcript_proximity(
                    adata,
                    cells,
                    transcripts.loc[transcripts['gene'].isin({marker1, marker2}), :],
                    label_transcripts = False,
                    pairwise = False,
                    ax = axs[i, j],
                    legend = None,
                    s = 10,
                    label = label,
                    hue_order = [marker1, marker2]
                )
                axs[i, j].set_title(marker2 if i == 0 else None)
                axs[i, j].set_ylabel(marker1 if j == 0 else None)
                axs[i, j].set_xlabel(None)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
        if label is not None:
            legend_ax = axs[int((len(genes) - 1) / 2), len(genes) - 1]
            handles, labels = legend_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_ax.legend(by_label.values(), by_label.keys(),
                             loc = 'center left', bbox_to_anchor = (1, 0.5))
        plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
        return fig
    else:
        cell_bounds = adata.obs["bounds"]
        ax = plt.axes() if ax is None else ax
        cell_labels = [None] * len(cell_bounds) if label is None else adata.obs[label][cells]
        color_options = list(plt.get_cmap('tab20').colors)
        label_codes = pd.Categorical(cell_labels).codes
        _lines = [
            ax.plot(
                bounds[0],
                bounds[1],
                color = color_options[label_codes[i]],
                label = cell_labels[i]
            ) for i, bounds in enumerate(cell_bounds)]
        sns.scatterplot(
            x = transcripts['x'],
            y = transcripts['y'],
            hue = transcripts['gene'],
            ax = ax,
            **kwargs
        )
        if label_transcripts:
            ax.legend(loc = 'center left', bbox_to_anchor = (1, 0.5))
        return ax
