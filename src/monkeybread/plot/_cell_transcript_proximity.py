from typing import List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData


def cell_transcript_proximity(
    adata: AnnData,
    cells: List[str],
    transcripts: Optional[pd.DataFrame] = None,
    label: Optional[str] = None,
    legend: Optional[bool] = False,
    pairwise: Optional[bool] = False,
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[Union[plt.Axes, plt.Figure]]:
    """Plots cell boundaries and optionally transcripts in the surrounding area.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cells
        A list of cell ids to plot.
    transcripts
        Transcripts to plot, as calculated by :func:`monkeybread.calc.cell_transcript_proximity`.
    label
        A column in `adata` to use to color cells.
    legend
        Whether to include cell/transcript labels in a legend.
    pairwise
        Whether to draw a matrix of subplots [i, j] where the plot at [i, j] contains only the genes
        demarcated by the row and column. Allows for more fine-tuned observation of specific genes.
    show
        Whether to show the plot or return the Axes object.
    ax
        An axis object to use, only used if `pairwise = False`.
    kwargs
        Keyword arguments passed to :func:`seaborn.scatterplot` for transcripts.

    Returns
    -------
    A matplotlib Axes containing the plot if `pairwise = False`, otherwise a matplotlib Figure
    containing all of the subplots. Only returned if `show = False`.
    """
    if pairwise:
        if transcripts is None:
            raise ValueError(
                "When plotting pairwise by gene, you must provide the transcript \
                            results from mb.calc.cell_transcript_proximity."
            )

        # Pull out unique genes from dataframe and create n x n subplots
        genes = set(transcripts["gene"])
        fig, axs = plt.subplots(nrows=len(genes), ncols=len(genes), figsize=(12, 8))
        for i, marker1 in enumerate(genes):
            for j, marker2 in enumerate(genes):
                # Call function recursively for each pairwise comparison
                cell_transcript_proximity(
                    adata,
                    cells,
                    transcripts.loc[transcripts["gene"].isin({marker1, marker2}), :],
                    legend=False,
                    pairwise=False,
                    ax=axs[i, j],
                    s=10,
                    label=label,
                    hue_order=[marker1, marker2],
                    show=False,
                    **kwargs,
                )

                # Set titles/labels on specific axes
                axs[i, j].set_title(marker2 if i == 0 else None)
                axs[i, j].set_ylabel(marker1 if j == 0 else None)
                axs[i, j].set_xlabel(None)
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])

        if label is not None:
            # Add legend for cells
            legend_ax = axs[int((len(genes) - 1) / 2), len(genes) - 1]
            handles, labels = legend_ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            legend_ax.legend(by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1, 0.5))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        if show:
            plt.show()
        else:
            return fig
    else:
        # Pull out cell boundaries and plot according to label
        cell_bounds = adata[cells].obs["bounds"]
        ax = plt.axes() if ax is None else ax
        cell_labels = [None] * len(cell_bounds) if label is None else adata.obs[label][cells]
        color_options = list(plt.get_cmap("tab20").colors)
        label_codes = pd.Categorical(cell_labels).codes
        lines = [  # noqa: F841
            ax.plot(bounds.T[0], bounds.T[1], color=color_options[label_codes[i]], label=cell_labels[i])
            for i, bounds in enumerate(cell_bounds)
        ]

        # Plot transcripts if provided
        if transcripts is not None:
            sns.scatterplot(
                x=transcripts["global_x"],
                y=transcripts["global_y"],
                hue=transcripts["gene"],
                ax=ax,
                legend="auto" if legend else None,
                **kwargs,
            )
        ax.set_title(f"Cell Boundaries{' and Transcripts' if transcripts is not None else ''}")
        ax.set_ylabel("spatial_y")
        ax.set_xlabel("spatial_x")

        # Add legend for cells + transcripts
        if legend:
            ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if show:
            plt.show()
        else:
            return ax
