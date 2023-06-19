import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import Optional

def number_of_neighbors(
    neighbor_counts: pd.DataFrame,
    plot: Optional[str] = 'box',
    stripplot: Optional[bool] = False,
    box_kwargs: Optional = None,
    strip_kwargs: Optional = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    show: Optional[bool] = True
):
    if ax is None:
        ax = plt.axes()

    if box_kwargs is None:
        box_kwargs = {}
    if strip_kwargs is None:
        strip_kwargs = {}

    if plot == 'box':
        sns.boxplot(
            neighbor_counts, 
            x='group', 
            y='num_neighbors', 
            ax=ax, 
            **box_kwargs
         )
    elif plot == 'violin':
        sns.violinplot(
            neighbor_counts, 
            x='group', 
            y='num_neighbors', 
            ax=ax, 
            **box_kwargs
        )

    if stripplot:
        sns.stripplot(
            neighbor_counts, 
            x='group', 
            y='num_neighbors', 
            ax=ax, 
            c='black', 
            **strip_kwargs
         )

    if ylabel is None:
        ax.set_ylabel('Number of neighbors')
    else:
        ax.set_ylabel(ylabel)

    if xlabel is None:
        ax.set_xlabel('Group')
    else:
        ax.set_xlabel(xlabel)

