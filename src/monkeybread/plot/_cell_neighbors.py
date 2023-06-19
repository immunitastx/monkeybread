from typing import Dict, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
from anndata import AnnData

import monkeybread as mb


def cell_neighbor_embedding(
    adata: AnnData,
    cell_to_neighbors: Dict[str, Set[str]],
    group: Optional[str] = None,
    basis: Optional[str] = "X_spatial",
    dot_size_group1: Optional[float] = None,
    dot_size_group2: Optional[float] = None,
    dot_size_unselected: Optional[float] = None,
    cell_color_group1: Optional[str] = "red",
    cell_color_group2: Optional[str] = "blue",
    cell_color_unselected: Optional[str] = "lightgrey",
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Shows embeddings of cells with their neighbors.

    Plots the results of :func:`monkeybread.calc.cell_neighbors`, highlighting the cells and their neighbors.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        Cells and their associated neighbors as calculated by :func:`monkeybread.calc.cell_neighbors`.
    group
        Column in `adata.obs` for which to label cells.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `X_spatial`.
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

    # Subset adata to only include cells in first group
    cell_list_g1 = set(cell_to_neighbors.keys())
    adata_group1 = adata[list(cell_list_g1)]

    # Subset adata to only include cells in second group
    cell_list_g2 = set()
    for s in cell_to_neighbors.values():
        cell_list_g2 = cell_list_g2.union(s)
    adata_group2 = adata[list(cell_list_g2)]

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
        na_color=cell_color_group1,
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
        na_color=cell_color_group2,
        alpha=1.0,
        size=dot_size_group2,
        **kwargs,
    )
    
    if show:
        plt.show()
    else:
        return ax


def cell_contact_histplot(
    adata: AnnData,
    groupby: str,
    contacts: Dict[str, Set[str]],
    expected_contacts: Tuple[np.ndarray, float],
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Plots a histogram highlighting the observed cell contact compared to permutations.

    Uses the results of :func:`monkeybread.calc.cell_contact` and
    :func:`monkeybread.stat.cell_contact`.

    Creates a histogram displaying the distribution of contacts from the permutation test with a
    line indicating the actual distribution of contact counts.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A column in `adata.obs` to group cells by.
    contacts
        The actual cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    expected_contacts
        The expected cell contacts, as calculated by :func:`monkeybread.stat.cell_contact`.
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
    expected_contacts, p_val = expected_contacts

    # Pull out cells corresponding to the categories found in the contact dictionary
    g1_cats = set(adata[list(contacts.keys())].obs[groupby])
    g2_cats = set(adata[[v for vals in contacts.values() for v in vals]].obs[groupby])
    g1 = adata[[g in g1_cats for g in adata.obs[groupby]]].obs.index
    g2 = adata[[g in g2_cats for g in adata.obs[groupby]]].obs.index

    # Find observed contact count
    num_contacts = mb.util.contact_count(contacts, g1, g2)

    # Plot expected contact count distribution with vertical line for observed contacts and
    # annotated p-value
    sns.histplot(expected_contacts, ax=ax, **kwargs)
    observed_count = ax.axvline(num_contacts, 0, 1, color="red", linestyle="--")
    plt.text(0.98, 0.98, f"p = {p_val : .2f}", transform=ax.transAxes, va="top", ha="right")
    ax.legend(handles=[observed_count], labels=["Observed Count"], loc="center left", bbox_to_anchor=(1, 0.5))

    ax.set_ylabel("Permutation Count")
    ax.set_xlabel("Cell Contact Counts")
    ax.set_title("Permuted Distribution of Counts")

    if show:
        plt.show()
    else:
        return ax


def cell_contact_heatmap(
    adata: AnnData,
    groupby: str,
    contacts: Optional[Dict[str, Set[str]]] = None,
    expected_contacts: Optional[pd.DataFrame] = None,
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Optional[plt.Axes]:
    """Produces a heatmap highlighting cell contact between groups in a pairwise manner.

    Uses the results of :func:`monkeybread.calc.cell_contact` and optionally
    :func:`monkeybread.stat.cell_contact`.

    Produces a heatmap where rows correspond to `group1` and columns correspond to `group2`.
    An entry in the heatmap depicts either the raw contact frequencies or the p-values for those
    contact frequencies, depending on whether `expected_contacts` is provided. Annotations of
    cells are included by default.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A column in `adata.obs` to group cells by.
    contacts
        The actual cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    expected_contacts
        The expected cell contacts, as calculated by :func:`monkeybread.stat.cell_contact`. This
        must have been calculated using `split_groups = True`.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to `seaborn.heatmap`.

    Returns
    -------
    If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
    within.
    """
    # Create axes
    if ax is None:
        ax = plt.axes()

    # Create heatmap df and annotation df
    if expected_contacts is not None:
        # If we're plotting the expected contact significance, just use the p-values
        contact_df_normalized = expected_contacts.T
        contact_df_annot = contact_df_normalized
    elif contacts is not None:
        # Pull out group1 and group2 from the contact dictionary
        group1 = sorted(set(adata[list(contacts.keys())].obs[groupby]))
        group2 = sorted(set(adata[[v for vals in contacts.values() for v in vals]].obs[groupby]))

        # Count contacts for each pairwise group comparison and create dataframe
        contacting_counts = {
            g1: {
                g2: mb.util.contact_count(
                    contacts, adata[adata.obs[groupby] == g1].obs.index, adata[adata.obs[groupby] == g2].obs.index
                )
                for g2 in group2
            }
            for g1 in group1
        }
        contact_df = pd.DataFrame(contacting_counts)
        contact_df.fillna(0, inplace=True)

        # Generate annotations based on row proportions
        contact_df_annot = contact_df.T
        contact_df_normalized = contact_df.T.apply(
            lambda arr: arr / (np.sum(arr) if np.sum(arr) > 0 else 1), axis=1, raw=True
        )
    else:
        raise ValueError("One of 'contacts', 'expected_contacts' must be provided.")

    # Format annotations as float (expected) or integers (actual)
    fmt = ".3f" if expected_contacts is not None else "d"
    # If fmt is customized by user in kwargs, use that instead and remove it from kwargs
    if "fmt" in kwargs:
        fmt = kwargs["fmt"]
        del kwargs["fmt"]

    # Create heatmap
    sns.heatmap(
        contact_df_normalized,
        ax=ax,
        cmap=f'plasma{"_r" if expected_contacts is not None else ""}',
        annot=contact_df_annot,
        fmt=fmt,
        **kwargs,
    )

    ax.set_ylabel("Group 1")
    ax.set_xlabel("Group 2")
    ax.set_title("Observed Contacts" if expected_contacts is None else "Contact p-values")

    if show:
        plt.show()
    else:
        return ax
