from anndata import AnnData
from typing import Union, Set, Dict, Optional, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
import pandas as pd


def cell_contact_embedding(
    adata: AnnData,
    contacts: Dict[str, Set[str]],
    group: Optional[str] = None,
    basis: Optional[str] = "spatial",
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """
    Plots the results of :func:`monkeybread.calc.cell_contact`, showing spatial positions of cells
    in contact.

    Parameters
    ----------
    adata
        Annotated data matrix.
    contacts
        The actual cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    group
        Column in `adata.obs` to label cell contacts by.
    basis
        Grouping in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to :func:`scanpy.pl.embedding`.

    Returns
    -------
    ax
        If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
        within.
    """
    if ax is None:
        ax = plt.axes()
    cell_list = set(contacts.keys())
    for s in contacts.values():
        cell_list = cell_list.union(s)
    adata_contact = adata[list(cell_list)].copy()
    sc.pl.embedding(
        adata,
        basis = basis,
        na_color = "lightgrey",
        show = False,
        alpha = 0.5,
        ax = ax,
        size = 12000 / adata.shape[0],
        **kwargs
    )
    sc.pl.embedding(
        adata_contact,
        basis = basis,
        show = False,
        ax = ax,
        color = group,
        na_color = "red",
        size = (12000 / adata.shape[0]) * 5,
        **kwargs
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
    **kwargs
) -> Optional[plt.Axes]:
    """
    Plots the results of :func:`monkeybread.calc.cell_contact` and
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
    ax
        If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
        within.
    """
    if ax is None:
        ax = plt.axes()
    expected_contacts, p_val = expected_contacts

    g1_cats = set(adata[list(contacts.keys())].obs[groupby])
    g2_cats = set(adata[[v for vals in contacts.values() for v in vals]].obs[groupby])
    g1 = adata[[g in g1_cats for g in adata.obs[groupby]]].obs.index
    g2 = adata[[g in g2_cats for g in adata.obs[groupby]]].obs.index
    num_contacts = sum(len(v) for v in contacts.values()) - \
        int(0.5 * sum(0 if k not in g2 else sum(v in g1 for v in values)
            for k, values in contacts.items()))

    sns.histplot(expected_contacts, ax = ax, **kwargs)
    ax.axvline(num_contacts, 0, 1, color = "red", linestyle = '--')
    plt.text(0.98, 0.98, f"p = {p_val : .2f}",
             transform = ax.transAxes, va = "top", ha = "right")
    if show:
        plt.show()
    else:
        return ax


def cell_contact_heatmap(
    adata: AnnData,
    groupby: str,
    contacts: Dict[str, Set[str]],
    expected_contacts: Optional[pd.DataFrame] = None,
    count_multi: Optional[bool] = False,
    show: Optional[bool] = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """
    Plots the results of :func:`monkeybread.calc.cell_contact` and optionally
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
        The expected cell contacts, as calculated by :func:`monkeybread.stat.cell_contact`.
    count_multi
        Count each contact when making calculations. E.g., if a cell A is contacting two cells B and
        C, if both B and C are cell type X, count as 2 contacts instead of 1.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to `seaborn.heatmap`.

    Returns
    -------
    ax
        If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
        within.
    """
    if ax is None:
        ax = plt.axes()
    group1 = set(adata[list(contacts.keys())].obs[groupby])
    group2 = set(adata[[v for vals in contacts.values() for v in vals]].obs[groupby])
    contacting_counts = {
        g1: {g2: 0 for g2 in group2} for g1 in group1
    }
    for g1, g2s in contacts.items():
        g1_type = adata.obs[groupby][g1]
        g2_types = [adata.obs[groupby][t] for t in g2s]
        for g2_type in (g2_types if count_multi else set(g2_types)):
            contacting_counts[g1_type][g2_type] += 1
    contact_df = pd.DataFrame(contacting_counts)
    contact_df.fillna(0, inplace = True)

    if expected_contacts is not None:
        contact_df_normalized = expected_contacts.T
        contact_df_annot = contact_df_normalized
    else:
        contact_df_annot = contact_df.T
        contact_df_normalized = contact_df.T.apply(
            lambda arr: arr / (np.sum(arr) if np.sum(arr) > 0 else 1), axis = 1, raw = True
        )

    fmt = ".3f" if expected_contacts is not None else "d"
    if "fmt" in kwargs:
        fmt = kwargs["fmt"]
        del kwargs["fmt"]

    sns.heatmap(contact_df_normalized,
                ax = ax,
                cmap = f'plasma{"_r" if expected_contacts is not None else ""}',
                annot = contact_df_annot,
                fmt = fmt,
                **kwargs)
    if show:
        plt.show()
    else:
        return ax
