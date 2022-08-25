from anndata import AnnData
from typing import Union, Set, Dict, Optional, Tuple
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc


def cell_contact(
    adata: AnnData,
    contacts: Dict[str, Set[str]],
    expected_contacts: Optional[Union[np.ndarray, Tuple[np.ndarray, float]]] = None,
    show: Optional[bool] = False,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Optional[plt.Axes]:
    """Plots the results of :func:`~monkeybread.calc.cell_contact`.

    Two different plots are produced depending on input. If `expected_contacts` is omitted, the
    output plot will be an embedding showing the spatial positions of the cells in contact. If
    `expected_contacts` contains the result of `monkeybread.stat.cell_contact`, the output will be a
    histogram displaying the distribution of contacts from the permutation test with a line showing
    the actual distribution of contact counts.

    Parameters
    ----------
    adata
        Annotated data matrix.
    contacts
        The actual cell contacts, as calculated by `monkeybread.calc.cell_contact`.
    expected_contacts
        The expected cell contacts, as calculated by `monkeybread.stat.cell_contact`.
    show
        Whether to show the plot or return the Axes object.
    ax
        An Axes object to add the plots to.
    kwargs
        Keyword arguments that will be passed to `scanpy.pl.embedding` or `seaborn.histplot` in
        accordance with the plot being produced.

    Returns
    -------
    ax
        If `show = True`, returns nothing. Otherwise, returns the Axes object the plot is contained
        within.
    """
    if ax is None:
        ax = plt.axes()
    if expected_contacts is None:
        cell_list = list(contacts.keys())
        for s in contacts.values():
            cell_list.extend(s)
        adata_contact = adata[cell_list].copy()
        sc.pl.embedding(
            adata,
            basis = "spatial",
            na_color = "lightgrey",
            show = False,
            alpha = 0.5,
            ax = ax,
            **kwargs
        )
        sc.pl.embedding(
            adata_contact,
            basis = "spatial",
            show = False,
            ax = ax,
            na_color = "red",
            **kwargs
        )
    else:
        p_val = None
        if type(expected_contacts) == tuple:
            expected_contacts, p_val = expected_contacts
        num_contacts = sum([len(v) for v in contacts.values()])
        sns.histplot(expected_contacts, ax = ax, **kwargs)
        ax.axvline(num_contacts, 0, 1, color = "red", linestyle = '--')
        if p_val is not None:
            plt.text(0.98, 0.98, f"p = {p_val : .2f}",
                     transform = ax.transAxes, va = "top", ha = "right")
    if show:
        plt.show()
    else:
        return ax
