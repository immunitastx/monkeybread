from anndata import AnnData
from typing import Tuple, List, Optional, Union, Literal
import numpy as np
import pandas as pd

cases = {
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "eq": lambda a, b: a == b
}


def subset_cells(
    adata: AnnData,
    by: Union[Literal["gene"], Literal["spatial"]],
    subset: Union[Tuple[str, str, float], List[Tuple[str, str, float]]],
    label_obs: Optional[str] = None,
    label: Optional[str] = None
) -> AnnData:
    """Subsets cells based on gene expression and optionally labels them.

    Parameters
    ----------
    adata
        Annotated data matrix.
    by
        Either `'gene'` or `'spatial'`. Indicates whether `subset` refers to gene counts in each
        cell or x and y locations of each cell.
    subset
        Either a single condition or a list of conditions. Each condition consists of a length-3
        tuple where the first element is either a gene or x/y, the second element is one of gt, gte,
        lt, lte, or eq, and the third element is a number.
    label_obs
        A categorical column in `adata.obs` to add a label to if it passes the subset conditions.
        Creates the column if it does not exist, and sets other values to "Unknown". If the
        column exists, existing labels other than "Unknown" will take precedence.
    label
        The label to assign to cells passing the subset conditions in `adata.obs[label_obs]`.

    Returns
    -------
    adata_subset
        A copy of `adata` containing only cells matching the subset conditions.
    """
    if by != "spatial" and by != "gene":
        raise ValueError(f"Argument `by` must be one of 'gene' or 'spatial'. Value provided: {by}")
    if type(subset) is tuple:
        obs, relation, value = subset
        if relation not in cases:
            raise ValueError("Relation not one of 'gt', 'gte', 'lt', 'lte', 'eq'.")
        if by == "spatial":
            mask = [cases[relation](count, value) for count in
                    adata.obsm["X_spatial"].transpose()[1 if obs == "y" else 0]]
        elif by == "gene":
            mask = [cases[relation](count, value) for count in adata.transpose()[obs].X[0]]
    elif type(subset) is list:
        mask = np.full(adata.shape[0], fill_value = True)
        for (obs, relation, value) in subset:
            if relation not in cases:
                raise ValueError("Relation not one of 'gt', 'gte', 'lt', 'lte', 'eq'.")
            if by == "spatial":
                mask = np.logical_and(
                    mask,
                    [cases[relation](count, value) for count in
                     adata.obsm["X_spatial"].transpose()[1 if obs == "y" else 0]]
                )
            elif by == "gene":
                mask = np.logical_and(
                    mask,
                    [cases[relation](count, value) for count in adata.transpose()[obs].X[0]]
                )
    else:
        raise TypeError("Positional argument `subset` must be a tuple or list.")
    if label_obs is not None and label is not None:
        if label_obs not in adata.obs:
            adata.obs[label_obs] = np.full(adata.shape[0], fill_value = "Unknown")
        adata.obs[label_obs] = pd.Categorical(
            [label if in_mask and existing == "Unknown" else existing for (in_mask, existing) in
             zip(mask, adata.obs[label_obs])]
        )
    return adata[mask].copy()
