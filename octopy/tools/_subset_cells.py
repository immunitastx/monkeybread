from anndata import AnnData
from typing import Tuple, List, Optional, Union
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
    subset: Union[Tuple[str, str, float], List[Tuple[str, str, float]]],
    label_obs: Optional[str] = None,
    label: Optional[str] = None
) -> AnnData:
    """Subsets cells based on gene expression and optionally labels them.

    :param adata: Annotated data matrix
    :param subset: Either a single condition or a list of conditions. Each condition contains of a
    length-3 tuple where the first element is a column name, the second element is one of gt, gte,
    lt, lte, or eq, and the third element is a number.
    :param label_obs: A categorical column in `adata.obs` to add a label to if it passes the subset
    conditions. Creates the column if it does not exist, and sets other values to "Unknown". If the
    column exists, existing labels other than "Unknown" will take precedence.
    :param label: The label to assign to cells passing the subset conditions in
    `adata.obs[label_obs]`
    :return: A copy of `adata` containing only cells matching the subset conditions
    """

    if type(subset) is tuple:
        gene, relation, value = subset
        if relation not in cases:
            raise ValueError("Relation not one of 'gt', 'gte', 'lt', 'lte', 'eq'.")
        mask = [cases[relation](count, value) for count in adata.transpose()[gene].X[0]]
    elif type(subset) is list:
        mask = [True] * adata.shape[0]
        for (gene, relation, value) in subset:
            if relation not in cases:
                raise ValueError("Relation not one of 'gt', 'gte', 'lt', 'lte', 'eq'.")
            mask = [curr and cases[relation](count, value) for (curr, count) in
                    zip(mask, adata.transpose()[gene].X[0])]
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
