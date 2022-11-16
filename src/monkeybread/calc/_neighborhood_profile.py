from anndata import AnnData
from typing import Optional
import monkeybread as mb
import numpy as np
import pandas as pd
from collections import Counter


def neighborhood_profile(
    adata: AnnData,
    groupby: str,
    basis: Optional[str] = "spatial",
    radius: Optional[float] = 50
) -> AnnData:
    """Calculates a neighborhood profile for each cell. The resulting AnnData object will have the
    same index corresponding to rows/cells, but a new index corresponding to columns, one column for
    each category in `adata.obs[groupby]`. Instead of a gene expression profile, each column
    corresponds to the proportion of cells in the surrounding radius that belong to the respective
    category.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `obs` to use for neighborhood profile calculations.
    basis
        A key in `adata.obsm` to use for cell coordinates.
    radius
        Radius in coordinate space to include nearby cells for neighborhood profile calculation.

    Returns
    -------
    A new AnnData object where columns now correspond to neighborhood profile proportions. All .obs
    columns will be carried over from the provided AnnData object, and they will share an index.
    """
    categories = adata.obs[groupby].cat.categories
    # Calculate all contacts
    contacts = mb.calc.cell_contact(adata, groupby, categories, categories,
                                    basis = basis, radius = radius)
    group_mapping = adata.obs[groupby]
    # Sum the number of contacts for each cell
    n_neighbors = np.array([len(contacts[cell]) if cell in contacts else 0 for cell in adata.obs.index])
    # Count the number of each group in the neighborhood of each cell
    counters = {
        cell: Counter(group_mapping[c] for c in contacts[cell]) if cell in contacts else dict()
        for cell in adata.obs.index
    }
    # Turn counts into fractions for each cell summing to 1
    adata_neighbors = pd.DataFrame(counters).T.fillna(0)\
        .apply(func = lambda arr: arr / np.sum(arr), axis = 1).fillna(0)
    # Create a new AnnData object with cells as rows and group names as columns
    return AnnData(
        X = adata_neighbors,
        obs = pd.DataFrame({"n_neighbors": n_neighbors, **adata.obs}, index = adata.obs.index),
        uns = {"neighbor_radius": radius}
    )
