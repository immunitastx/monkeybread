from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData

import monkeybread as mb


def neighborhood_profile(
    adata: AnnData,
    groupby: str,
    basis: Optional[str] = "spatial",
    neighborhood_groups: Optional[Sequence[str]] = None,
    subset_groups: Optional[Sequence[str]] = None,
    radius: Optional[float] = 50,
    normalize_counts: Optional[bool] = True,
) -> AnnData:
    """Calculates a neighborhood profile for each cell.

    The resulting AnnData object will have the same index corresponding to rows/cells, but a new
    index corresponding to columns, one column for each category in `adata.obs[groupby]`. Instead of
    a gene expression profile, each column corresponds to the proportion of cells in the surrounding
    radius that belong to the respective category.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `obs` to use for neighborhood profile calculations.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    neighborhood_groups
        A list of groups from `adata.obs[groupby]` to include in the resulting `adata.var_names`.
        Will not affect the calculations themselves, only which results are provided.
    subset_groups
        A list of groups in `adata.obs[groupby]` to restrict the resulting AnnData object to. Only
        cells in those groups will be included in the resulting `adata.obs_names`.
    radius
        Radius in coordinate space to include nearby cells for neighborhood profile calculation.
    normalize_counts
        Normalize neighborhood counts to proportions instead of raw counts. Note, if
        `neighborhood_groups` is provided and `normalize_counts = True`, the normalization step will
        be performed before removing groups, so proportions will not sum to 1.

    Returns
    -------
    A new AnnData object where columns now correspond to neighborhood profile proportions. All .obs
    columns will be carried over from the provided AnnData object, and they will share an index.
    """
    if adata.obs[groupby].dtype != "category":
        raise ValueError(f"adata.obs['{groupby}'] is not categorical.")
    categories = adata.obs[groupby].cat.categories
    # Calculate all contacts
    contacts = mb.calc.cell_contact(adata, groupby, categories, categories, basis=basis, radius=radius)
    group_mapping = adata.obs[groupby]
    mask = [True] * adata.shape[0]
    if subset_groups is not None:
        mask = [g in subset_groups for g in adata.obs[groupby]]
    # Sum the number of contacts for each cell
    n_neighbors = np.array([len(contacts[cell]) if cell in contacts else 0 for cell in adata[mask].obs.index])
    # Count the number of each group in the neighborhood of each cell
    counters = {
        cell: Counter(group_mapping[c] for c in contacts[cell]) if cell in contacts else {}
        for cell in adata[mask].obs.index
    }
    # Turn counts into fractions for each cell summing to 1
    adata_neighbors = pd.DataFrame(counters).T.fillna(0)
    if normalize_counts:
        adata_neighbors = adata_neighbors.apply(func=lambda arr: arr / np.sum(arr), axis=1).fillna(0)
    # Create a new AnnData object with cells as rows and group names as columns
    neighbor_adata = AnnData(
        X=adata_neighbors,
        obs=pd.DataFrame({"n_neighbors": n_neighbors, groupby: adata[mask].obs[groupby]}, index=adata[mask].obs.index),
        uns={"neighbor_radius": radius},
        obsm={f"X_{basis}": adata[mask].obsm[f"X_{basis}"].copy()},
        dtype=adata_neighbors.to_numpy().dtype,
    )
    if neighborhood_groups is not None:
        return neighbor_adata[:, neighborhood_groups].copy()
    else:
        return neighbor_adata
