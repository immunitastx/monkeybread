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
    cell_to_neighbors = mb.calc.cell_contact(adata, groupby, categories, categories, basis=basis, radius=radius)
    cell_to_group = adata.obs[groupby]
    if subset_groups is not None:
        mask = [g in subset_groups for g in adata.obs[groupby]]
    else:
        mask = [True] * adata.shape[0]

    # Count the number of each group in the neighborhood of each cell
    cell_to_neighbor_counts = {
        cell: Counter(cell_to_group[c] for c in cell_to_neighbors[cell]) if cell in cell_to_neighbors else {}
        for cell in adata[mask].obs.index
    }

    # Turn counts into fractions for each cell summing to 1
    neighbors_df = pd.DataFrame(cell_to_neighbor_counts).T.fillna(0)
    n_neighbors = neighbors_df.sum(axis=1)
    if normalize_counts:
        neighbors_df = neighbors_df.apply(func=lambda arr: arr / np.sum(arr), axis=1).fillna(0)

    # Create a new AnnData object with cells as rows and group names as columns
    neighbor_adata = AnnData(
        X=neighbors_df,
        obs=pd.DataFrame(
            {"n_neighbors": n_neighbors, groupby: adata[neighbors_df.index].obs[groupby]}, index=neighbors_df.index
        ),
        uns={"neighbor_radius": radius},
        obsm={f"X_{basis}": adata[neighbors_df.index].obsm[f"X_{basis}"].copy()},
        dtype=neighbors_df.to_numpy().dtype,
    )
    if neighborhood_groups is not None:
        return neighbor_adata[:, neighborhood_groups].copy()
    else:
        return neighbor_adata
