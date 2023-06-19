from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

import monkeybread as mb


def neighborhood_profile(
    adata: AnnData,
    groupby: str,
    basis: Optional[str] = "X_spatial",
    subset_output_features: Optional[Sequence[str]] = None,
    subset_cells_by_group: Optional[Sequence[str]] = None,
    radius: Optional[float] = None,
    n_neighbors: Optional[int] = None,
    normalize_counts: Optional[bool] = True,
    standard_scale: Optional[bool] = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None
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
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.
    subset_output_features
        A list of groups from `adata.obs[groupby]` to include in the resulting `adata.var_names`.
        Will not affect the calculations themselves, only which results are provided.
    subset_cells_by_group
        A list of groups in `adata.obs[groupby]` to restrict the resulting AnnData object to. Only
        cells in those groups will be included in the resulting `adata.obs_names`.
    radius
        Radius in coordinate space to include nearby cells for neighborhood profile calculation.
    n_neighbors
        Number of neighbors to consider for the neighborhood profile calculations.
    normalize_counts
        Normalize neighborhood counts to proportions instead of raw counts. Note, if
        `subset_output_features` is provided and `normalize_counts = True`, the normalization step will
        be performed before removing groups, so proportions will not sum to 1.
    standard_scale
        Compute z-score of neighborhood values by subtracting the mean and dividing by
        the standard deviation.
    clip_min
        Clip values that are less than `clip_min` to `clip_min`
    clip_max
        Clip values that are greater than `clip_max` to `clip_max`

    Returns
    -------
    A new AnnData object where columns now correspond to neighborhood profile proportions. All .obs
    columns will be carried over from the provided AnnData object, and they will share an index.
    """
    if adata.obs[groupby].dtype != "category":
        raise ValueError(f"adata.obs['{groupby}'] is not categorical.")
    categories = adata.obs[groupby].cat.categories

    if radius is not None and n_neighbors is not None:
        raise ValueError("Cannot specify both radius and n_neighbors")

    # Calculate cell -> neighbors dictionary
    if radius is not None:
        # Use radius to define neighborhood
        cell_to_neighbors = mb.calc.cell_neighbors(
            adata, 
            groupby, 
            categories, 
            categories, 
            basis=basis, 
            radius=radius
        )
    elif n_neighbors is not None:
        # Use set number of neighbors to define neighborhood
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[basis])
        distances, indices = nbrs.kneighbors(adata.obsm[basis])
        # Convert to dictionary, remove self-inclusion
        cell_to_neighbors = {
            cell: set(adata[neighbors].obs.index).difference({cell})
            for cell, neighbors in zip(adata.obs.index, indices)
        }
    else:
        raise ValueError("Must specify either radius or n_neighbors")

    # Get group for each cell
    cell_to_group = adata.obs[groupby]

    # Remove certain cells from neighborhood calculations
    if subset_cells_by_group is not None:
        mask = [g in subset_cells_by_group for g in adata.obs[groupby]]
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
        uns={"neighbor_radius": radius} if radius is not None else {},
        obsm={basis: adata[neighbors_df.index].obsm[basis].copy()},
        dtype=neighbors_df.to_numpy().dtype,
    )

    # Standard scale
    if standard_scale:
        sc.pp.scale(
            neighbor_adata,  
            zero_center=True
        )

    # Clip max and min
    if clip_min and clip_max:
        neighbor_adata.X = np.clip(
            neighbor_adata.X, 
            a_min=clip_min, 
            a_max=clip_max
        )


    if subset_output_features is not None:
        return neighbor_adata[:, subset_output_features].copy()
    else:
        return neighbor_adata
