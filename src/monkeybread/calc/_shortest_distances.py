from typing import List, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def shortest_distances(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    basis: Optional[str] = "spatial",
) -> np.ndarray:
    """Calculates the distance from each cell in one group to the nearest cell in another group.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `adata.obs` to classify groups.
    group1
        Either one group or a list of groups from `adata.obs[groupby]`.
    group2
        Either one group or a list of groups from `adata.obs[groupby]`.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    An array with length equal to group1. Each index contains a length-3 array, where index 0
    contains a cell index from group1, index 1 contains the distance to the nearest member of
    group2, and index 2 contains the index of the nearest member of group2.
    """
    # Convert groups to lists if single group provided
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Create adata views of cells from each group
    group1_cells = adata[[g in group1 for g in adata.obs[groupby]]]
    group2_cells = adata[[g in group2 for g in adata.obs[groupby]]]

    # Find nearest neighbor in group2 from group1
    nbrs = NearestNeighbors(n_neighbors=1).fit(group2_cells.obsm[f"X_{basis}"])
    distances, indices = nbrs.kneighbors(group1_cells.obsm[f"X_{basis}"])
    group2_indices = group2_cells.obs.index[indices.transpose()[0]].to_numpy()

    # Return array of lists mapping each group1 cell to its nearest group2 cell, including distance
    return np.array(list(zip(group1_cells.obs.index, distances.transpose()[0], group2_indices)))
