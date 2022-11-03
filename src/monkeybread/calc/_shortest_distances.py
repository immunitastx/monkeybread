from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
from typing import Union, List
import numpy as np


def shortest_distances(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
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

    Returns
    -------
    distances
        An array with length equal to group1. Each index contains a length-3 array, where index 0
        contains a cell index from group1, index 1 contains the distance to the nearest member of
        group2, and index 2 contains the index of the nearest member of group2.
    """
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]
    group1_cells = adata[[g in group1 for g in adata.obs[groupby]]].copy()
    group2_cells = adata[[g in group2 for g in adata.obs[groupby]]].copy()
    nbrs = NearestNeighbors(n_neighbors = 1).fit(group2_cells.obsm["X_spatial"])
    distances, indices = nbrs.kneighbors(group1_cells.obsm["X_spatial"])
    group2_indices = group2_cells.obs.index[indices.transpose()[0]].to_numpy()
    return np.array(list(
        zip(
            group1_cells.obs.index,
            distances.transpose()[0],
            group2_indices
        )
    ))
