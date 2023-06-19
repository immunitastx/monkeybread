from typing import List, Optional, Union, Dict

import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def shortest_distances(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    basis: Optional[str] = "X_spatial",
) -> pd.DataFrame:
    """Calculates the distance from each cell in one group to the nearest cell in another group.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `adata.obs` to classify groups.
    group1
        Either one value or a list of values from `adata.obs[groupby]`.
    group2
        Either one value or a list of values from `adata.obs[groupby]`.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    A dataframe, indexed by cells in `group1`. The dataframe has two columns, `distance` and
    `nearest_cell`, corresponding to the distance to the nearest cell in `group2` and the index of
    that cell respectively.
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
    nbrs = NearestNeighbors(n_neighbors=1).fit(group2_cells.obsm[f"{basis}"])
    distances, indices = nbrs.kneighbors(group1_cells.obsm[f"{basis}"])
    group2_indices = group2_cells.obs.index[indices.transpose()[0]].to_numpy()

    # Return array of lists mapping each group1 cell to its nearest group2 cell, including distance
    return pd.DataFrame(
        {
            "distance": distances.transpose()[0],
            "nearest_cell": group2_indices,
        },
        index=group1_cells.obs.index,
    )

def shortest_distances_pairwise(
    adata: AnnData,
    groupby: str,
    group1: List[str],
    group2: List[str],
    basis: Optional[str] = "X_spatial",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Calculates the distance from each cell in one group to the nearest cell in another group.
    Performs this calculation between all pairs of cell types in group1 and group2. 

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `adata.obs` to classify groups.
    group1
        A list of values from `adata.obs[groupby]`.
    group2
        A list of values from `adata.obs[groupby]`.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    A dictionary mapping each value in `group1` to a dictionary that maps each value in 
    `group2` to a dataframe. This dataframe is indexed by by cells in the `group1` value. 
    The dataframe has two columns, `distance` and `nearest_cell`, corresponding to the distance 
    to the nearest cell in the given `group2` cell type and the index of that cell respectively.
    """
    g1_to_g2_to_df = {}
    for g1 in group1:
        g1_to_g2_to_df[g1] = {}
        for g2 in group2:
            dists_df = shortest_distances(
                adata, 
                groupby, g1, g2
            )
            g1_to_g2_to_df[g1][g2] = dists_df
    return g1_to_g2_to_df        


