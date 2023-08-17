"""
Statistical tests for cellular co-localization.
"""
import random as rand
from collections import Counter
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def shortest_distances(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    n_perms: Optional[int] = 100,
    observed: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    basis: Optional[str] = "X_spatial",
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Calculates an expected null distribution of shortest distances from cells in `group1` to 
    cells in `group2` by permuting coordinates of cells that aren't in group1. The null hypothesis
    assumes no coherent spatial co-localization between cells of `group1` and cells of `group2`.
    If `group1` is a list, then this function will consider the union of all cells with a label
    in `group1`. Likewise if `group2` is a list then this function will consider the union of
    all cells with a label in `group2`.

    If `observed` and `threshold` are provided, a p-value is computed on the following test
    statistic: the number of shortest distances from cells in `group1` to cells in `group2`
    that are less than `threshold`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `adata.obs` to use for grouping.
    group1
        One or more levels from `adata.obs[groupby]` to use as sources for shortest distance.
    group2
        One or more levels from `adata.obs[groupby]` to use as destinations for shortest distance.
    n_perms
        The number of permutations to run. Defaults to 100.
    observed
        The observed distribution of shortest distances, as calculated by
        :func:`monkeybread.calc.shortest_distances`.
    threshold
        A distance threshold to use for significance calculation, in coordinate units.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    If `threshold` is not provided, an array containing the expected null distribution as described
    above. If `threshold` is provided, a length-2 tuple will be returned, where the first element
    is the array containing the expected null distribution. The second element corresponds to the
    the p-value calculated.
    """
    # Calculates the number of distances within the threshold
    calc_test_statistic = lambda x: np.count_nonzero(np.less_equal(x, threshold))

    # Converts groups to lists if one group provided
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Pulls out group1 spatial locations, group2 spatial locations, and number of cells in group2
    # The reason we need this information is because of the way we implement this statistical test.
    # Specifically, instead of actually shuffling all labels, recomputing the test statistic each time,
    # we instead randomly sample coordinates from non-group1 cells and compute the shortest
    # distances from group1 cells to these sampled coordinates. This approach is equivalent to "true
    # shuffling" approach
    group1_coords = adata.obsm[basis][[g in group1 for g in adata.obs[groupby]]]
    non_group1_ad = adata[[g not in group1 for g in adata.obs[groupby]]]
    group_to_counts = Counter(non_group1_ad.obs[groupby])
    n_coords_in_group2 = sum([group_to_counts[g] for g in group2])
    non_group1_coords = list(non_group1_ad.obsm[basis])

    # Create both distances and test statistics arrays, number of cells in group2
    all_distances = []
    statistics_under_perm = np.zeros(n_perms)

    for i in range(n_perms):
        # For each permutation, randomly sample from the non-group1 coordinates and run
        # shortest distances calculation
        random_coords = rand.sample(non_group1_coords, k=n_coords_in_group2)
        nbrs = NearestNeighbors(n_neighbors=1).fit(random_coords)
        distances, indices = nbrs.kneighbors(group1_coords)

        # Add distances to array
        all_distances.extend(distances.transpose()[0])
        if threshold is not None:
            # Add number of distances within threshold to statistics array
            statistics_under_perm[i] = calc_test_statistic(distances.transpose()[0])

    all_distances = np.array(all_distances)

    if observed is None or threshold is None:
        # If observed and thresholds aren't provided, don't do p-value calculation
        return all_distances

    # Calculate number of distances under threshold
    observed_statistic = calc_test_statistic(observed["distance"])

    # Calculate p value by comparing the observed statistic to each of the permutations
    # Add pseudocount to permutation
    p_val = (np.sum(np.where(statistics_under_perm >= observed_statistic, 1, 0)) + 1) / (n_perms + 1)

    return all_distances, p_val


def shortest_distances_pairwise(
    adata: AnnData,
    groupby: str,
    group1: List[str],
    group2: List[str],
    n_perms: Optional[int] = 100,
    observed: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    threshold: Optional[float] = None,
    basis: Optional[str] = "spatial",
) -> Union[Dict[str, Dict[str, np.ndarray]], Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]]:
    """Performs the same statistical analyses implemented by :func:`monkeybread.stat.shortest_distances`,
    but in a pairwise fashion between every cell type in `group1` and every cell type in `group2`. This
    stands in contrast to :func:`monkeybread.stat.shortest_distances` that considered the union of cells
    with cell type labels in `group1` as the first group and union of cell type labels in `group2` as the 
    second group.

    For every pair of cell types in `group1` and `group2`, this function calculates an expected 
    null distribution of shortest distances from cells in the `group1` cell type to cells in the 
    `group2` cell type by permuting coordinates of cells that aren't in the `group1` cell type. The null 
    hypothesis assumes no coherent spatial co-localization between cells of the `group1` cell type and 
    cells of the `group2` cell type.

    If `observed` and `threshold` are provided, p-values are computed on the following test
    statistics: the number of shortest distances from cells in each `group1` cell type to 
    cells in each `group2` cell type that are less than `threshold`.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `adata.obs` to use for grouping.
    group1
        One or more levels from `adata.obs[groupby]` to use as sources for shortest distance.
    group2
        One or more levels from `adata.obs[groupby]` to use as destinations for shortest distance.
    n_perms
        The number of permutations to run. Defaults to 100.
    observed
        The observed distribution of shortest distances, as calculated by
        :func:`monkeybread.calc.shortest_distances`.
    threshold
        A distance threshold to use for significance calculation, in coordinate units.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    If `threshold` is not provided, an dictonary of dictonaries is returned that maps each cell type 
    of `group1` to a cell type of `group2` to a list containing the expected null distribution as 
    described above. If `threshold` is provided, a length-3 tuple will be returned, where the first element
    is the dictionary mapping to the null distributions. The second is a dictionary mapping to the
    the p-values.
    """
    g1_to_g2_to_dists = {}
    g1_to_g2_to_pvals = {}
    for g1 in group1:
        # Instantiate dictionaries
        g1_to_g2_to_dists[g1] = {}
        g1_to_g2_to_pvals[g1] = {}
        for g2 in group2:
            if observed and threshold: # If we are computing p-values
                obs = observed[g1][g2]
                dists, p_val = shortest_distances(
                    adata, 
                    groupby, 
                    g1, 
                    g2, 
                    threshold=threshold, 
                    observed=obs, 
                    n_perms=n_perms
                )
                g1_to_g2_to_dists[g1][g2] = dists
                g1_to_g2_to_pvals[g1][g2] = p_val
            else:
                dist = shortest_distances(
                    adata,
                    groupby,
                    g1,
                    g2,
                    n_perms=n_perms
                )
                g1_to_g2_to_dists[g1][g2] = dists

    if observed and threshold:
        return g1_to_g2_to_dists, g1_to_g2_to_pvals
    else:
        return g1_to_g2_to_dists 




