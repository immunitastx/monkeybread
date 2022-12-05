import random as rand
from collections import Counter
from typing import List, Optional, Tuple, Union

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
    actual: Optional[pd.DataFrame] = None,
    threshold: Optional[float] = None,
    basis: Optional[str] = "spatial",
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Calculates an expected distribution of shortest distances via permutation of labels.

    Calculation is the same as in :func:`monkeybread.calc.shortest_distances`. Label permutation
    excludes `group1`.

    If `actual` and `threshold` are provided, a p-value relating the proportion of distances under
    `threshold` in the actual data compared to the expected data is produced.

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
    actual
        The actual distribution of shortest distances, as calculated by
        :func:`monkeybread.calc.shortest_distances`.
    threshold
        A distance threshold to use for significance calculation, in coordinate units.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    If `threshold` is not provided, an array containing the expected distribution as described
    above. If `threshold` is provided, a length-3 tuple will be returned, where the first element
    is the array containing the expected distribution. The second element corresponds to the
    threshold, and the third element is the p-value calculated.
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
    group1_coords = adata.obsm[f"X_{basis}"][[g in group1 for g in adata.obs[groupby]]]
    non_group1_ad = adata[[g not in group1 for g in adata.obs[groupby]]]
    group_to_counts = Counter(non_group1_ad.obs[groupby])
    n_coords_in_group2 = sum([group_to_counts[g] for g in group2])
    non_group1_coords = list(non_group1_ad.obsm[f"X_{basis}"])

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

    if actual is None or threshold is None:
        # If observed and thresholds aren't provided, don't do p-value calculation
        return all_distances

    # Calculate number of distances under threshold
    actual_statistic = calc_test_statistic(actual["distance"])

    # Calculate p value by comparing actual statistic to each of the permutations
    # Add pseudocount to permutation
    p_val = (np.sum(np.where(statistics_under_perm >= actual_statistic, 1, 0)) + 1) / (n_perms + 1)

    return all_distances, threshold, p_val
