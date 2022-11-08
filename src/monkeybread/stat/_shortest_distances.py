from anndata import AnnData
import random as rand
from typing import Union, List, Optional, Tuple
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import numpy as np


def shortest_distances(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    n_perms: Optional[int] = 100,
    actual: Optional[np.ndarray] = None,
    threshold: Optional[float] = None
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Calculates an expected distribution of shortest distances via permutation of labels excluding
    `group2`.

    Calculation is the same as in :func:`monkeybread.calc.shortest_distances`.

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

    Returns
    -------
    expected
        An expected distribution as described above.
    threshold
        Optionally, the threshold passed in corresponding to the p-value.
    p_val
        Optionally, a p-value as described above.
    """
    p_statistic = lambda x: np.count_nonzero(np.less(x, threshold))
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]
    group1_data = adata.obsm["X_spatial"][[g in group1 for g in adata.obs[groupby]]]
    filtered_data = adata[[g not in group1 for g in adata.obs[groupby]]]
    all_distances = []
    n_coords = sum([Counter(filtered_data.obs[groupby])[g] for g in group2])
    coords = list(filtered_data.obsm["X_spatial"])
    statistics = np.zeros(n_perms)
    for i in range(n_perms):
        random_coords = rand.sample(coords, k = n_coords)
        nbrs = NearestNeighbors(n_neighbors = 1).fit(random_coords)
        distances, indices = nbrs.kneighbors(group1_data)
        all_distances.extend(distances.transpose()[0])
        if threshold is not None:
            statistics[i] = p_statistic(distances.transpose()[0])
    all_distances = np.array(all_distances)
    if actual is None or threshold is None:
        return all_distances
    actual_statistic = p_statistic(np.vectorize(np.float)(actual))
    p_val = np.mean([1 if statistic > actual_statistic else 0 for statistic in statistics])
    return all_distances, threshold, p_val
