import itertools
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

import monkeybread as mb


def cell_neighbors(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    cell_to_neighbors_obs: Dict[str, Set[str]],
    neighbor_radius: Optional[float] = None,
    perm_radius: Optional[float] = 100,
    n_perms: Optional[int] = 1000,
    split_groups: Optional[bool] = False,
    basis: Optional[str] = "spatial",
) -> Union[Tuple[np.ndarray, float], pd.DataFrame]:
    """Calculates expected cell neighbors and p-value using a permutation test.

    Test described in :cite:p:`Fang2022`, consisting of position randomization within a radius.
    Instead of z-test, the p-value is derived from the number of permutations with higher count than
    observed in the data.

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
    cell_to_neighbors_obs
         Mapping of each cell to its neighbors in the observed data, as calculated 
         by :func:`monkeybread.calc.cell_neighbors`.
    neighbor_radius
        The radius in which cells are considered neighbors. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2 (i.e., cells in 
        direct contact). This requires width and height columns to be present in `adata.obs`. Should be 
        the same as used in :func:`monkeybread.calc.cell_neighbors`.
    perm_radius
        The radius within which to randomize location, in coordinate units.
    n_perms
        The number of permutations to run.
    split_groups
        Perform calculations using each possible pair from group1 and group2 instead of considering
        the groups as a whole.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.

    Returns
    -------
    If `split_groups = False`, a length-two tuple will be returned. The first element is an array
    containing the number of cells observed for each permutation. The second element is a p_value
    comparing the expected number of neighbors to the observed number of neighbors. If 
    `split_groups = True`, a dataframe will be provided where each cell contains `p_val` for that 
    combination of `group1` (columns) and `group2` (rows).
    """
    # Convert string groups to lists
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Pull out cells corresponding to both groups
    group_cells_ad = adata[[g in group1 or g in group2 for g in adata.obs[groupby]]].copy()
    g1_index = adata[[g in group1 for g in adata.obs[groupby]]].obs.index
    g2_index = adata[[g in group2 for g in adata.obs[groupby]]].obs.index

    # Runs through position permutations
    if split_groups:
        perm_neighbor_counts = {g1: {g2: np.zeros(n_perms) for g2 in group2} for g1 in group1}
    else:
        perm_neighbor_counts = np.zeros(n_perms)
    for i in range(n_perms):
        # Adds a layer X_{basis}_random to group_cells_ad.obsm
        mb.util.randomize_positions(group_cells_ad, radius=perm_radius, basis=basis)
        group1_to_group2_neighbors = mb.calc.cell_neighbors(
            group_cells_ad, groupby, group1, group2, radius=neighbor_radius, basis=f"{basis}_random"
        )

        if split_groups:
            # Splits groups into pairwise comparisons
            # Preferred over recursion to minimize randomization of positions
            group1_to_group2_neighbor_counts = {
                g1: {
                    g2: mb.util.neighbor_count(
                        group1_to_group2_neighbors,
                        group_cells_ad[group_cells_ad.obs[groupby] == g1].obs.index,
                        group_cells_ad[group_cells_ad.obs[groupby] == g2].obs.index,
                    )
                    for g2 in group2
                }
                for g1 in group1
            }
            for (g1, g2) in itertools.product(group1, group2):
                perm_neighbor_counts[g1][g2][i] = group1_to_group2_neighbor_counts[g1][g2]
        else:
            perm_neighbor_counts[i] = mb.util.neighbor_count(group1_to_group2_neighbors, g1_index, g2_index)

    # Calculate p_values
    if split_groups:
        group1_to_group2_actual_counts = {
            g1: {
                g2: mb.util.neighbor_count(
                    cell_to_neighbors_obs,
                    group_cells_ad[group_cells_ad.obs[groupby] == g1].obs.index,
                    group_cells_ad[group_cells_ad.obs[groupby] == g2].obs.index,
                )
                for g2 in group2
            }
            for g1 in group1
        }
        group1_to_group2_p_values = {
            g1: {
                g2: (np.sum(np.where(perm_neighbor_counts[g1][g2] >= group1_to_group2_actual_counts[g1][g2], 1, 0)) + 1)
                / (n_perms + 1)
                for g2 in group2
            }
            for g1 in group1
        }
        return pd.DataFrame(group1_to_group2_p_values)
    else:
        actual_count = mb.util.neighbor_count(cell_to_neighbors_obs, g1_index, g2_index)
        p_val = (np.sum(np.where(perm_neighbor_counts >= actual_count, 1, 0)) + 1) / (n_perms + 1)
        return perm_neighbor_counts, p_val
