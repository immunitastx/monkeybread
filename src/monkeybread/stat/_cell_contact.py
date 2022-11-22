import itertools
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData

import monkeybread as mb


def cell_contact(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    actual_contact: Dict[str, Set[str]],
    contact_radius: Optional[float] = None,
    perm_radius: Optional[float] = 100,
    n_perms: Optional[int] = 1000,
    split_groups: Optional[bool] = False,
    basis: Optional[str] = "spatial",
) -> Union[Tuple[np.ndarray, float], pd.DataFrame]:
    """Calculates expected cell contact and p-value using a permutation test.

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
    actual_contact
         The actual cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    contact_radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`. Should be the same as used in
        :func:`monkeybread.calc.cell_contact`.
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
    containing the number of contacts observed for each permutation. The second element is a p_value
    comparing the expected contact to the actual contact. If `split_groups = True`, a dataframe will
    be provided where each cell contains `p_val` for that combination of `group1` (columns) and
    `group2` (rows).
    """
    # Convert string groups to lists
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Pull out cells corresponding to both groups
    group_cells = adata[[g in group1 or g in group2 for g in adata.obs[groupby]]].copy()
    g1_index = adata[[g in group1 for g in adata.obs[groupby]]].obs.index
    g2_index = adata[[g in group2 for g in adata.obs[groupby]]].obs.index

    # Runs through position permutations
    if split_groups:
        perm_contact = {g1: {g2: np.zeros(n_perms) for g2 in group2} for g1 in group1}
    else:
        perm_contact = np.zeros(n_perms)
    for i in range(n_perms):
        mb.util.randomize_positions(group_cells, radius=perm_radius, basis=basis)
        perm_i_contact = mb.calc.cell_contact(
            group_cells, groupby, group1, group2, radius=contact_radius, basis=f"{basis}_random"
        )

        if split_groups:
            # Splits groups into pairwise comparisons
            # Preferred over recursion to minimize randomization of positions
            touches_dict = {
                g1: {
                    g2: mb.util.contact_count(
                        perm_i_contact,
                        group_cells[group_cells.obs[groupby] == g1].obs.index,
                        group_cells[group_cells.obs[groupby] == g2].obs.index,
                    )
                    for g2 in group2
                }
                for g1 in group1
            }
            for (g1, g2) in itertools.product(group1, group2):
                perm_contact[g1][g2][i] = touches_dict[g1][g2]
        else:
            perm_contact[i] = mb.util.contact_count(perm_i_contact, g1_index, g2_index)

    # Calculate p_values
    if split_groups:
        actual_count = {
            g1: {
                g2: mb.util.contact_count(
                    actual_contact,
                    group_cells[group_cells.obs[groupby] == g1].obs.index,
                    group_cells[group_cells.obs[groupby] == g2].obs.index,
                )
                for g2 in group2
            }
            for g1 in group1
        }
        p_values = {
            g1: {
                g2: (np.sum(np.where(perm_contact[g1][g2] >= actual_count[g1][g2], 1, 0)) + 1) / (n_perms + 1)
                for g2 in group2
            }
            for g1 in group1
        }
        return pd.DataFrame(p_values)
    else:
        actual_count = mb.util.contact_count(actual_contact, g1_index, g2_index)
        p_val = (np.sum(np.where(perm_contact >= actual_count, 1, 0)) + 1) / (n_perms + 1)
        return perm_contact, p_val
