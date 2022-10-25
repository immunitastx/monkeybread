import pandas as pd
from anndata import AnnData
import numpy as np
import monkeybread as mb
from typing import Optional, Union, List, Dict, Set, Tuple
import statsmodels.stats.weightstats as sm


def cell_contact(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    actual_contact: Dict[str, Set[str]],
    contact_radius: Optional[float] = None,
    perm_radius: Optional[float] = 100,
    n_perms: Optional[int] = 100,
    split_groups: Optional[bool] = False,
) -> Union[Tuple[np.ndarray, float], pd.DataFrame]:
    """Calculates expected cell contact and p-value using a permutation test as described in \
    `this paper<ncbi.nlm.nih.gov/pmc/articles/PMC9262715/>`.

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
         The actual cell contacts, as calculated by `monkeybread.calc.cell_contact`.
    contact_radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`. Should be the same as used in
        `monkeybread.calc.cell_contact`.
    perm_radius
        The radius within which to randomize location, in coordinate units.
    n_perms
        The number of permutations to run.
    split_groups
        Perform calculations using each possible pair from group1 and group2 instead of considering
        the groups as a whole.

    Returns
    -------
    expected_touches
        An array containing the number of contacts observed for each permutation.
    p_val
        A p_value comparing the expected contact to the actual contact.
    dataframe
        If `split_groups = True`, a dataframe will be provided where each cell contains
        `expected_touches` and `p_val` for that combination of `group1` (columns) and
        `group2` (rows).
    """
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]
    if split_groups:
        df_dict = {
            g1: {g2: cell_contact(adata, groupby, g1, g2, actual_contact, contact_radius,
                                  perm_radius, n_perms, split_groups = False)[1] for g2 in group2}
            for g1 in group1
        }
        return pd.DataFrame(df_dict)
    both_groups = set(group1).union(set(group2))
    data_groups = adata[[g in both_groups for g in adata.obs[groupby]]].copy()
    num_touches = lambda t: sum([0 if k not in group1 else sum([g2 in group2 for g2 in v]) for k, v
                                 in t.items()])
    expected_touches = np.zeros(n_perms)
    for i in range(n_perms):
        mb.util.randomize_positions(data_groups, radius = perm_radius)
        touches = mb.calc.cell_contact(data_groups, groupby, group1, group2,
                                       radius = contact_radius, basis = "spatial_random")
        expected_touches[i] = num_touches(touches)
    actual_touches = num_touches(actual_contact)
    t_score, p_val = sm.ztest(x1 = expected_touches, value = actual_touches,
                              alternative = "smaller")
    return expected_touches, p_val
