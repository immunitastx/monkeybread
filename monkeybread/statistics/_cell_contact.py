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
    actual_contact: Optional[Dict[str, Set[str]]] = None,
    contact_radius: Optional[float] = None,
    perm_radius: Optional[float] = 100,
    n_perms: Optional[int] = 100,
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """Calculates expected cell contact and p-value using a permutation test as described in \
    :link:`ncbi.nlm.nih.gov/pmc/articles/PMC9262715/`.

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

    Returns
    -------
    expected_touches
        An array containing the number of contacts observed for each permutation.
    p_val
        Optionally, a p_value comparing the expected contact to the actual contact. Only calculated
        if `actual_contact` is provided.
    """
    both_groups = set(group1).union(set(group2))
    data_groups = adata[[g in both_groups for g in adata.obs[groupby]]].copy()
    num_touches = lambda t: sum([len(v) for v in t.values()])
    expected_touches = np.zeros(n_perms)
    for i in range(n_perms):
        mb.util.randomize_positions(data_groups, radius = perm_radius)
        touches = mb.calc.cell_contact(data_groups, groupby, group1, group2,
                                       radius = contact_radius, basis = "spatial_random")
        expected_touches[i] = num_touches(touches)
    if actual_contact is not None:
        num_touches = sum([len(v) for v in actual_contact.values()])
        t_score, p_val = sm.ztest(x1 = expected_touches, x2 = [num_touches],
                                  alternative = "smaller")
        return expected_touches, p_val
    return expected_touches
