import itertools
from typing import Dict, Optional, Set, Tuple

import numpy as np
from anndata import AnnData

import monkeybread as mb


def ligand_receptor_score(
    adata: AnnData,
    contacts: Dict[str, Set[str]],
    actual_scores: Dict[Tuple[str, str], float],
    n_perms: Optional[int] = 100,
) -> Dict[Tuple[str, str], Tuple[np.ndarray, float]]:
    """Calculates statistical significance of ligand-receptor pairs in contacting cells.

    Statistical test is as described in :cite:p:`He2021.11.03.467020`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    contacts
        The cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    actual_scores
        The observed scores, as calculated by :func:`monkeybread.calc.ligand_receptor_score`.
    n_perms
        The number of permutations to run.

    Returns
    -------
    A mapping from ligand-receptor tuple pairs to a tuple containing the distribution of scores and
    p-value.
    """
    lr_pairs = list(actual_scores.keys())

    # Set up dictionary with empty values
    lr_to_dist = {lr: np.zeros(n_perms) for lr in actual_scores.keys()}

    # Pull out receptor cells as well as counts for each ligand cell for use in permutation
    receptor_cells = np.array(list(itertools.chain.from_iterable(contacts.values())))
    # Note that this corresponds to the starting index for each ligand_cell in the above receptor
    # cell array
    ligand_index_starts = np.array([0] + list(itertools.accumulate(len(v) for v in contacts.values())))

    # Iterate over permutations
    for i in range(n_perms):
        np.random.shuffle(receptor_cells)
        # Randomize linkages between ligand cells and receptor cells, pulling out the appropriate
        # number of linkages for each ligand and recentor
        perm_i_lcell_to_rcells = {
            lcell: receptor_cells[idx_start : idx_start + len(rcells)]
            for (lcell, rcells), idx_start in zip(contacts.items(), ligand_index_starts)
        }

        # Use permutation "contact" to generate scores
        perm_i_scores = mb.calc.ligand_receptor_score(adata, perm_i_lcell_to_rcells, lr_pairs=lr_pairs)

        # Add scores to distributions
        for lr_pair, score in perm_i_scores.items():
            lr_to_dist[lr_pair][i] = score

    # Calculate p-values
    lr_to_pval = {
        lr: (np.sum(np.where(lr_to_dist[lr] >= actual_score, 1, 0)) + 1) / (n_perms + 1)
        for lr, actual_score in actual_scores.items()
    }

    # Zip together distribution and p_vals
    lr_to_dist_pval = {
        lr: (dist, pval) for lr, dist, pval in zip(actual_scores.keys(), lr_to_dist.values(), lr_to_pval.values())
    }

    return lr_to_dist_pval
