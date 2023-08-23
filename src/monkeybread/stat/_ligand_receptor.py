"""
Runs a statistical test testing for enrichment of ligand-receptor co-expression 
between neighboring cells.

Authors: Dillon Scott and Matthew Bernstein
"""

import itertools
from typing import Dict, Optional, Set, Tuple
from tqdm import tqdm
import numpy as np
from anndata import AnnData

import monkeybread as mb


def ligand_receptor_score(
    adata: AnnData,
    cell_to_neighbors: Dict[str, Set[str]],
    actual_scores: Dict[Tuple[str, str], float],
    n_perms: Optional[int] = 100
) -> Dict[Tuple[str, str], Tuple[np.ndarray, float]]:
    """Calculates statistical significance of the co-expression of
    ligand-receptor pairs between neighboring cells.

    Statistical test is as described in :cite:p:`He2021.11.03.467020` (See 
    Figure 4). This function runs this test separately on each ligand-receptor
    pair among a set of pairs provided by the user.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        A dictionary mapping each cell to its neighbors as calculated by :func:`monkeybread.calc.cell_neighbors`
    actual_scores
        The observed co-expression scores as calculated by :func:`monkeybread.calc.ligand_receptor_score`
    n_perms
        Number of permutations to run in the permutation test

    Returns
    -------
    A mapping from ligand-receptor tuple pairs to a tuple containing the distribution of co-expression 
    scores under permutation (i.e., null distribution) with the associated p-value of the observed
    co-expression score.
    """
    lr_pairs = list(actual_scores.keys())

    # Set up dictionary with empty values
    lr_to_dist = {lr: np.zeros(n_perms) for lr in actual_scores.keys()}

    # Pull out receptor cells as well as counts for each ligand cell for use in permutation
    receptor_cells = np.array(list(itertools.chain.from_iterable(cell_to_neighbors.values())))
    # Note that this corresponds to the starting index for each ligand_cell in the above receptor
    # cell array
    ligand_index_starts = np.array([0] + list(itertools.accumulate(len(v) for v in cell_to_neighbors.values())))

    # Iterate over permutations
    for i in tqdm(range(n_perms)):
        np.random.shuffle(receptor_cells)
        # Randomize linkages between ligand cells and receptor cells, pulling out the appropriate
        # number of linkages for each ligand and receptor
        perm_i_lcell_to_rcells = {
            lcell: receptor_cells[idx_start : idx_start + len(rcells)]
            for (lcell, rcells), idx_start in zip(cell_to_neighbors.items(), ligand_index_starts)
        }

        # Use permutation "contact" to generate scores
        perm_i_scores = mb.calc.ligand_receptor_score(
            adata, 
            perm_i_lcell_to_rcells, 
            lr_pairs=lr_pairs
        )

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
        lr: (dist, pval) 
        for lr, dist, pval in zip(
            actual_scores.keys(), 
            lr_to_dist.values(), 
            lr_to_pval.values()
        )
    }

    return lr_to_dist_pval
