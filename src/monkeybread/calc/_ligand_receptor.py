"""
Calculate a ligand-receptor co-expression score between neighboring cells.
"""

import itertools
from typing import Dict, List, Set, Tuple, Union, Optional
from omnipath.interactions import import_intercell_network

import numpy as np
from anndata import AnnData


def ligand_receptor_score(
    adata: AnnData,
    cell_to_neighbors: Dict[str, Set[str]],
    lr_pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Calculates an average co-expression score of a 
    ligand-receptor pair between neighboring cells..

    Statistical test is as described in :cite:p:`He2021.11.03.467020` (See 
    Figure 4).

    Calculates scores for ligand-receptor pairs in neighboring cells.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        A mapping of cells to their neighbors, as calculated by 
        :func:`monkeybread.calc.cell_neighbors`.
    lr_pairs
        One or multiple tuples corresponding to (ligand, receptor). If `None` then
        ligand/receptor pairs will be downloaded from Omnipath via the 
        :func:`omnipath.interactions.import_intercell_network`.

    Returns
    -------
    A mapping from ligand-receptor tuple pairs to pair scores.
    """
    # Convert to list if only one pair provided
    if lr_pairs is None:
        print("Using ligand/receptor pairs from omnipath...")
        interactions = import_intercell_network(
            transmitter_params={"categories": "ligand"},
            receiver_params={"categories": "receptor"}
        )
        genes_in_data = set(adata.var_names)
        lr_pairs = [
            (s, t)
            for s, t in zip(
                interactions['genesymbol_intercell_source'], 
                interactions['genesymbol_intercell_target']
            )
            if s in genes_in_data and t in genes_in_data
        ]
    elif isinstance(lr_pairs, tuple):
        lr_pairs = [lr_pairs]
    
    # Convert contacts dictionary to list of cell_id tuples
    neighbor_pairs = np.array(
        list(
            itertools.chain.from_iterable(
                itertools.product([cell], neighbors) 
                for cell, neighbors in cell_to_neighbors.items()
            )
        )
    )

    # Define score calculation function
    calculate_score = (
        lambda l, r: np.sum(np.sqrt(
            adata[neighbor_pairs.T[0], l].X.toarray() * adata[neighbor_pairs.T[1], r].X.toarray()
        ))
        / neighbor_pairs.shape[0]
    )

    lr_pair_to_score = {(ligand, receptor): calculate_score(ligand, receptor) for ligand, receptor in lr_pairs}

    return lr_pair_to_score


def ligand_receptor_score_per_niche(
    adata: AnnData,
    niche_to_cell_to_neighbors: Dict[str, Set[str]],
    lr_pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
) -> Dict[str, Dict[Tuple[str, str], float]]:
    """
    Calculates an average co-expression score of a
    ligand-receptor pair between neighboring cells within each cellular niche
    calculated by :func:`monkeybread.calc.cellular_niches`. Statistical test is as described 
    in :cite:p:`He2021.11.03.467020` (See Figure 4). This function is a wrapper around 
    :func:`monkeybread.calc.ligand_receptor_score` and calls this function separately for 
    each niche.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_to_neighbors
        A mapping of cells to their neighbors, as calculated by
        :func:`monkeybread.calc.cell_neighbors`.
    lr_pairs
        One or multiple tuples corresponding to (ligand, receptor). If `None` then
        ligand/receptor pairs will be downloaded from Omnipath via the
        :func:`omnipath.interactions.import_intercell_network`.

    Returns
    -------
    A dictionary mapping each niche to a sub-dictionary mapping each ligand-receptor 
    pair to its co-expression score.
    """

    niche_to_lr_pair_to_score = {}
    for niche, cell_to_neighbors in niche_to_cell_to_neighbors.items():
        
        lr_pair_to_score = ligand_receptor_score(
            adata,
            cell_to_neighbors,
            lr_pairs,
        )
        niche_to_lr_pair_to_score[niche] = lr_pair_to_score
    return niche_to_lr_pair_to_score



