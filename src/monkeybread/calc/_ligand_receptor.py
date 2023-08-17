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
    contacts: Dict[str, Set[str]],
    lr_pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    "Calculates an average co-expression score of a 
    ligand-receptor pair between neighboring cells..

    Statistical test is as described in :cite:p:`He2021.11.03.467020` (See 
    Figure 4).

    Calculates scores for ligand-receptor pairs in contacting cells.

    Score calculation is as described in :cite:p:`He2021.11.03.467020`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    contacts
        The cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
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
                for cell, neighbors in contacts.items()
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
