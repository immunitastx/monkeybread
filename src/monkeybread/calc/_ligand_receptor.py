import itertools
from typing import Dict, List, Set, Tuple, Union

import numpy as np
from anndata import AnnData


def ligand_receptor_score(
    adata: AnnData,
    contacts: Dict[str, Set[str]],
    lr_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
) -> Dict[Tuple[str, str], float]:
    """Calculates scores for ligand-receptor pairs in contacting cells.

    Score calculation is as described in :cite:p:`He2021.11.03.467020`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    contacts
        The cell contacts, as calculated by :func:`monkeybread.calc.cell_contact`.
    lr_pairs
        One or multiple tuples corresponding to (ligand, receptor).

    Returns
    -------
    A mapping from ligand-receptor tuple pairs to pair scores.
    """
    # Convert to list if only one pair provided
    if isinstance(lr_pairs, tuple):
        lr_pairs = [lr_pairs]

    # Convert contacts dictionary to list of cell_id tuples
    linkages = np.array(
        list(
            itertools.chain.from_iterable(
                itertools.product([cell], cell_contacts) for cell, cell_contacts in contacts.items()
            )
        )
    )

    # Define score calculation function
    calculate_score = (
        lambda l, r: np.sum(np.sqrt(adata[linkages.T[0], l].X.toarray() * adata[linkages.T[1], r].X.toarray()))
        / linkages.shape[0]
    )

    lr_pair_to_score = {(ligand, receptor): calculate_score(ligand, receptor) for ligand, receptor in lr_pairs}

    return lr_pair_to_score
