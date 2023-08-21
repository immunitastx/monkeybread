from typing import Optional
from anndata import AnnData
from omnipath.interactions import import_intercell_network

def load_ligand_receptor_pairs_omnipath(
        adata: AnnData, 
        require_gene: Optional[str]=None
    ):
    """
    Load ligand-receptor pairs from the Omnipath database that are also
    in the provided dataset.

    Parameters
    ----------
    adata
        An AnnData object 
    require_gene
        Only return ligand-receptor pairs where either the ligand or 
        receptor is this argument

    Returns
    -------
    List of ligand-receptor pair tuples
    """
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

    if require_gene:
        lr_pairs = [
            (s, t)
            for s, t in lr_pairs
            if s == require_gene or t == require_gene
        ]
    return lr_pairs
