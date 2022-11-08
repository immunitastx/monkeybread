from anndata import AnnData
from typing import Union, List, Optional, Dict, Set
from sklearn.neighbors import NearestNeighbors
import numpy as np


def cell_contact(
        adata: AnnData,
        groupby: str,
        group1: Union[str, List[str]],
        group2: Union[str, List[str]],
        basis: Optional[str] = "spatial",
        radius: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """Detects contact between two groups of cells. The output takes the form of a dictionary
    mapping from cells in `group1` to cells in `group2` that the cell contacts. If a cell is present
    in both `group1` and `group2`, the mapping will be reflexive - the cell may both appear in the
    keys and the values.

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
    basis
        A key in `adata.obsm` to use for cell coordinates.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    contacts
        A mapping from cell ids in `group1` to sets of cell ids in `group2` indicating contact.
    """
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]
    group1_cells = adata[[g in group1 for g in adata.obs[groupby]]]
    group2_cells = adata[[g in group2 for g in adata.obs[groupby]]]
    obsm_key = f"X_{basis}"
    if radius is None:
        radius = np.mean([np.mean([np.mean(group1_cells.obs["height"]),
                                   np.mean(group1_cells.obs["width"])]),
                          np.mean([np.mean(group2_cells.obs["height"]),
                                   np.mean(group2_cells.obs["width"])])])
    nbrs = NearestNeighbors(radius = radius).fit(group2_cells.obsm[obsm_key])
    distances, indices = nbrs.radius_neighbors(group1_cells.obsm[obsm_key])
    mask = [len(d) > 0 for d in distances]
    group1_indices = group1_cells.obs.index[mask]
    group2_indices = [group2_cells.obs.index[index] for index in indices[mask]]
    touches = {
        g1_index: set(g2_indices).difference({g1_index}) for g1_index, g2_indices in
        zip(group1_indices, group2_indices)
    }
    contact_empty_removed = {
        k: v for k, v in touches.items() if len(v) > 0
    }
    return contact_empty_removed
