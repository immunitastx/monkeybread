from typing import Dict, List, Optional, Set, Union

import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def cell_contact(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    basis: Optional[str] = "spatial",
    radius: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """Detects contact between two groups of cells.

    The output takes the form of a dictionary mapping from cells in `group1` to cells in `group2`
    that the cell contacts. If a cell is present in both `group1` and `group2`, the mapping will be
    reflexive - the cell may both appear in the keys and the values.

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
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    A mapping from cell ids in `group1` to sets of cell ids in `group2` indicating contact.
    """
    # Convert groups to lists if single group provided
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Create adata views of cells from each group
    group1_cells = adata[[g in group1 for g in adata.obs[groupby]]]
    group2_cells = adata[[g in group2 for g in adata.obs[groupby]]]
    obsm_key = f"X_{basis}"

    # Infer radius if not provided
    if radius is None:
        radius = 0.5 * np.mean(
            [np.mean(group1_cells.obs["height"]), np.mean(group1_cells.obs["width"])]
        ) + 0.5 * np.mean([np.mean(group2_cells.obs["height"]), np.mean(group2_cells.obs["width"])])

    # Find nearest neighbors within radius
    nbrs = NearestNeighbors(radius=radius).fit(group2_cells.obsm[obsm_key])
    distances, indices = nbrs.radius_neighbors(group1_cells.obsm[obsm_key])

    # Filter out cells with no contact and pull out cell indices
    mask = [len(d) > 0 for d in distances]
    group1_indices = group1_cells.obs.index[mask]
    group2_indices = [group2_cells.obs.index[i] for i in indices[mask]]

    # Convert to adjacency list format and remove self contact
    touches = {
        g1_index: set(g2_indices).difference({g1_index}) for g1_index, g2_indices in zip(group1_indices, group2_indices)
    }
    contact_empty_removed = {k: v for k, v in touches.items() if len(v) > 0}
    return contact_empty_removed
