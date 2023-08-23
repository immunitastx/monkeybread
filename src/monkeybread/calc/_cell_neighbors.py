from typing import Dict, List, Optional, Set, Union, Iterable
import numpy as np
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

def cell_neighbors(
    adata: AnnData,
    groupby: str,
    group1: Union[str, List[str]],
    group2: Union[str, List[str]],
    basis: Optional[str] = "X_spatial",
    radius: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """Calculate cell neighbors.

    For each cell within a given group of cells, calculate all neighbors that are members of a
    second group of cells (e.g., for all T cells, calculate all neighbors that are B cells).

    The output takes the form of a dictionary mapping from cells in `group1` to cells in `group2`
    that are neighbors. Note, if a cell is present in both `group1` and `group2`, the mapping will be
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
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    A mapping from cell ids in `group1` to sets of cell ids in `group2` that are its neighbors 
    (i.e., within `radius` distance).
    """
    # Convert groups to lists if single group provided
    if type(group1) == str:
        group1 = [group1]
    if type(group2) == str:
        group2 = [group2]

    # Create adata views of cells from each group
    group1_cells = adata[[g in group1 for g in adata.obs[groupby]]]
    group2_cells = adata[[g in group2 for g in adata.obs[groupby]]]
    obsm_key = basis

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
    cell_to_neighbors = {
        g1_index: set(g2_indices).difference({g1_index}) 
        for g1_index, g2_indices in zip(group1_indices, group2_indices)
    }
    cell_to_neighbors = {k: v for k, v in cell_to_neighbors.items()}

    return cell_to_neighbors


def cell_neighbors_from_masks(
        adata: AnnData,
        mask_group1: Iterable[bool],
        mask_group2: Iterable[bool],
        basis="X_spatial",
        radius=None,
    ) -> Dict[str, Set[str]]:
    """Calculate cell neighbors.

    For each cell within a given group of cells, calculate all neighbors that are members of a
    second group of cells (e.g., for all T cells, calculate all neighbors that are B cells). This
    function differs from `cell_neighbors` based on how the two groups are defined. This function
    accepts two masks (Boolean valued iterables). The first mask defines the cells in the first group 
    (Those indices set to `True`) and the second mask defines the cells in the second group. 

    The output takes the form of a dictionary mapping from cells in `group1` to cells in `group2`
    that are neighbors. Note, if a cell is present in both `group1` and `group2`, the mapping will be
    reflexive - the cell may both appear in the keys and the values. 

    The output takes the form of a dictionary mapping from cells in `group1` to cells in `group2`
    that the cell contacts. If a cell is present in both `group1` and `group2`, the mapping will be
    reflexive - the cell may both appear in the keys and the values.

    Parameters
    ----------
    adata
        Annotated data matrix.
    mask_group1
        An iterable of Boolean values specifying cells in the first group (all indices set to `True`).
    mask_group2
        An iterable of Boolean values specifying cells in the second group (all indices set to `True`).
    basis
        Coordinates in `adata.obsm[basis]` to use. Defaults to `X_spatial`.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    A mapping from cell ids in `group1` to sets of cell ids in `group2` that are its 
    neighbors (i.e., within `radius` distance).
    """
    # Key for coordinates
    obsm_key = basis

    # Subset the AnnData object
    adata1 = adata[mask_group1]
    adata2 = adata[mask_group2]

    # Infer radius if not provided
    if radius is None:
        radius = 0.5 * np.mean(
            [np.mean(group1_cells.obs["height"]), np.mean(group1_cells.obs["width"])]
        ) + 0.5 * np.mean([np.mean(group2_cells.obs["height"]), np.mean(group2_cells.obs["width"])])

    # Find nearest neighbors within radius
    nbrs = NearestNeighbors(radius=radius).fit(adata2.obsm[obsm_key])
    distances, indices = nbrs.radius_neighbors(adata1.obsm[obsm_key])

    # Compute neighborhoods
    cells2 = list(adata2.obs.index)
    cell_to_neighbors = {}
    for cell, dists, inds  in zip(adata1.obs.index, distances, indices):
        if len(inds) > 0:
            # Get neighbors
            neighbors = set([
                cells2[i]
                for i in inds
            ])

            # Remove self from neighbors list
            neighbors -= {cell} 
    
            # Add neighbors
            cell_to_neighbors[cell] = neighbors

    return cell_to_neighbors


def cell_neighbors_per_niche(
        adata: AnnData,
        niche_key: str,
        groupby: str,
        group1: Union[str, List[str]],
        group2: Union[str, List[str]],
        basis: Optional[str] = "X_spatial",
        radius: Optional[float] = None,
    ) -> Dict[str, Dict[str, Set[str]]]:
    """
    Calculate cell neighbors, but within the cellular niches calculated by
    :func:`monkeybread.calc.cellular_niches`.

    This function is a wrapper around :func:`monkebread.calc.cell_neighbors`
    that calls this function separately on cells within each niche

    Parameters
    ----------
    adata
        Annotated data matrix.
    niche_key
        The column in `adata.obs` storing the annotated niche of each cells
        as calculated by :func:`monkeybread.calc.cellular_niches`
    groupby
        A categorical column in `adata.obs` to classify groups.
    group1
        Either one group or a list of groups from `adata.obs[groupby]`.
    group2
        Either one group or a list of groups from `adata.obs[groupby]`.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    A mapping from each niche id to another mapping that maps each cell id in `group1` to sets 
    of cell ids in `group2` that are its neighbors (i.e., within `radius` distance).
    """
    niche_to_cell_to_neighbors = {}
    for niche in sorted(set(adata.obs[niche_key])):
        adata_niche = adata[adata.obs[niche_key] == niche]

        cell_to_neighbors = cell_neighbors(
            adata_niche,
            groupby,
            group1,
            group2,
            basis,
            radius,
        )
        niche_to_cell_to_neighbors[niche] = cell_to_neighbors
    return niche_to_cell_to_neighbors


def cell_neighbors_per_niche_from_masks(
        adata: AnnData,
        niche_key: str,
        mask_group1: Iterable[bool],
        mask_group2: Iterable[bool],
        basis="X_spatial",
        radius=None,
    ) -> Dict[str, Dict[str, Set[str]]]:
    """
    Calculate cell neighbors, but within the cellular niches calculated by
    :func:`monkeybread.calc.cellular_niches_from_masks`. This is similar to 
    :func:`monkeybread.calc.cellular_niches_per_niche`, but differs from it based on how the two 
    groups are defined. This function accepts two masks (Boolean valued iterables). The first 
    mask defines the cells in the first group (those indices set to `True`) and the second mask 
    defines the cells in the second group.

    This function is a wrapper around :func:`monkebread.calc.cell_neighbors_from_masks`
    that calls this function separately on cells within each niche. 

    Parameters
    ----------
    adata
        Annotated data matrix.
    mask_group1
        An iterable of Boolean values specifying cells in the first group (all indices set to `True`).
    mask_group2
        An iterable of Boolean values specifying cells in the second group (all indices set to `True`).
    basis
        Coordinates in `adata.obsm[basis]` to use. Defaults to `X_spatial`.
    radius
        The radius in which cells are considered touching. If not provided, will be calculated using
        half of the average radius of group1 + half of the average radius of group2. This requires
        width and height columns to be present in `adata.obs`.

    Returns
    -------
    A mapping from each niche id to another mapping that maps each cell id in `group1` to sets 
    of cell ids in `group2` that are its neighbors (i.e., within `radius` distance).
    """
    niche_to_cell_to_neighbors = {}
    for niche in sorted(set(adata.obs[niche_key])):
        adata_niche = adata[adata.obs[niche_key] == niche]

        mask_group1_niche = np.array(mask_group1)[adata.obs[niche_key] == niche]
        mask_group2_niche = np.array(mask_group2)[adata.obs[niche_key] == niche]
        
        cell_to_neighbors=cell_neighbors_from_masks(
            adata_niche,
            mask_group1_niche,
            mask_group2_niche,
            basis,
            radius,
        )         
        niche_to_cell_to_neighbors[niche] = cell_to_neighbors
    return niche_to_cell_to_neighbors



