from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors
import scanpy as sc

import monkeybread as mb


def neighborhood_profile(
    adata: AnnData,
    groupby: str,
    basis: Optional[str] = "X_spatial",
    subset_output_features: Optional[Sequence[str]] = None,
    subset_cells_by_group: Optional[Sequence[str]] = None,
    radius: Optional[float] = None,
    n_neighbors: Optional[int] = None,
    normalize_counts: Optional[bool] = True,
    standard_scale: Optional[bool] = True,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None
) -> AnnData:
    """Calculates a neighborhood profile for each cell.

    The resulting AnnData object will have the same index corresponding to rows/cells, but a new
    index corresponding to columns, one column for each category in `adata.obs[groupby]`. Instead of
    a gene expression profile, each column corresponds to the proportion of cells in the surrounding
    radius that belong to the respective category.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `obs` to use for neighborhood profile calculations.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`.
    subset_output_features
        A list of groups from `adata.obs[groupby]` to include in the resulting `adata.var_names`.
        Will not affect the calculations themselves, only which results are provided.
    subset_cells_by_group
        A list of groups in `adata.obs[groupby]` to restrict the resulting AnnData object to. Only
        cells in those groups will be included in the resulting `adata.obs_names`.
    radius
        Radius in coordinate space to include nearby cells for neighborhood profile calculation.
    n_neighbors
        Number of neighbors to consider for the neighborhood profile calculations.
    normalize_counts
        Normalize neighborhood counts to proportions instead of raw counts. Note, if
        `subset_output_features` is provided and `normalize_counts = True`, the normalization step will
        be performed before removing groups, so proportions will not sum to 1.
    standard_scale
        Compute z-score of neighborhood values by subtracting the mean and dividing by
        the standard deviation.
    clip_min
        Clip values that are less than `clip_min` to `clip_min`
    clip_max
        Clip values that are greater than `clip_max` to `clip_max`

    Returns
    -------
    A new AnnData object where columns now correspond to neighborhood profile proportions. All .obs
    columns will be carried over from the provided AnnData object, and they will share an index.
    """
    if adata.obs[groupby].dtype != "category":
        raise ValueError(f"adata.obs['{groupby}'] is not categorical.")
    categories = adata.obs[groupby].cat.categories

    if radius is not None and n_neighbors is not None:
        raise ValueError("Cannot specify both radius and n_neighbors")

    # Calculate cell -> neighbors dictionary
    if radius is not None:
        # Use radius to define neighborhood
        cell_to_neighbors = mb.calc.cell_neighbors(
            adata, 
            groupby, 
            categories, 
            categories, 
            basis=basis, 
            radius=radius
        )
    elif n_neighbors is not None:
        # Use set number of neighbors to define neighborhood
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(adata.obsm[basis])
        distances, indices = nbrs.kneighbors(adata.obsm[basis])
        # Convert to dictionary, remove self-inclusion
        cell_to_neighbors = {
            cell: set(adata[neighbors].obs.index).difference({cell})
            for cell, neighbors in zip(adata.obs.index, indices)
        }
    else:
        raise ValueError("Must specify either radius or n_neighbors")

    # Get group for each cell
    cell_to_group = adata.obs[groupby]

    # Remove certain cells from neighborhood calculations
    if subset_cells_by_group is not None:
        mask = [g in subset_cells_by_group for g in adata.obs[groupby]]
    else:
        mask = [True] * adata.shape[0]

    # Count the number of each group in the neighborhood of each cell
    cell_to_neighbor_counts = {
        cell: Counter(cell_to_group[c] for c in cell_to_neighbors[cell]) if cell in cell_to_neighbors else {}
        for cell in adata[mask].obs.index
    }

    # Turn counts into fractions for each cell summing to 1
    neighbors_df = pd.DataFrame(cell_to_neighbor_counts).T.fillna(0)
    n_neighbors = neighbors_df.sum(axis=1)
    if normalize_counts:
        neighbors_df = neighbors_df.apply(func=lambda arr: arr / np.sum(arr), axis=1).fillna(0)

    # Create a new AnnData object with cells as rows and group names as columns
    neighbor_adata = AnnData(
        X=neighbors_df,
        obs=pd.DataFrame(
            {"n_neighbors": n_neighbors, groupby: adata[neighbors_df.index].obs[groupby]}, index=neighbors_df.index
        ),
        uns={"neighbor_radius": radius} if radius is not None else {},
        obsm={basis: adata[neighbors_df.index].obsm[basis].copy()},
        dtype=neighbors_df.to_numpy().dtype,
    )

    # Standard scale
    if standard_scale:
        sc.pp.scale(
            neighbor_adata,  
            zero_center=True
        )

    # Clip max and min
    if clip_min and clip_max:
        neighbor_adata.X = np.clip(
            neighbor_adata.X, 
            a_min=clip_min, 
            a_max=clip_max
        )


    if subset_output_features is not None:
        return neighbor_adata[:, subset_output_features].copy()
    else:
        return neighbor_adata



def cellular_niches(
        adata, 
        cell_type_key,
        radius,
        normalize_counts=True,
        standard_scale=True,
        clip_min=-5,
        clip_max=5,
        mask=None,
        n_neighbors=100,
        resolution=0.25,
        min_niche_size=0,
        key_added='niche',
        non_niche_value='other'
    ):
    """Compute cellular niches. Each cell is assigned to a "niche" that represents a set of 
    colocalizing cells with a distinct cell type composition. To generate niches, for each 
    cell, a neighborhood profile is calculated via :func:`monkeybread.calc.neighborhood_profile`.
    These neighborhood profiles are then clustered. The cluster ID of each cell is its assigned
    niche.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cell_type_key
        A categorical column in `obs` to use that stores the cell type of each cell to be
        used in generating the neighborhood profiles.
    radius
        Radius in coordinate space to include nearby cells for neighborhood profile calculation.
    normalize_counts
        Normalize neighborhood counts to proportions instead of raw counts. Note, if
        `subset_output_features` is provided and `normalize_counts = True`, the normalization step 
        will be performed before removing groups, so proportions will not sum to 1.
    standard_scale
        Compute z-score of neighborhood values by subtracting the mean and dividing by
        the standard deviation.
    clip_min
        Clip values that are less than `clip_min` to `clip_min`
    clip_max
        Clip values that are greater than `clip_max` to `clip_max`
    mask
        A Boolean-valued vector used to denote all cells for which a niche will be assigned. All 
        cells that are filtered out will be assigned to a niche with the name provided by the 
        `non_niche_value` argument
    n_neighbors
        Parameter to :func:`scanpy.pp.neighbors` that determines the number of neighbors used 
        to generate the k-nearest neighbors graph of the neighborhood profiles that will then 
        undergo graph clustering
    resolution
        Parameter to :func:`scanpy.tl.leiden` that determines the granularity of the clustering
        of the neighborhood profiles
    min_niche_size
        Consider only clusters larger than this value to denote cellular niches. Cells within
        any cluster that are smaller than this value will be assigned to a niche with the name
        provided by the `non_niche_value` argument
    key_added
        Column name added `adata.obs` that stores each cell's assigned niche
    non_niche_value
        Cells that have been filtered out using the `mask` argument or that fall within a cluster
        smaller than `min_niche_size` will be assigned a niche with this name

    Returns
    -------
    An AnnData object storing the neighborhood profiles as computed by 
    :func:`monkeybread.calc.neighborhood_profile`
    """
    # Compute neighborhood profiles
    print("Computing neighborhood profiles...")
    adata_neighbors = neighborhood_profile(
        adata,
        groupby=cell_type_key,
        radius=radius,
        normalize_counts=normalize_counts,
        standard_scale=standard_scale,
        clip_min=clip_min,
        clip_max=clip_max
    )
    adata_neighbors = adata_neighbors[adata.obs.index]

    # Filter cells to keep only cells we wish to use for niche
    # analysis
    if mask is not None:
        adata_neighbors_mask = adata_neighbors[mask]
    else:
        adata_neighbors_mask

    # Cluster neighborhood profiles
    print("Clustering neighborhood profiles...")
    sc.pp.neighbors(
        adata_neighbors_mask, 
        n_neighbors=n_neighbors
    )
    sc.tl.leiden(
        adata_neighbors_mask, 
        resolution=resolution
    )

    # Filter out small clusters to generate final niches
    print("Generating niches...")
    clust_to_count = Counter(adata_neighbors_mask.obs['leiden'])
    small_clusts = [
        clust
        for clust, count in clust_to_count.items()
        if count < min_niche_size
    ]
    adata_neighbors_mask.obs[key_added] = [
        clust
        if clust not in small_clusts
        else non_niche_value
        for clust in adata_neighbors_mask.obs['leiden']
    ]

    # Add niche values ot the original AnnData object 
    cell_to_niche = {
        cell: niche
        for cell, niche in zip(
            adata_neighbors_mask.obs.index, 
            adata_neighbors_mask.obs[key_added]
        )
    }
    adata.obs[key_added] = [
        cell_to_niche[cell]
        if cell in cell_to_niche
        else non_niche_value
        for cell in adata.obs.index
    ]
    
    return adata_neighbors_mask
