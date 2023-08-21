"""
Visualization of spatial distributions of cell types via kernel density estimation.
"""

import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union
from tqdm import tqdm
import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances
from scipy.spatial import cKDTree
from scipy.sparse import dok_matrix 

def _sparse_distance_matrix(data, distance_threshold):
    N = data.shape[0]
    
    # Build a KD-Tree for fast nearest neighbor search
    kdtree = cKDTree(data)
    
    # Initialize a sparse matrix to store the distance matrix
    distance_matrix = dok_matrix((N, N), dtype=float)
    
    # Query the KD-Tree for neighbors within the distance threshold
    # and set the corresponding entries in the distance matrix
        
    for i in tqdm(range(N)):
        neighbors = kdtree.query_ball_point(data[i], distance_threshold)
        distances = np.linalg.norm(data[i] - data[neighbors], axis=1)
        distance_matrix[i, neighbors] = distances
        distance_matrix[neighbors, i] = distances
    return distance_matrix


def cell_density(
    adata: AnnData,
    groupby: Optional[str] = "all",
    groups: Optional[Union[str, List[str]]] = "all",
    groupname: Optional[str] = None,
    basis: Optional[str] = "X_spatial",
    bandwidth: Optional[float] = 1.0,
    approx: Optional[bool] = True,
    resolution: Optional[float] = 1.0,
    radius_threshold: Optional[float] = 250
) -> Union[str, Dict[str, str]]:
    """Calculates the spatial distribution of cells of a given cell type using kernel density 
    estimation. 
    
    This can be computed via two approximations: the first approximation is closest to the true
    kernel density estimate, however, the kernel between pairs of cells are zeroed out if they
    are greater than `radius_threshold` from eachother. This enables the kernel estimation to run
    on sparse matrices.

    The second approximation is faster and can be run if `approx` is set to true. A grid is 
    overlayed onto the spatial coordinates, and each cell is assigned to its nearest gridpoint. 
    Each gridpoint is then used in a kernel density estimation calculation. The density value for 
    each gridpoint is then assigned back to each cell. Fineness of the grid can be modified using 
    the `resolution` parameter where a high resolution leads to a more accurate calculation of 
    density for each cell.  Final density values are scaled between 0 and 1 corresponding to the 
    minimum and maximum density, respectively.

    Parameters
    ----------
    adata
        Annotated data matrix
    groupby
        A categorical column in `obs` to use for density calculations. If omitted, each cell will
        contribute equally towards density calculations
    groups
        Groups in `adata.obs[groupby]` that contribute towards density
    groupname
       Column in `adata.obs` to assign density values to.
    basis
        Coordinates in `adata.obsm[{basis}]` to use. Defaults to `spatial`
    bandwidth
        Bandwidth for kernel density estimation
    approx
        If true, approximate the kernel density estimation by grouping cells into bins
    resolution
        How small each gridpoint is. Number of points per row/column = resolution * 10.
        Used only if `approx` is `True`
    radius_threshold
        Sparsifies the pairwise distance matrix to only consider cells if they are within
        this threshold distance from eachother. Only used if `approx` is False.

    Returns
    -------
    The column name `adata.obs` storing the density values for each cell.
    """
    # Convert groups into a list of values in adata.obs[groupby]
    if groupby != "all" and (type(groups) == str or groups is None):
        if groups == "all" or groups is None:
            groups = adata.obs[groupby].cat.categories
        else:
            groups = [groups]
    if groupname is None:
        groupname = "_".join(groups)

    if approx:
        # Multiply base resolution by 10 to get number of bins per row/column
        res = int(10 * resolution)

        # Calculate bounds of spatial coordinates, x and y increments for each bin, and number of bins
        [x_coords, y_coords] = adata.obsm[basis].transpose()
        (x_min, x_max) = (min(x_coords) - 0.01, max(x_coords) + 0.01)
        (y_min, y_max) = (min(y_coords) - 0.01, max(y_coords) + 0.01)
        (x_inc, y_inc) = ((x_max - x_min) / res, (y_max - y_min) / res)
        num_bins = int(res**2)

        # Calculate distance from the center of each bin to the center of every other bin
        x_bin_centers = [x_min + x_inc * (i + 1) / 2 for i in range(res)]
        y_bin_centers = [y_min + y_inc * (j + 1) / 2 for j in range(res)]
        bin_centers = list(itertools.product(x_bin_centers, y_bin_centers))
        bin_distances = pairwise_distances(bin_centers)

        # Calculate kernel based on distances and bandwidth
        kernel = 1 / np.exp(np.square(bin_distances / bandwidth))

        # Determine the bin each cell belongs to based on its location
        # Bin number corresponds to index in bin_distancesx_c
        location_to_bin = lambda x, y: int((x - x_min) / x_inc) * res + int((y - y_min) / y_inc)
        cell_to_bin = [location_to_bin(x, y) for (x, y) in adata.obsm[basis]]

        bin_counts = np.zeros(num_bins, dtype=int)
        if groupby == "all":
            for bin_id, count in Counter(cell_to_bin).items():
                bin_counts[bin_id] = count
        else:
            # Iterate over each cell, counting the number in groups in each bin
            for (cell_group, bin_id) in zip(adata.obs[groupby], cell_to_bin):
                bin_counts[bin_id] += 1 if groupby == "all" or cell_group in groups else 0

        # Compute bin densities
        bin_densities = np.matmul(kernel, bin_counts)
        
        # Scale between zero and one
        min_density, max_density = min(bin_densities), max(bin_densities)
        if min_density != max_density:
            bin_densities = [
                (d - min_density) / (max_density - min_density) 
                for d in bin_densities
            ]

        # Map kernel densities back to cells and assign to a column in obs
        cell_densities = [bin_densities[bin_id] for bin_id in cell_to_bin]
    else:
        # Compute kernel matrix
        distances = _sparse_distance_matrix(adata.obsm[basis], radius_threshold)
        kernel = np.expm1(-(distances/bandwidth).power(2))
        kernel[kernel.nonzero()] += 1
        
        # Get cell type binary vector
        is_group = np.array(adata.obs[groupby].isin(groups)).astype(int)

        # Compute densities
        densities = kernel.dot(is_group)

        # Scale between zero and one
        cell_densities = (densities - densities.min()) / (densities.max() - densities.min())

    adata.obs[f"{groupby}_density_{groupname}"] = cell_densities
    return f"{groupby}_density_{groupname}"
        


