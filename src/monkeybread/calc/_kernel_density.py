from anndata import AnnData
from typing import Union, Optional, List, Dict
from collections import defaultdict, Counter
from sklearn.metrics import pairwise_distances
import numpy as np


def kernel_density(
    adata: AnnData,
    groupby: Optional[str] = "all",
    group: Optional[Union[str, List[str]]] = "all",
    groupname: Optional[str] = None,
    bandwidth: Optional[float] = 1.0,
    resolution: Optional[float] = 1.0,
    separate_groups = False
) -> Union[str, Dict[str, str]]:
    """Calculates kernel density estimation on large cell quantities.

    Instead of a single-cell resolution for density, a grid is overlayed onto the spatial
    coordinates, and each cell is assigned to its nearest gridpoint. Each gridpoint is then used to
    calculate kernel density, and the density for each gridpoint is assigned to each cell at that
    spot. Fineness of the grid can be modified using the `resolution` parameter. Density will be
    scaled between 0 and 1 corresponding to the minimum and maximum density, respectively.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        A categorical column in `obs` to use for density calculations. If omitted, each cell will
        contribute equally towards density calculations.
    group
        Groups in `adata.obs[groupby]` that contribute towards density.
    groupname
       Column in `adata.obs` to assign density values to.
    bandwidth
        Bandwidth for kernel density estimation.
    resolution
        How small each gridpoint is. Number of points per row/column = resolution * 10
    separate_groups
        Whether groups should be calculated separately or aggregated.

    Returns
    -------
    densities
        If `separate_groups = True`,  a dictionary mapping groups to keys in `adata.obs` is
        returned. If `separate_groups = False`, a single key in `adata.obs` is returned.
    """
    # Convert group into a list of values in adata.obs[groupby]
    if groupby != "all" and (type(group) == str or group is None):
        if group != "all":
            group = [group]
        else:
            group = adata.obs[groupby].cat.categories
    if groupname is None:
        groupname = '_'.join(group)

    # Multiply base resolution by 10 to get number of bins per row/column
    res = int(10 * resolution)

    # Calculate bounds of spatial coordinates, x and y increments for each bin, and number of bins
    [(x_min, x_max), (y_min, y_max)] = [
        (min(x) - 0.01, max(x) + 0.01) for x in adata.obsm["X_spatial"].transpose()
    ]
    (x_inc, y_inc) = ((x_max - x_min) / res, (y_max - y_min) / res)
    num_bins = int((x_max - x_min) / x_inc) * int((y_max - y_min) / y_inc)

    # Calculate distance from the center of each bin to the center of every other bin
    distances = pairwise_distances([(x_min + x_inc * (i + 1) / 2, y_min + y_inc * (j + 1) / 2)
                                    for i in range(res) for j in range(res)])

    # Calculate kernel based on distances and bandwidth
    kernel = 1 / np.exp(np.square(distances / bandwidth))

    # Determine the kernel number of each cell based on its location
    kernel_locations = [
        int((x - x_min) / x_inc) * int((x_max - x_min) / x_inc) + int((y - y_min) / y_inc)
        for (x, y) in adata.obsm["X_spatial"]]

    # Counts the number of cells in group for each kernel
    if separate_groups:
        kernel_counts = defaultdict(lambda: np.zeros(num_bins, dtype = int))
        for (cell_group, location) in zip(adata.obs[groupby], kernel_locations):
            kernel_counts[cell_group][location] += 1 if cell_group in group else 0
    else:
        kernel_counts = np.zeros(num_bins, dtype = int)
        if groupby == "all":
            for location, count in Counter(kernel_locations).items():
                kernel_counts[location] = count
        else:
            for (cell_group, location) in zip(adata.obs[groupby], kernel_locations):
                kernel_counts[location] += 1 if groupby == "all" or cell_group in group else 0

    # Calculate the density of each kernel, and scale to be between 0 and 1
    if separate_groups:
        kernel_densities = {
            cell_group: np.matmul(kernel, value) / num_bins for (cell_group, value) in
            kernel_counts.items()
        }
        min_density = min(min(densities) for densities in kernel_densities.values())
        max_density = max(max(densities) for densities in kernel_densities.values())
        if min_density != max_density:
            kernel_densities = {
                cell_group: [(d - min_density) / (max_density - min_density) for d in
                             kernel_densities[cell_group]]
                for (cell_group, value) in kernel_densities.items()
            }
    else:
        kernel_densities = np.matmul(kernel, kernel_counts) / num_bins
        min_density, max_density = min(kernel_densities), max(kernel_densities)
        if min_density != max_density:
            kernel_densities = [(d - min_density) / (max_density - min_density) for d in
                                kernel_densities]

    # Map kernel densities back to cells and assign to a column in obs
    if separate_groups:
        cell_densities = {
            cell_group: [kernel_densities[cell_group][location] for location
                                  in kernel_locations]
            for cell_group in group
        }
        for (cell_group, densities) in cell_densities.items():
            adata.obs[f"{groupby}_density_{cell_group}"] = densities
        return {
            cell_group: f"{groupby}_density_{cell_group}" for cell_group in cell_densities.keys()
        }
    else:
        cell_densities = [kernel_densities[location] for location in kernel_locations]
        adata.obs[f"{groupby}_density_{groupname}"] = cell_densities
        return f"{groupby}_density_{groupname}"
