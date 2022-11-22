import itertools
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances


def kernel_density(
    adata: AnnData,
    groupby: Optional[str] = "all",
    groups: Optional[Union[str, List[str]]] = "all",
    groupname: Optional[str] = None,
    basis: Optional[str] = "spatial",
    bandwidth: Optional[float] = 1.0,
    resolution: Optional[float] = 1.0,
    separate_groups=False,
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
    groups
        Groups in `adata.obs[groupby]` that contribute towards density.
    groupname
       Column in `adata.obs` to assign density values to.
    basis
        Coordinates in `adata.obsm[X_{basis}]` to use. Defaults to `spatial`.
    bandwidth
        Bandwidth for kernel density estimation.
    resolution
        How small each gridpoint is. Number of points per row/column = resolution * 10
    separate_groups
        Whether groups should be calculated separately or aggregated.

    Returns
    -------
    If `separate_groups = True`,  a dictionary mapping groups to keys in `adata.obs` is
    returned. If `separate_groups = False`, a single key in `adata.obs` is returned.
    """
    # Convert groups into a list of values in adata.obs[groupby]
    if groupby != "all" and (type(groups) == str or groups is None):
        if groups == "all" or groups is None:
            groups = adata.obs[groupby].cat.categories
        else:
            groups = [groups]
    if groupname is None:
        groupname = "_".join(groups)

    # Multiply base resolution by 10 to get number of bins per row/column
    res = int(10 * resolution)

    # Calculate bounds of spatial coordinates, x and y increments for each bin, and number of bins
    [x_coords, y_coords] = adata.obsm[f"X_{basis}"].transpose()
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
    cell_to_bin = [location_to_bin(x, y) for (x, y) in adata.obsm[f"X_{basis}"]]

    # Counts the number of cells in group for each bins
    if separate_groups:
        group_to_bin_counts = defaultdict(lambda: np.zeros(num_bins, dtype=int))
        # Iterate over each cell, counting the number of each group in each bin
        for (cell_group, bin_id) in zip(adata.obs[groupby], cell_to_bin):
            group_to_bin_counts[cell_group][bin_id] += 1 if cell_group in groups else 0
    else:
        bin_counts = np.zeros(num_bins, dtype=int)
        if groupby == "all":
            for bin_id, count in Counter(cell_to_bin).items():
                bin_counts[bin_id] = count
        else:
            # Iterate over each cell, counting the number in groups in each bin
            for (cell_group, bin_id) in zip(adata.obs[groupby], cell_to_bin):
                bin_counts[bin_id] += 1 if groupby == "all" or cell_group in groups else 0

    # Calculate the density of each kernel, and scale to be between 0 and 1
    if separate_groups:
        group_to_bin_densities = {
            cell_group: np.matmul(kernel, bin_counts) / num_bins
            for (cell_group, bin_counts) in group_to_bin_counts.items()
        }
        mins = [min(densities) for densities in group_to_bin_densities.values()]
        maxes = [max(densities) for densities in group_to_bin_densities.values()]
        group_to_bin_densities = {
            cell_group: [(d - min) / (max - min) for d in group_to_bin_densities[cell_group]]
            for min, max, cell_group in zip(mins, maxes, group_to_bin_densities.keys())
        }
    else:
        bin_densities = np.matmul(kernel, bin_counts) / num_bins
        min_density, max_density = min(bin_densities), max(bin_densities)
        if min_density != max_density:
            bin_densities = [(d - min_density) / (max_density - min_density) for d in bin_densities]

    # Map kernel densities back to cells and assign to a column in obs
    if separate_groups:
        cell_densities = {
            cell_group: [group_to_bin_densities[cell_group][bin_id] for bin_id in cell_to_bin] for cell_group in groups
        }
        for (cell_group, densities) in cell_densities.items():
            adata.obs[f"{groupby}_density_{cell_group}"] = densities
        return {cell_group: f"{groupby}_density_{cell_group}" for cell_group in cell_densities.keys()}
    else:
        cell_densities = [bin_densities[bin_id] for bin_id in cell_to_bin]
        adata.obs[f"{groupby}_density_{groupname}"] = cell_densities
        return f"{groupby}_density_{groupname}"
