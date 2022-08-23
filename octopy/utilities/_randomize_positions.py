import math
import random as rand
from anndata import AnnData
from tqdm.notebook import tqdm
import numpy as np
from typing import Optional, Tuple
from numba import njit, prange


@njit(parallel = True)
def randomize_positions(
    adata: AnnData,
    radius: float,
    progress_bar: Optional[bool] = False,
    return_positions: Optional[bool] = False
) -> Optional[np.ndarray]:
    """Randomizes positions within a given radius.

    Parameters
    ----------
    adata
        Annotated data matrix. Coordinates are taken from `adata.obsm["X_spatial"]`
    radius
        Radius to randomize within. Measured in same units as coordinates.
    progress_bar
        Whether to show a progress bar.
    return_positions
        Whether to return the randomized positions or assign them to
        `adata.obsm["X_spatial_random"]`.

    Returns
    -------
    random_coords
        If `return_positions=True`, returns a list of the randomized coordinates corresponding to
        the coordinates in `adata.obsm["X_spatial"]`. Otherwise, these coordinates are assigned to
        `adata.obsm["X_spatial_random"].
    """

    bar = lambda x: x
    if progress_bar:
        bar = tqdm
    transformations = np.array(adata.shape[0])
    for i in bar(prange(adata.shape[0])):
        transformations[i] = [math.sin(math.pi * rand.random()) * radius,
                              math.cos(math.pi * rand.random()) * radius]
    if return_positions:
        return transformations + adata.obsm["X_spatial"]
    else:
        adata.obsm["X_spatial_random"] = transformations + adata.obsm["X_spatial"]
