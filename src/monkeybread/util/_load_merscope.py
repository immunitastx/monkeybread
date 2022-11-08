from typing import Optional, Dict
import os
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import h5py

default_paths = {
    "cache": "adata.h5ad",
    "counts": "cell_by_gene.csv",
    "coordinates": "cell_metadata.csv",
    "cell_bounds": "cell_boundaries/",
    "transcripts": "detected_transcripts.csv"
}


def load_merscope(
    folder: Optional[str] = ".",
    use_cache: Optional[str] = None,
    cell_bounds: Optional[bool] = None,
    transcript_locations: Optional[bool] = None,
    paths: Optional[Dict[str, str]] = None
) -> ad.AnnData:
    """Loads data from MERSCOPE, in accordance to the folder structure of the FFPE data release.

    Parameters
    ----------
    folder
        A path from the current working directory to the folder containing the MERSCOPE data.
    use_cache
        How to use a cached AnnData object. If None, does not use a cached object. If "all", only
        reads from the cached object and does not read from other files. If "spatial", reads from
        the cached object and adds spatial data in regards to cell boundaries and transcripts.
    cell_bounds
        Whether or not to include cell boundaries in a column in the resulting AnnData object.
        Default is to include if the folder exists.
    transcript_locations
        Whether or not to include transcript locations in `.uns['transcripts']` in the resulting
        AnnData object. Default is to include if the file exists.
    paths
        Paths to each of the files output by MERSCOPE. Defaults are `cache: 'adata.h5ad'`,
        `counts: 'cell_by_gene.csv'`, `coordinates: 'cell_metadata.csv'`,
        `cell_bounds: 'cell_boundaries/'`, and `transcripts: 'detected_transcripts.csv'`. Default
        values will be filled in if an incomplete dictionary is provided.

    Returns
    -------
    adata
        An annotated data matrix containing spatial data from MERSCOPE.
    """
    if paths is None:
        paths = {}
    for k, v in default_paths.items():
        if k not in paths:
            paths[k] = v
    data: ad.AnnData
    if use_cache is not None:
        if use_cache == "all":
            return ad.read(f"{folder}/{paths['cache']}")
        elif use_cache == "spatial":
            data = ad.read(f"{folder}/{paths['cache']}")
        else:
            raise ValueError("use_cache must be None, 'all', or 'spatial'")
    else:
        counts = sc.read(f"{folder}/{paths['counts']}", first_column_names = True, cache = True)
        coordinates = pd.read_csv(f"{folder}/{paths['coordinates']}")
        coordinates = coordinates.rename({"Unnamed: 0": "cell_id"}, axis = 1)
        data = counts[coordinates.cell_id]
        data.obsm["X_spatial"] = coordinates[["center_x", "center_y"]].to_numpy()
        data.obs["width"] = coordinates["max_x"].to_numpy() - coordinates["min_x"].to_numpy()
        data.obs["height"] = coordinates["max_y"].to_numpy() - coordinates["min_y"].to_numpy()
        data.obs["fov"] = coordinates["fov"].to_numpy()
    if cell_bounds or (cell_bounds is None and
                       os.path.exists(f"{folder}/{paths['cell_bounds']}")):
        data.obs["bounds"] = np.array(data.obs.shape[0], dtype = object)
        for fov in pd.Categorical(data.obs["fov"]).categories:
            with h5py.File(f"{folder}/{paths['cell_bounds']}/feature_data_{fov}.hdf5",
                           "r") as f:
                for cell_id in data.obs.index[data.obs["fov"] == fov]:
                    data.obs["bounds"][cell_id] = np.array(f[
                        f"featuredata/{cell_id}/zIndex_0/p_0/coordinates"
                    ][0])
    if transcript_locations or (transcript_locations is None and
                                os.path.exists(f"{folder}/{paths['transcripts']}")):
        data.uns["transcripts"] = pd.read_csv(f"{folder}/{paths['transcripts']}", index_col = 0)
    data.raw = data
    return data
