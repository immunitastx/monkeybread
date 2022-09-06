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
    use_cache: Optional[bool] = None,
    cell_bounds: Optional[bool] = None,
    transcript_locations: Optional[bool] = None,
    paths: Optional[Dict[str, str]] = None
) -> ad.AnnData:
    if paths is None:
        paths = default_paths
    if use_cache or (use_cache is None and os.path.exists(f"{folder}/{paths['cache']}")):
        return ad.read(f"{folder}/{paths['cache']}")
    else:
        counts = sc.read(f"{folder}/{paths['counts']}", first_column_names = True, cache = True)
        coordinates = pd.read_csv(f"{folder}/{paths['coordinates']}")
        coordinates = coordinates.rename({"Unnamed: 0": "cell_id"}, axis = 1)
        data = counts[coordinates.cell_id]
        data.obsm["X_spatial"] = coordinates[["center_x", "center_y"]].to_numpy()
        data.obs["width"] = coordinates["max_x"].to_numpy() - coordinates["min_x"].to_numpy()
        data.obs["height"] = coordinates["max_y"].to_numpy() - coordinates["min_y"].to_numpy()
        if cell_bounds or (cell_bounds is None and
                           os.path.exists(f"{folder}/{paths['cell_bounds']}")):
            data.obs["bounds"] = np.array(data.obs.shape[0])
            for fov in pd.Categorical(data.obs["fov"]).categories:
                with h5py.File(f"{folder}/{paths['cell_bounds']}/feature_data_{fov}.hdf5",
                               "r") as f:
                    for cell_id in data.obs.index[data.obs["fov"] == fov]:
                        data.obs["bounds"][cell_id] = f[
                            f"featuredata/{cell_id}/zIndex_0/p_0/coordinates"
                        ][0],
        if transcript_locations or (transcript_locations is None and
                                    os.path.exists(f"{folder}/{paths['transcripts']}")):
            data.uns["transcripts"] = pd.read_csv(f"{folder}/{paths['transcripts']}", index_col = 0)
        data.raw = data
        return data
