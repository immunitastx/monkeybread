import os
from typing import Dict, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

default_paths = {
    "cache": "adata.h5ad",
    "counts": "cell_by_gene.csv",
    "coordinates": "cell_metadata.csv",
    "transcripts": "detected_transcripts.csv",
}


def load_merscope(
    folder: Optional[str] = ".",
    transcript_locations: Optional[bool] = None,
    paths: Optional[Dict[str, str]] = None,
) -> ad.AnnData:
    """Loads data from MERSCOPE, in accordance to the folder structure of the FFPE data release.

    Parameters
    ----------
    folder
        A path from the current working directory to the folder containing the MERSCOPE data.
    paths
        Paths to each of the files output by MERSCOPE. Defaults are `cache: 'adata.h5ad'`,
        `counts: 'cell_by_gene.csv'`, `coordinates: 'cell_metadata.csv'`,
        and `transcripts: 'detected_transcripts.csv'`. Default values will be filled in if 
        an incomplete dictionary is provided.

    Returns
    -------
    An annotated data matrix containing spatial data from MERSCOPE.
    """
    # Add default paths to path dictionary, maintaining overrides
    if paths is None:
        paths = {}
    for k, v in default_paths.items():
        if k not in paths:
            paths[k] = v

    data: ad.AnnData

    counts = sc.read(f"{folder}/{paths['counts']}", first_column_names=True, cache=True)
    coordinates = pd.read_csv(f"{folder}/{paths['coordinates']}")
    coordinates = coordinates.rename({"Unnamed: 0": "cell_id"}, axis=1)
    data = counts[coordinates.cell_id]  # Slice data by coordinate index
    data.obsm["X_spatial"] = coordinates[["center_x", "center_y"]].to_numpy()
    data.obs["width"] = coordinates["max_x"].to_numpy() - coordinates["min_x"].to_numpy()
    data.obs["height"] = coordinates["max_y"].to_numpy() - coordinates["min_y"].to_numpy()
    data.obs["fov"] = coordinates["fov"].to_numpy()
    data.obs.index = [str(x) for x in  data.obs.index] # Ensure cell indices are strings

    # Read transcripts
    if transcript_locations or (transcript_locations is None and os.path.exists(f"{folder}/{paths['transcripts']}")):
        data.uns["transcripts"] = pd.read_csv(
            f"{folder}/{paths['transcripts']}", index_col=0, usecols=["Unnamed: 0", "gene", "global_x", "global_y"]
        )

    data.raw = data
    return data
