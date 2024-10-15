import os
from typing import Dict, Optional

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import zarr

# Extended deafult paths to incldue cell and nuceleus boundaries, and morphology images
default_paths = {
    "cache": "adata.h5ad",
    "counts": "cells.csv.gz",
    "coordinates": "cell_metadata.csv",
    "transcripts": "transcripts.zarr.zip",
    "cell_boundaries": "cell_boundaries.csv.gz",
    "nucleus_boundaries": "nucleus_boundaries.csv.gz",
    "morphology": "morphology.ome.tiff",
}


def load_xenium(
    folder: Optional[str] = ".",
    transcript_locations: Optional[bool] = None,
    paths: Optional[Dict[str, str]] = None,
) -> ad.AnnData:
    """Loads data from Xenium, in accordance to the folder structure of the FFPE data release.

    Parameters
    ----------
    folder : str 
        A path from the current working directory to the folder containing the Xenium data.
    transcript_locations: bool
        Whether to include transcript location data in the final `AnnData` object.
    paths : dict
        Path to each of the files output by Xenium. Defaults are provided for common files.

    Returns
    -------
    An annotated data matrix containing spatial data from Xenium.
    """
    # Add default paths to path dictionary, maintaining overrides
    if paths is None:
        paths = {}
    for k, v in default_paths.items():
        if k not in paths:
            paths[k] = v

    # Load the cell-by-gene count matrix using Scanpy
    counts = sc.read(f"{folder}/{paths['counts']}", first_column_names=True, cache=True)
    
    # Load the cell coordinates metadata
    coordinates = pd.read_csv(f"{folder}/{paths['coordinates']}")
    coordinates = coordinates.rename({"Unnamed: 0": "cell_id"}, axis=1)
    
    # Slice the data matrix to match the available coordinates (if some cells are filtered)
    data = counts[coordinates.cell_id]  # Slice data by coordinate index
    
    # Add spatial coordinates to the AnnData object
    data.obsm["X_spatial"] = coordinates[["center_x", "center_y"]].to_numpy()
    data.obs["width"] = coordinates["max_x"].to_numpy() - coordinates["min_x"].to_numpy()
    data.obs["height"] = coordinates["max_y"].to_numpy() - coordinates["min_y"].to_numpy()
    data.obs["fov"] = coordinates["fov"].to_numpy()
    data.obs.index = [str(x) for x in  data.obs.index] # Ensure cell indices are strings

    # Read transcripts locations (support for both CSV and Zarr)
    if transcript_locations or (transcript_locations is None and os.path.exists(f"{folder}/{paths['transcripts']}")):
        transcript_file = f"{folder}/{paths['transcripts']}"
        if transcript_file.endswith("zarr.zip"):
            transcript_data = zarr.open(transcript_file, mode='r')
            # Here we assume zarr structure includes a 'gene', 'global_x', 'global_y' keys
            transcript_df = pd.DataFrame({
                'gene': transcript_data['gene'][:],
                'global_x': transcript_data['global_x'][:],
                'global_y': transcript_data['global_y'][:]
            })
        else:
            transcript_df = pd.read_csv(transcript_file, index_col=0, usecols=["Unnamed: 0", "gene", "global_x", "global_y"])
        data.uns["transcripts"] = transcript_df
        
    # Load cell boundaries
    if os.path.exists(f"{folder}/{paths['cell_boundaries']}"):
        cell_boundaries = pd.read_csv(f"{folder}/{paths['cell_boundaries']}")
        data.obs["cell_boundaries"] = cell_boundaries
    
    # Load nucleus boundaries
    if os.path.exists(f"{folder}/{paths['nucleus_boundaries']}"):
        nucleus_boundaries = pd.read_csv(f"{folder}/{paths['nucleus_boundaries']}")
        data.obs["nucleus_boundaries"] = nucleus_boundaries

    # Load morphology image
    if os.path.exists(f"{folder}/{paths['morphology']}"):
        morphology = tifffile.imread(f"{folder}/{paths['morphology']}")
        data.uns["morphology"] = morphology
        
    # Set raw data (preprocessed, but useful for further analyses)
    data.raw = data

    return data

# Example usage:
# adata = load_xenium(folder="path/to/xenium/data")