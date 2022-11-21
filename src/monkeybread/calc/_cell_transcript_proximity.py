from typing import List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData


def cell_transcript_proximity(
    adata: AnnData,
    cells: List[str],
    genes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Detects transcripts in proximity or inside a given list of cells.

    Parameters
    ----------
    adata
        Annotated data matrix.
    cells
        A list of cell ids to determine proximity.
    genes
        An optional list of genes to filter. If not provided, all transcripts will be included.

    Returns
    -------
    Transcripts in proximity to the list of cells provided, with a column for gene, x, and y.
    """
    # Check for presence of bounds/transcripts
    if "bounds" not in adata.obs or "transcripts" not in adata.uns:
        raise ValueError(
            "Bounds/transcripts not found. To use this function, you must have loaded\
         the AnnData object with both detected_transcripts.csv and cell_bounds/ present."
        )

    # Extract bounds
    cell_bounds = adata.obs["bounds"][cells].apply(np.transpose)
    transcripts = adata.uns["transcripts"]

    # Find x and y boundaries of cells
    mins = np.min(np.hstack(cell_bounds), axis=1)
    maxes = np.max(np.hstack(cell_bounds), axis=1)

    # Subset by genes if provided
    if genes is not None:
        transcripts = transcripts.loc[transcripts["gene"].isin(genes), :]

    # Subset transcripts by location based on x and y boundaries
    transcripts = transcripts[
        (mins[0] <= transcripts["global_x"])
        & (maxes[0] >= transcripts["global_x"])
        & (mins[1] <= transcripts["global_y"])
        & (maxes[1] >= transcripts["global_y"])
    ]
    return transcripts
