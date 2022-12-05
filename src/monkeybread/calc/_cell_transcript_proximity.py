from typing import List, Optional

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
    min_x = min(adata[cells].obsm["X_spatial"].T[0] - 0.5 * adata[cells].obs["width"])
    max_x = max(adata[cells].obsm["X_spatial"].T[0] + 0.5 * adata[cells].obs["width"])
    min_y = min(adata[cells].obsm["X_spatial"].T[1] - 0.5 * adata[cells].obs["height"])
    max_y = max(adata[cells].obsm["X_spatial"].T[1] + 0.5 * adata[cells].obs["height"])
    transcripts = adata.uns["transcripts"]

    # Subset by genes if provided
    if genes is not None:
        transcripts = transcripts.loc[transcripts["gene"].isin(genes), :]

    # Subset transcripts by location based on x and y boundaries
    transcripts = transcripts[
        (min_x <= transcripts["global_x"])
        & (max_x >= transcripts["global_x"])
        & (min_y <= transcripts["global_y"])
        & (max_y >= transcripts["global_y"])
    ]
    return transcripts
