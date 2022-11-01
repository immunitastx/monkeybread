import pandas as pd
from anndata import AnnData
from typing import List, Optional
import numpy as np


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
    filtered_transcripts
        Transcripts in proximity to the list of cells provided, with a column for gene, x, and y.
    """
    cell_bounds = adata.obs["bounds"][cells]
    transcripts = adata.uns["transcripts"]
    mins = np.min(np.hstack(*cell_bounds), axis = 1)
    maxes = np.max(np.hstack(*cell_bounds), axis = 1)
    transcript_locations = transcripts[['gene', 'global_x', 'global_y']]
    transcript_locations.rename({"global_x": "x", "global_y": "y"}, axis = 1, inplace = True)
    if genes is not None:
        transcript_locations = transcript_locations.loc[transcript_locations['gene'].isin(genes), :]
    transcript_locations = transcript_locations.loc[[
        mins[0] <= x <= maxes[0] and
        mins[1] <= y <= maxes[1]
        for i, (g, x, y) in transcript_locations.iterrows()], :]
    return transcript_locations
