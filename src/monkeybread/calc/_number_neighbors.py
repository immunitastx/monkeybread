from typing import Dict, List, Optional
from anndata import AnnData
import pandas as pd

import monkeybread as mb

def number_of_neighbors(
        adata: AnnData,
        groupby: str,
        query_groups: List[str],
        reference_group: str,
        radius: Optional[float] = 75
    ):
    """Given a set of query groups of cells, group_1, group_2, ... group_N, and a reference group of cells,
    calculate the number of reference cells within the neighborhood of each cell in the query groups.

    For example, our query groups might be "B cell" and "T cell" and our reference group might be "tumor cell".
    This function computes the number of tumor cells in the neighborhood around each B cell and each T cell.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        Key for column of `adata.obs` storing the cell type column used to group cells.
    query_groups
        List of query cell types in `adata.obs[{groupby}]`
    reference_group
        Name of the query cell type in `adata.obs[{groupby}]`
    radius
        Radius of the neighborhoods centered around each query cell.

    Returns
    -------
    A `pd.DataFrame` storing the number of neighbors that are of the reference group to each cell
    in the query groups.
    """

    # Compute cell neighbors for each query group 
    query_to_cell_to_neighbors = {
        query: mb.calc.cell_neighbors(
            adata, 
            groupby=groupby,
            group1=query, 
            group2=reference_group, 
            radius=radius
        )
        for query in query_groups
    }

    # Create dataframe storing the results
    da = []
    for query, cell_to_neighbors in query_to_cell_to_neighbors.items():
        for cell, neighbors in cell_to_neighbors.items():
            da.append((cell, query, len(neighbors)))
    df_plot = pd.DataFrame(
        data=da,
        columns=['cell', 'group', 'num_neighbors']
    )
    df_plot = df_plot.set_index('cell')

    return df_plot



def number_of_neighbors_from_masks(
        adata: AnnData,
        query_to_mask: Dict[str, List[bool]],
        reference_mask: List[bool],
        radius: Optional[float] = 75
    ):
    """Given a set of query groups of cells, group_1, group_2, ... group_N, and a reference group of cells, 
    calculate the number of reference cells within the neighborhood of each cell in the query groups.

    Groups are specified from masks (i.e., Boolean-valued lists) that specify whether each cell in the 
    dataset belongs to each group.

    For example, our query groups might be "B cell" and "T cell" and our reference group might be "tumor cell".
    This function computes the number of tumor cells in the neighborhood around each B cell and each T cell.

    Parameters
    ----------
    adata
        Annotated data matrix.
    query_to_mask
        A dictionary mapping the name of a query group to a Boolean mask (i.e., Boolean-valued list). Each 
        element of a mask corresponds to an index in `adata`. `True` indicates the cell belongs to the group,
        and `False` indicates the cell does not belong to the group.
    reference_mask
        A mask specifying cells in the reference group.
    radius
        Radius of the neighborhoods centered around each query cell.

    Returns
    -------
    A `pd.DataFrame` storing the number of neighbors that are of the reference group to each cell
    in the query groups.
    """
    # Compute cell neighbors for each query group 
    query_to_cell_to_neighbors = {
        query: mb.calc.cell_neighbors_from_masks(adata, mask, reference_mask, radius=radius)
        for query, mask in query_to_mask.items()
    }

    # Create dataframe storing the results
    da = []
    for query, cell_to_neighbors in query_to_cell_to_neighbors.items():
        for cell, neighbors in cell_to_neighbors.items():
            da.append((cell, query, len(neighbors)))
    df_plot = pd.DataFrame(
        data=da,
        columns=['cell', 'group', 'num_neighbors']
    )
    df_plot = df_plot.set_index('cell')
    
    return df_plot
