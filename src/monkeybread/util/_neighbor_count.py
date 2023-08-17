from typing import Dict, Set


def neighbor_count(cell_to_neighbors: Dict[str, Set[str]], group1: Set[str], group2: Set[str]) -> int:
    """Counts neighbors observed by :func:`monkeybread.calc.cell_neighbors`.

    Sums the number of unique neighbor-pairs given the neighbor dictionary and a set of
    of cells in each of the two groups.

    Parameters
    ----------
    cell_to_neighbors
        A dictionary mapping from cell indices to other cell indices, as returned by
        :func:`monkeybread.calc.cell_neighbors`.
    group1
        Cell indices corresponding to `group1`.
    group2
        Cell indices corresponding to `group2`.

    Returns
    -------
    The number of unique neighbor-pairs found between a cell in `group1` and a cell
    in `group2`.
    """
    # Sum number of g1-g2 neighbors, then subtract half of the double counting that occurs when cells
    # are in both g1 and g2
    counts = sum(
        0 
        if k not in group1 
        else sum(v in group2 for v in values) 
        for k, values in cell_to_neighbors.items()
    )
    double_counted = sum(
        sum(
            k in group1 and k in group2 and v in group1 and v in group2 
            for v in values
        )
        for k, values in cell_to_neighbors.items()
    )
    return counts - int(0.5 * double_counted)
