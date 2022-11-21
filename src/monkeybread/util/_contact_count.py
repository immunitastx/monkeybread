from typing import Dict, Set


def contact_count(contacts: Dict[str, Set[str]], group1: Set[str], group2: Set[str]) -> int:
    """Counts contact observed by :func:`monkeybread.calc.cell_contact`.

    Sums the number of unique contacts given the contact dictionary and ids corresponding to
    cells in each group. Can be used to extract more specific contact counts out of a larger
    dictionary.

    Parameters
    ----------
    contacts
        A dictionary mapping from cell indices to other cell indices, as returned by
        :func:`monkeybread.calc.cell_contact`.
    group1
        Cell indices corresponding to `group1`.
    group2
        Cell indices corresponding to `group2`.

    Returns
    -------
    The number of unique contacts found between `group1` and `group2`.
    """
    # Sum number of g1-g2 contacts, then subtract half of the double counting that occurs when cells
    # are in both g1 and g2
    return sum(0 if k not in group1 else sum(v in group2 for v in values) for k, values in contacts.items()) - int(
        0.5
        * sum(
            sum(k in group1 and k in group2 and v in group1 and v in group2 for v in values)
            for k, values in contacts.items()
        )
    )
