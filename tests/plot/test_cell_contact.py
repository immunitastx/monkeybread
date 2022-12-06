import random

import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

CELL_CONTACT_ROOT = ROOT / "cell_contact"
CELL_CONTACT_FIGS = FIGS / "cell_contact"


def test_cell_contact_embedding(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    mb.plot.cell_contact_embedding(dense_sample, cell_contact, group="cell_type", show=False)

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("embedding")


def test_cell_contact_embedding_no_label(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    mb.plot.cell_contact_embedding(dense_sample, cell_contact, show=False)

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("embedding_no_label")


def test_cell_contact_embedding_zoom(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    fig = mb.plot.cell_contact_embedding_zoom(
        dense_sample,
        cell_contact,
        left_pct=0.1,
        top_pct=0.2,
        height_pct=0.47,
        width_pct=0.3,
        group="cell_type",
        show=False,
    )

    fig.set_size_inches((6.4, 3.2))

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("embedding_zoom")


def test_cell_contact_histplot(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(dense_sample, groupby="cell_type", group1="ct1", group2="ct2", radius=1)
    random.seed(0)
    expected_contacts = mb.stat.cell_contact(
        dense_sample,
        groupby="cell_type",
        group1="ct1",
        group2="ct2",
        actual_contact=cell_contact,
        contact_radius=1,
        perm_radius=1,
        n_perms=10,
    )
    mb.plot.cell_contact_histplot(
        dense_sample, groupby="cell_type", contacts=cell_contact, expected_contacts=expected_contacts, show=False
    )

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("histplot")


def test_cell_contact_heatmap_counts(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(
        dense_sample, groupby="cell_type", group1=["ct1", "ct2"], group2=["ct1", "ct2"], radius=1
    )
    mb.plot.cell_contact_heatmap(dense_sample, groupby="cell_type", contacts=cell_contact, show=False)

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("heatmap_counts")


def test_cell_contact_heatmap_pvals(dense_sample, image_comparer):
    cell_contact = mb.calc.cell_contact(
        dense_sample, groupby="cell_type", group1=["ct1", "ct2"], group2=["ct1", "ct2"], radius=1
    )
    random.seed(0)
    expected_contacts = mb.stat.cell_contact(
        dense_sample,
        groupby="cell_type",
        group1=["ct1", "ct2"],
        group2=["ct1", "ct2"],
        actual_contact=cell_contact,
        contact_radius=1,
        perm_radius=1,
        n_perms=10,
        split_groups=True,
    )
    mb.plot.cell_contact_heatmap(dense_sample, groupby="cell_type", expected_contacts=expected_contacts, show=False)

    save_and_compare_images = image_comparer(CELL_CONTACT_ROOT, CELL_CONTACT_FIGS, tol=15)
    save_and_compare_images("heatmap_pvals")
