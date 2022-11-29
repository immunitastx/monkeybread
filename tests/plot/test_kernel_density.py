import monkeybread as mb
from tests.plot.conftest import FIGS, ROOT

KERNEL_DENSITY_ROOT = ROOT / "kernel_density"
KERNEL_DENSITY_FIGS = FIGS / "kernel_density"


def test_kernel_density_one_group(ct3_sample, image_comparer):
    density = mb.calc.kernel_density(ct3_sample, groupby="cell_type", groups="ct1")
    mb.plot.kernel_density(ct3_sample, density, show=False, spot_size=2000)

    save_and_compare_images = image_comparer(KERNEL_DENSITY_ROOT, KERNEL_DENSITY_FIGS, tol=15)
    save_and_compare_images("one_group")


def test_kernel_density_all_groups(ct3_sample, image_comparer):
    density = mb.calc.kernel_density(ct3_sample, groupby="cell_type", groups="all")
    mb.plot.kernel_density(ct3_sample, density, show=False, spot_size=2000)

    save_and_compare_images = image_comparer(KERNEL_DENSITY_ROOT, KERNEL_DENSITY_FIGS, tol=15)
    save_and_compare_images("all_groups")


def test_kernel_density_all_groups_split(ct3_sample, image_comparer):
    density = mb.calc.kernel_density(ct3_sample, groupby="cell_type", groups="all", separate_groups=True)
    fig = mb.plot.kernel_density(ct3_sample, density, show=False, spot_size=200)
    fig.set_size_inches((6.4, 1.6))

    save_and_compare_images = image_comparer(KERNEL_DENSITY_ROOT, KERNEL_DENSITY_FIGS, tol=15)
    save_and_compare_images("all_groups_split")
