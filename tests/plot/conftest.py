# Source: https://github.com/scverse/scanpy/blob/master/scanpy/tests/conftest.py
from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib import pyplot
from matplotlib.testing.compare import compare_images, make_test_filename

mpl.use("agg")

# In case pytest-nunit is not installed, defines a dummy fixture
try:
    import pytest_nunit  # noqa

except ModuleNotFoundError:

    @pytest.fixture
    def add_nunit_attachment(request):
        def noop(file, description):
            pass

        return noop


ROOT = Path(__file__).parent / "_images"
FIGS = Path(__file__).parent / "figures"


@pytest.fixture
def check_same_image(add_nunit_attachment):
    def _(pth1, pth2, *, tol: int, basename: str = ""):
        def fmt_descr(descr):
            if basename != "":
                return f"{descr} ({basename})"
            else:
                return descr

        pth1, pth2 = Path(pth1), Path(pth2)
        try:
            result = compare_images(str(pth1), str(pth2), tol=tol)
            assert result is None, result
        except AssertionError as e:
            diff_pth = make_test_filename(pth2, "failed-diff")
            add_nunit_attachment(str(pth1), fmt_descr("Expected"))
            add_nunit_attachment(str(pth2), fmt_descr("Result"))
            if Path(diff_pth).is_file():
                add_nunit_attachment(str(diff_pth), fmt_descr("Difference"))
            raise e

    return _


@pytest.fixture
def image_comparer(check_same_image):
    def make_comparer(path_expected: Path, path_actual: Path, *, tol: int):
        def save_and_compare(basename, tol=tol):
            path_actual.mkdir(parents=True, exist_ok=True)
            out_path = path_actual / f"{basename}.png"
            pyplot.tight_layout()
            pyplot.savefig(out_path, dpi=100)
            pyplot.close()
            check_same_image(path_expected / f"{basename}.png", out_path, tol=tol)

        return save_and_compare

    return make_comparer
