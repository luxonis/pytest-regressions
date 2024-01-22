from pathlib import Path
from typing import Any
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from .data_regression import DataRegressionFixture
    from .dataframe_regression import DataFrameRegressionFixture
    from .ndarrays_regression import NDArraysRegressionFixture
    from .file_regression import FileRegressionFixture
    from .num_regression import NumericRegressionFixture
    from .num_relative_regression import NumericRelativeRegressionFixture
    from .image_regression import ImageRegressionFixture


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("regressions")
    group.addoption(
        "--force-regen",
        action="store_true",
        default=False,
        help="Regenerate regression data files, failing tests with different data.",
    )
    group.addoption(
        "--regen-all",
        action="store_true",
        default=False,
        help="Regenerate all files, letting tests pass (use to regenerate everything in one run).",
    )
    group.addoption(
        "--with-test-class-names",
        action="store_true",
        default=False,
        help="Do not ignore the names of the test classes when composing the name of the regression data files.",
    )
    group.addoption(
        "--output-directory",
        default=None,
        help="Save the outputs of each run in this directory instead, the --output-directory has to be set as well",
        type=str
    )
    group.addoption(
        "--baseline-directory",
        default=None,
        help="Save the baseline of each run in this directory instead, the --baseline-directoy has to be set as well",
    )


@pytest.fixture
def data_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "DataRegressionFixture":
    """
    Fixture used to test arbitrary data against known versions previously
    recorded by this same fixture. Useful to test 3rd party APIs or where testing directly
    generated data from other sources.

    Create a dict in your test containing a arbitrary data you want to test, and
    call the `Check` function. The first time it will fail but will generate a file in your
    data directory.

    Subsequent runs against the same data will now compare against the generated file and pass
    if the dicts are equal, or fail showing a diff otherwise. To regenerate data for failed tests,
    either set `force_regen` attribute to True or pass the `--force-regen` flag to pytest
    which will regenerate the data files for all failed tests. Make sure to diff the files to ensure
    the new changes are expected and not regressions.

    The dict may be anything serializable by the `yaml` library.
    """
    from .data_regression import DataRegressionFixture

    return DataRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def dataframe_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "DataFrameRegressionFixture":
    """
    Example usage:

    def testSomeData(dataframe_regression):
        dataframe_regression.check(
            pandas.DataFrame.from_dict(
                {
                    'U_gas': U[0],
                    'U_liquid': U[1],
                    'gas_vol_frac [-]': α[0],
                    'liquid_vol_frac [-]': α[1],
                    'P': Pa_to_bar(P),
                }
            ),
            default_tolerance=dict(atol=1e-8, rtol=1e-8)
        )
    """
    from .dataframe_regression import DataFrameRegressionFixture

    return DataFrameRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def ndarrays_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "NDArraysRegressionFixture":
    """
    Similar to num_regression, but supports numpy arrays with arbitrary shape. The
    dictionary is stored as an NPZ file. The values of the dictionary must be accepted
    by ``np.asarray``.

    Example::

        def test_some_data(ndarrays_regression):
            points, values = some_function()
            ndarrays_regression.check(
                {
                    'points': points,  # array with shape (100, 3)
                    'values': values,  # array with shape (100,)
                },
                default_tolerance=dict(atol=1e-8, rtol=1e-8)
            )
    """
    from .ndarrays_regression import NDArraysRegressionFixture

    return NDArraysRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def file_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "FileRegressionFixture":
    """
    Very similar to `data_regression`, but instead of saving data to YAML file it saves to an
    arbitrary format.

    Useful when used to compare contents of files of specific format (like documents, for instance).
    """
    from .file_regression import FileRegressionFixture

    return FileRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def num_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "NumericRegressionFixture":
    """
    Example usage:

    def testSomeData(num_regression):
        num_regression.check(
            {
                'U_gas': U[0],
                'U_liquid': U[1],
                'gas_vol_frac [-]': α[0],
                'liquid_vol_frac [-]': α[1],
                'P': Pa_to_bar(P),
            },
            data_index=positions,
            default_tolerance=dict(atol=1e-8, rtol=1e-8)
        )
    """
    from .num_regression import NumericRegressionFixture

    return NumericRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def num_relative_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "NumericRelativeRegressionFixture":
    """
    Example usage:

    # baseline data is:
    # data1 = 1.1 * np.ones(5000)
    # data2 = 2.2 * np.ones(5000)

    data1 = 2.5 * np.ones(5000)
    data2 = 2.1 * np.ones(5000)
    num_relative_regression.check({"data1": data1, "data2": data2},
                                  compare_operator=">=",
                                  max_absolute_diff=-0.1)
    data1 = 0.6 * np.ones(5000)
    data2 = 2.7 * np.ones(5000)
    num_relative_regression.check({"data1": data1, "data2": data2},
                                  compare_operator="<=",
                                  max_absolute_diff=0.5)
    """
    from .num_relative_regression import NumericRelativeRegressionFixture

    return NumericRelativeRegressionFixture(datadir, original_datadir, request)


@pytest.fixture
def image_regression(
    datadir: Path, original_datadir: Path, request: pytest.FixtureRequest
) -> "ImageRegressionFixture":
    """
    Regression checks for images, accounting for small differences between them.

    Example:
        def test_plots(datadir, image_regression):
            path = generate_plot(datadir / 'plot.png')
            image_regression.check(path.read_bytes(), diff_threshold=2.5)  # 2.5%
    """
    from .image_regression import ImageRegressionFixture

    return ImageRegressionFixture(datadir, original_datadir, request)
