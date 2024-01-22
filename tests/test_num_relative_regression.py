import sys

import numpy as np
import pandas as pd
import pytest

from pytest_regressions.testing import check_regression_fixture_workflow


@pytest.fixture
def no_regen(num_relative_regression, request):
    if num_relative_regression._force_regen or request.config.getoption("force_regen"):
        pytest.fail("--force-regen should not be used on this test.")


def test_common_cases(num_relative_regression, no_regen):
    # Most common case: Data is valid, is present and should pass
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

