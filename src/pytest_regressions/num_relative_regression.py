import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence

from .common import import_error_message
from .num_regression import NumericRegressionFixture


class NumericRelativeRegressionFixture(NumericRegressionFixture):

    def check(
        self,
        data_dict: Dict[str, Any],
        basename: Optional[str] = None,
        fullpath: Optional["os.PathLike[str]"] = None,
        tolerances: Optional[Dict[str, Dict[str, float]]] = None,
        default_tolerance: Optional[Dict[str, float]] = None,
        data_index: Optional[Sequence[int]] = None,
        fill_different_shape_with_nan: bool = True,
        max_absolute_diff: float = 0.0,
        compare_operator: str = ">=",
    ) -> None:
        self._max_absolute_diff = max_absolute_diff
        self._compare_operator = compare_operator
        super().check(data_dict, basename, fullpath, tolerances,
                      default_tolerance, data_index,
                      fill_different_shape_with_nan)


    def _check_fn(self, obtained_filename: Path, expected_filename: Path) -> None:
        """
        Check if dict contents dumped to a file match the contents in expected file.
        """
        print("JUHEEEEEEJ MY CUSTOM CHECK FUNCTION CALLED >****>>>>**********")
        try:
            import numpy as np
        except ModuleNotFoundError:
            raise ModuleNotFoundError(import_error_message("NumPy"))
        try:
            import pandas as pd
        except ModuleNotFoundError:
            raise ModuleNotFoundError(import_error_message("Pandas"))

        __tracebackhide__ = True

        obtained_data = pd.read_csv(str(obtained_filename))
        expected_data = pd.read_csv(str(expected_filename))

        comparison_tables_dict = {}
        for k in obtained_data.keys():
            obtained_column = obtained_data[k]
            expected_column = expected_data.get(k)

            if expected_column is None:
                error_msg = f"Could not find key '{k}' in the expected results.\n"
                error_msg += "Keys in the obtained data table: ["
                for k in obtained_data.keys():
                    error_msg += f"'{k}', "
                error_msg += "]\n"
                error_msg += "Keys in the expected data table: ["
                for k in expected_data.keys():
                    error_msg += f"'{k}', "
                error_msg += "]\n"
                error_msg += "To update values, use --force-regen option.\n\n"
                raise AssertionError(error_msg)

            tolerance_args = self._tolerances_dict.get(k, self._default_tolerance)

            self._check_data_types(k, obtained_column, expected_column)
            self._check_data_shapes(obtained_column, expected_column)

            if np.issubdtype(obtained_column.values.dtype, np.inexact):
                if self._compare_operator == "<=":
                    if (self._max_absolute_diff < 0):
                        raise AssertionError("it makes no sense to use max_absolute value < 0 when using compare_operator <=")
                    not_close_mask = ~np.less_equal(
                        obtained_column.values,
                        np.add(expected_column.values, self._max_absolute_diff),
                    )
                elif self._compare_operator == ">=":
                    if (self._max_absolute_diff > 0):
                        raise AssertionError("it makes no sense to use max_absolute value > 0 when using compare_operator >=")
                    not_close_mask = ~np.greater_equal(
                        obtained_column.values,
                        np.add(expected_column.values, self._max_absolute_diff),
                    )
                else:
                    raise AssertionError("only allowed values for compare_operator are <= or >=, you used: " + self._compare_operator)
            else:
                not_close_mask = obtained_column.values != expected_column.values
                # If Empty/NaN data is expected, then the values are equal:
                not_close_mask[
                    np.logical_and(
                        pd.isna(obtained_column.values), pd.isna(expected_column.values)
                    )
                ] = False

            if np.any(not_close_mask):
                diff_ids = np.where(not_close_mask)[0]
                diff_obtained_data = obtained_column[diff_ids]
                diff_expected_data = expected_column[diff_ids]
                if obtained_column.values.dtype == bool:
                    diffs = np.logical_xor(obtained_column, expected_column)[diff_ids]
                elif obtained_column.values.dtype == object:
                    diffs = diff_obtained_data.copy()
                    diffs[:] = "?"
                else:
                    diffs = np.abs(obtained_column - expected_column)[diff_ids]

                comparison_table = pd.concat(
                    [diff_obtained_data, diff_expected_data, diffs], axis=1
                )
                comparison_table.columns = [f"obtained_{k}", f"expected_{k}", "diff"]
                comparison_tables_dict[k] = comparison_table

        if len(comparison_tables_dict) > 0:
            error_msg = "Values are not sufficiently close.\n"
            error_msg += "To update values, use --force-regen option.\n\n"
            for k, comparison_table in comparison_tables_dict.items():
                error_msg += f"{k}:\n{comparison_table}\n\n"
            if obtained_column.values.dtype == object:
                error_msg += (
                    "WARNING: diffs for this kind of data type cannot be computed."
                )
            raise AssertionError(error_msg)


