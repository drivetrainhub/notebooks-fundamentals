# Copyright 2019 Drivetrain Hub LLC
# For non-commercial use only.  For commercial products and services, visit https://www.drivetrainhub.com.

"""Notebook module for rotational kinematics."""

from math import pi
import numpy as np
import pandas as pd
from IPython.display import display, HTML


def df_display(data, columns, round_columns=(), n_round=3):
    """Display a pandas DataFrame as HTML, rounding specified columns.

    :param data: Data array for DataFrame.
    :param columns: List of column headings for DataFrame.
    :param round_columns: List of columns to be rounded.
    :param n_round: Number of digits to round to.
    :return: None.
    """

    df = pd.DataFrame(data=data, columns=columns)

    for col in round_columns:
        df[col] = df[col].astype(float).round(n_round)

    display(HTML(df.to_html(index=False)))


def display_eqn(eqn_str: str, value: float = None):
    """Display equation in LaTeX format."""

    if value is not None:
        eqn_str += f' = {value}'

    display(HTML(eqn_str))


def rpm(omega):
    """Convert radians per second to revolutions per minute."""

    return omega * 30 / pi


def solve_parallel_gears():
    # TODO
    pass


def solve_planetary_gears():
    # TODO
    pass
