# Copyright 2019 Drivetrain Hub LLC
# For non-commercial use only.  For commercial products and services, visit https://www.drivetrainhub.com.

"""Notebook module for rotational mechanics math primer."""

from math import hypot, cos, sin, pi
from cmath import exp
import numpy as np
import pandas as pd
from IPython.display import display, HTML
from typing import Callable


# region HELPERS

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
        # METHOD 1 - does not work with lists or arrays
        # df[col] = df[col].astype(float).round(n_round)

        # METHOD 2 - works with lists and arrays
        for i, row in enumerate(df[col]):
            df[col][i] = np.round(row, n_round)

    display(HTML(df.to_html(index=False)))


def display_linear_eqn(coefficients, constant):
    """Display linear equation in LaTeX format.

    :return: LaTeX equation of the form Ax = b.
    """

    n_variables = len(coefficients)

    eqn_str = f'$'  # initialize
    for i, a in enumerate(coefficients):
        var = n_variables - i
        eqn_str += f'{a}x_{var} +'

    eqn_str = eqn_str[:-1]
    eqn_str += f'={constant}$'
    display(HTML(eqn_str))


def display_euler_eqn(amplitude, angular_frequency, phase_angle, include_equals=False):
    """Display Euler formula in LaTeX format.

    :return: LaTeX equation of Euler's formula.
    """

    phase = round(phase_angle, 3)

    eqn_str = f'${amplitude} e^{{i ({angular_frequency} t + {phase}) }}'

    if include_equals:
        eqn_str += f'= {amplitude} \\left(' \
            f'\\cos\\left( {angular_frequency} t + {phase} \\right) + ' \
            f'i \\sin\\left( {angular_frequency} t + {phase} \\right)' \
            f'\\right)'

    eqn_str += '$'

    display(HTML(eqn_str))


# endregion


# region POINTS & VECTORS

def distance(point1: np.ndarray, point2: np.ndarray):
    """Distance between two points in a Cartesian coordinate system.

    :param point1: Cartesian coordinate of point 1.
    :param point2: Cartesian coordinate of point 2.
    :return: Absolute distance between the two points.
    """

    subtracted = np.subtract(point2, point1)  # point2 - point1

    subtracted_magnitude = np.linalg.norm(subtracted)

    return subtracted_magnitude


def vector_magnitude(x, y, z):
    """Magnitude of a Cartesian vector.

    :param x: Vector x-component.
    :param y: Vector y-component.
    :param z: Vector z-component.
    :return: Magnitude of vector.
    """

    return hypot(hypot(x, y), z)


# endregion


# region PERIODIC SIGNALS

def euler_form(amplitude: float, phase: float):
    """Formulate signal in time-independent Euler form, i.e. angular frequency of signal is excluded.

    :param amplitude: Amplitude of signal.
    :param phase: Phase angle of signal.
    :return: Euler form of time-independent signal.
    """

    return amplitude * exp(1j * phase)


def euler_fcn(amplitude: float, omega: float, phase: float) -> Callable:
    """Create a function of the Euler formula to represent a vector rotating in the complex plane.

    :param amplitude: Amplitude of signal.
    :param omega: Angular frequency of signal.
    :param phase: Phase angle of signal.
    :return: Function of Euler formula, with independent variable 't' as its input argument.
    """

    return lambda t: amplitude * np.exp(1j * (omega * t + phase))


def fourier_series(a0: float, an: Callable, bn: Callable, n_harmonics: int, num: int = 20):
    """Fourier series expansion, calculated from sin-cos form.

    Considers only one period of the Fourier series.

    :param a0: a_0 coefficient.
    :param an: a_n coefficient function, accepting nth harmonic as its only input arg.
    :param bn: b_n coefficient function, accepting nth harmonic as its only input arg.
    :param n_harmonics: Total number of harmonics to include in Fourier series.
    :param num: Number of points to calculate for a single period.
    :return: Fourier series expansion evaluated at the values of independent variable, x.
    """

    p = 1
    x = np.linspace(0, p, num=num)
    y = a0 / 2

    for n in range(1, n_harmonics + 1):
        y += an(n) * np.cos(2 * pi * n * x) / p + bn(n) * np.sin(2 * pi * n * x) / p

    return y


def fourier_series_square_wave(amplitude: float, period: float, duty_cycle: float, n_harmonics: int, num: int = 20):
    """Fourier series of a square wave.

    Considers only one period of the square wave.

    :param amplitude: Amplitude of square wave.
    :param period: Duration of square wave period.
    :param duty_cycle: Fraction of period at the amplitude.
    :param n_harmonics: Number of harmonics in Fourier series.
    :param num: Number of points to calculate for a single period of the highest harmonic.
    :return: (x, y) of independent variable and Fourier series expansion for one period of a square wave.
    """

    num_total = n_harmonics * num
    x = np.linspace(0, period, num=num_total)
    a0 = 2 * amplitude * duty_cycle

    def an(n):
        return 2 * amplitude * sin(2 * pi * n * duty_cycle) / (pi * n)

    def bn(n):
        return 2 * amplitude * (sin(pi * n * duty_cycle) ** 2) / (pi * n)

    return x, fourier_series(a0, an, bn, n_harmonics, num_total)


# endregion


# region LINEAR ALGEBRA

def is_orthogonal(matrix: np.ndarray) -> bool:
    """Check for orthogonality by verifying the matrix is an orthonormal set.

    :param matrix: Matrix, 2-D array (3x3), to check for orthogonality.
    :return: Boolean to indicate orthogonality.
    """

    orthogonal = \
        np.dot(matrix[0], matrix[1]) == 0 and \
        np.dot(matrix[0], matrix[2]) == 0 and \
        np.dot(matrix[1], matrix[2]) == 0

    return orthogonal


def rotation_matrix_about_axis(theta: float, axis: np.array):
    """Rotational coordinate transform with rotation about a vector by an angle.  Right-hand rule is used.

               |xxC+c   xyC+zs  xzC-ys|
    R(theta) = |yxC-zs  yyC+c   yzC+xs| --> passive transform
               |zxC+ys  zyC-xs  zzC+c |

    where,
        * c = cos(theta)
        * s = sin(theta)
        * C = 1 - c

    :param theta: Angle of rotation about axis.
    :param axis: Unit vector for axis of rotation, 1-D array.
    :return: Coordinate transform rotation matrix (passive transform), 2-D array.
    """

    unit_axis = axis / np.linalg.norm(axis)
    x, y, z = unit_axis
    c = cos(theta)
    s = sin(theta)
    C = 1 - c

    # passive transformation
    rotation_matrix = np.array([[x * x * C + c, x * y * C + z * s, x * z * C - y * s],
                                [y * x * C - z * s, y * y * C + c, y * z * C + x * s],
                                [z * x * C + y * s, z * y * C - x * s, z * z * C + c]])

    return rotation_matrix


def transform_active(rotation_matrix_passive: np.array, arr: np.array):
    """Apply passive rotation matrix to a vector or matrix with pre-multiplication.

    Active transformation use cases:
        1. Rotate a point or vector in its current coordinate system.
        2. Express coordinate axes in the coordinate system defined by the rotation matrix.

    :param rotation_matrix_passive: Rotation matrix of passive transformation, 2-D array.
    :param arr: 1-D (vector) or 2-D (matrix) array.
    :return: If vector input, then returns rotated vector, 1-D array.  If matrix input, then returns coordinate axes,
        2-D array, expressed in the coordinate system defined by the rotation matrix.
    """

    rotation_matrix_active = np.transpose(rotation_matrix_passive)

    vector_rotated = np.dot(rotation_matrix_active, arr)

    return vector_rotated


def transform_passive(rotation_matrix_passive: np.array, arr: np.array):
    """Apply passive rotation matrix to a vector or matrix with pre-multiplication.

    Passive transformation use cases:
        1. Rotate coordinate axes in its current coordinate system.
        2. Express a point or vector in the coordinate system defined by the rotation matrix.

    :param rotation_matrix_passive: Rotation matrix of passive transformation, 2-D array.
    :param arr: 1-D (vector) or 2-D (matrix) array.
    :return: Rotated coordinate system, 2-D array (3x3).
    :return: If vector input, then returns vector, 1-D array, expressed in the coordinate system defined by the rotation matrix.
        If matrix input, then returns rotated coordinate axes, 2-D array.
    """

    coordinate_system_rotated = np.dot(rotation_matrix_passive, arr)

    return coordinate_system_rotated


def transform_rotate_vector(rotation_matrix_passive: np.array, vector: np.array):
    """Rotate vector by the specified rotation matrix.

    Syntactic sugar for active transformation.

    :param rotation_matrix_passive: Rotation matrix of passive transformation, 2-D array.
    :param vector: Vector, 1-D array.
    :return: Rotated vector, 1-D array.
    """

    vector_rotated = transform_active(rotation_matrix_passive, vector)

    return vector_rotated


def transform_rotate_axes(rotation_matrix_passive: np.array, axes: np.array):
    """Rotate coordinate system per rotation matrix.

    Syntactic sugar for passive transformation.

    :param rotation_matrix_passive: Rotation matrix of passive transformation, 2-D array.
    :param axes: Coordinate axes, 2-D array.
    :return: Rotated coordinate axes, 2-D array.
    """

    coordinate_system_rotated = transform_passive(rotation_matrix_passive, axes)

    return coordinate_system_rotated


def transform_vector_in_coord_sys(rotation_matrix: np.array, vector: np.array):
    """Express vector in the coordinate system defined by a rotation matrix.

    Syntactic sugar for passive transformation.

    :param rotation_matrix: Rotation matrix defining the orientation of Cartesian coordinate axes.
    :param vector: Vector, 1-D array.
    :return: Vector, 1-D array, expressed in the rotated coordinate system.
    """

    return transform_passive(rotation_matrix, vector)


def transform_axes_in_coord_sys(rotation_matrix: np.array, axes: np.array):
    """Express coordinate axes in the coordinate system defined by a rotation matrix.

    Syntactic sugar for active transformation.

    :param rotation_matrix: Rotation matrix defining the orientation of Cartesian coordinate axes.
    :param axes: Coordinate axes, 2-D array.
    :return: Coordinate axes, 2-D array, expressed in the rotated coordinate system.
    """

    return transform_active(rotation_matrix, axes)

# endregion
