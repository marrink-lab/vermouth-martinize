# -*- coding: utf-8 -*-
# Copyright 2018 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Geometric operations.
"""

import numpy as np


def distance_matrix(coordinates_a, coordinates_b):
    """
    Compute a distance matrix between two set of points.

    Notes
    -----
    This function does **not** account for periodic boundary conditions.

    Parameters
    ----------
    coordinates_a: numpy.ndarray
        Coordinates of the points in the selections. Each row must correspond
        to a point and each column to a dimension.
    coordinates_b: numpy.ndarray
        Coordinates of the points in the selections. Each row must correspond
        to a point and each column to a dimension.

    Returns
    -------
    numpy.ndarray
        Rows correspond to the points from `coordinates_a`, columns correspond
        from `coordinates_b`.
    """
    return np.sqrt(
        np.sum(
            (coordinates_a[:, np.newaxis, :] - coordinates_b[np.newaxis, :, :]) ** 2,
            axis=-1)
    )


def angle(vector_ba, vector_bc):
    """
    Calculate the angle in radians between two vectors.

    The function assumes the following situation::

          B
         / \\
        A   C

    It returns the angle between BA and BC.
    """
    nominator = np.dot(vector_ba, vector_bc)
    denominator = np.linalg.norm(vector_ba) * np.linalg.norm(vector_bc)
    cosine = nominator / denominator
    # Floating errors at the limits may cause issues.
    cosine = np.clip(cosine, -1, 1)
    return np.arccos(cosine)


def dihedral(coordinates):
    """
    Calculate the dihedral angle in radians.

    Parameters
    ----------
    coordinates: numpy.ndarray
        The coordinates of 4 points defining the dihedral angle. Each row
        corresponds to a point, and each column to a dimension.

    Returns
    -------
    float
        The calculated angle between -pi and +pi.
    """
    vector_ab = coordinates[1, :] - coordinates[0, :]
    vector_bc = coordinates[2, :] - coordinates[1, :]
    vector_cd = coordinates[3, :] - coordinates[2, :]
    normal_abc = np.cross(vector_ab, vector_bc)
    normal_bcd = np.cross(vector_bc, vector_cd)
    psin = np.dot(normal_abc, vector_cd) * np.linalg.norm(vector_bc)
    pcos = np.dot(normal_abc, normal_bcd)
    return np.arctan2(psin, pcos)


def dihedral_phase(coordinates):
    """
    Calculate a dihedral angle in radians with a -pi phase correction.

    Parameters
    ----------
    coordinates: numpy.ndarray
        The coordinates of 4 points defining the dihedral angle. Each row
        corresponds to a point, and each column to a dimension.

    Returns
    -------
    float
        The calculated angle between -pi and +pi.

    See Also
    --------
    dihedral
        Calculate a dihedral angle.
    """
    dihedral_angle = dihedral(coordinates)
    dihedral_angle -= np.pi
    if dihedral_angle > np.pi:
        dihedral_angle -= 2 * np.pi
    if dihedral_angle < -np.pi:
        dihedral_angle += 2 * np.pi
    return dihedral_angle
