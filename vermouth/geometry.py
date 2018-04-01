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

    coordinates_a, coordinates_b: np.ndarray
        Coordinates of the points in the selections. Each row must correspond
        to a point and each column to a dimension.

    Returns
    -------
    np.ndarray
        Rows correspond to the points from 'coordinates_a', columns correspond
        from 'coordinates_b'.
    """
    return np.sqrt(
        np.sum(
            (coordinates_a[:, np.newaxis, :] - coordinates_b[np.newaxis, :, :]) ** 2,
            axis=-1)
    )


def angle(vectorBA, vectorBC):
    """
    Calculate the angle in radians between two vectors.

    The function assumes the following situation:

          B
         / \
        A   C

    It returns the angle between BA and BC.
    """
    nominator = np.dot(vectorBA, vectorBC)
    denominator = np.linalg.norm(vectorBA) * np.linalg.norm(vectorBC)
    cosine = nominator / denominator
    # Floating errors at the limits may cause issues.
    cosine = np.clip(cosine, -1, 1)
    return np.arccos(cosine)


def dihedral(coordinates):
    """
    Calculate the dihedral angle in radians.

    Parameters
    ----------
    coordinates: np.ndarray
        The coordinates of 4 points defining the dihedral angle. Each row
        corresponds to a point, and each column to a dimension.

    Returns
    -------
    float
        The calculated angle between -pi and +pi.
    """
    vectorAB = coordinates[1, :] - coordinates[0, :]
    vectorBC = coordinates[2, :] - coordinates[1, :]
    vectorCD = coordinates[3, :] - coordinates[2, :]
    normalABC = np.cross(vectorAB, vectorBC)
    normalBCD = np.cross(vectorBC, vectorCD)
    psin = np.dot(normalABC, vectorCD) * np.linalg.norm(vectorBC)
    pcos = np.dot(normalABC, normalBCD)
    return np.arctan2(psin, pcos)


def dihedral_phase(coordinates):
    """
    Calculate a dihedral angle in radians with a -pi phase correction.

    Parameters
    ----------
    coordinates: np.ndarray
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
    angle = dihedral(coordinates)
    angle -= np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle
