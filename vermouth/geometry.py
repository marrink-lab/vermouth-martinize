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
    if cosine > 1:
        cosine = 1
    elif cosine < -1:
        cosine = -1
    return np.arccos(cosine)


def dihedral(vectorAB, vectorBC, vectorCD):
    """
    Calculate the dihedral angle in radians formed by 3 vectors.

    The function assumes the following situation:

        A - B
             \
              C - D
    
    The angle around the BC axis is returned with values between -pi and +pi.
    The direction of the angle follows the right hand convention.
    """
    normalABC = np.cross(vectorAB, vectorBC)
    normalBCD = np.cross(vectorBC, vectorCD)
    psin = np.dot(normalABC, vectorCD) * np.linalg.norm(vectorBC)
    pcos = np.dot(normalABC, normalBCD)
    return np.arctan2(psin, pcos)


def dihedral_left(vectorAB, vectorBC, vectorCD):
    """
    Calculate a dihedral angle in radians following the left hand convention.

    See Also
    --------
    dihedral
        Calculate a dihedral angle in the right hand convention.
    """
    angle = dihedral(vectorAB, vectorBC, vectorCD)
    angle -= np.pi
    if angle > np.pi:
        angle -= 2 * np.pi
    if angle < -np.pi:
        angle += 2 * np.pi
    return angle
