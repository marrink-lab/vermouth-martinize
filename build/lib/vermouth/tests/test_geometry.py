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
Tests for the geometry module.
"""

import itertools
import numpy as np
import pytest
from vermouth import geometry


def _generate_test_angles(n_angles):
    """
    Gererate a series of coordinates of 3 points at a different angles.

    Generate 'n_angles' structures with angles between 0 and pi with regular
    angle spacing.

    Parameters
    ----------
    n_angles: int
        Number of structures to generate.

    Yields
    ------
    coordinates: np.ndrarray
        The coordinates of the points. Each row corresponds to a pointm and
        each column corresponds to a dimension.
    angle: float
        The angle between the 3 points.
    """
    # The shift is arbitrary. Its purpose is to avoid having the angle centered
    # at the origin as it may be a special case.
    shift = np.array([1.1, -7.2, 9.1])
    coordinates = np.array([
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 0],  # Will be redifined
    ]) + shift
    # The distance of the 3rd point to the second one, also the radius in polar
    # coordinates. The value is totally arbitrary, and it should not change the
    # angle; but avoid setting the radius to 1 as it may hide normalization
    # errors.
    radius = 4.2
    for angle in np.linspace(0, np.pi, num=n_angles):
        coordinates[-1, 0] = radius * np.cos(angle) + shift[0]
        coordinates[-1, 1] = radius * np.sin(angle) + shift[1]
        yield coordinates.copy(), angle


def _generate_test_dihedrals(n_angles):
    """
    Generate a series of coordinates for 4 points at different torsion angles.

    Generate 'n_angles' structures with torsion angles between -pi and +pi
    with regular angle spacing.

    Parameters
    ----------
    n_angles: int
        Number of structures to generate.

    Yields
    ------
    coordinates: np.ndrarray
        The coordinates of the points. Each row corresponds to a pointm and
        each column corresponds to a dimension.
    angle: float
        The angle around the ais between the middle points.
    """
    # The shift is arbitrary. Its purpose is to avoid having the angle centered
    # at the origin as it may be a special case.
    shift = np.array([1.1, -7.2, 9.1])
    coordinates = np.array([
        [2, 0, 0],
        [0, 0, 0],
        [0, 0, 5],
        [0, 0, 5],  # Will be redifined
    ]) + shift
    # The distance of the 3rd point to the second one, also the radius in polar
    # coordinates. The value is totally arbitrary, and it should not change the
    # angle; but avoid setting the radius to 1 as it may hide normalization
    # errors.
    radius = 4.2
    for angle in np.linspace(-np.pi, +np.pi, num=n_angles):
        coordinates[-1, 0] = radius * np.cos(angle) + shift[0]
        coordinates[-1, 1] = radius * np.sin(angle) + shift[1]
        yield coordinates.copy(), angle


@pytest.mark.parametrize(
    'points, angle',
    itertools.chain(
        _generate_test_angles(10),
        ((np.array([[0,  3, 0], [0, 0, 0], [0, 6, 0]]), 0), ),  # pylint: disable=bad-whitespace
        ((np.array([[0, -9, 0], [0, 0, 0], [0, 2, 0]]), np.pi), ),
    )
)
def test_angle(points, angle):
    vectorBA = points[0, :] - points[1, :]
    vectorBC = points[2, :] - points[1, :]
    assert np.allclose(geometry.angle(vectorBA, vectorBC), angle)


@pytest.mark.parametrize(
    'points, angle',
    itertools.chain(
        _generate_test_dihedrals(10),
        ((np.array([[0,  3, 0], [0, 0, 0], [4, 0, 0], [7, 6, 0]]), 0), ),  # pylint: disable=bad-whitespace
        ((np.array([[0, -9, 0], [0, 0, 0], [4, 0, 0], [5, 6, 0]]), np.pi), ),
    )
)
def test_dihedral(points, angle):
    calc_angle = geometry.dihedral(points)
    # +pi and -pi are the same angle; we normalize them to pi
    if np.allclose(calc_angle, -np.pi):
        calc_angle *= -1
    if np.allclose(angle, -np.pi):
        angle *= -1
    assert np.allclose(calc_angle, angle)


@pytest.mark.parametrize(
    'points, angle',
    itertools.chain(
        _generate_test_dihedrals(10),
        ((np.array([[0,  3, 0], [0, 0, 0], [4, 0, 0], [7, 6, 0]]), 0), ),  # pylint: disable=bad-whitespace
        ((np.array([[0, -9, 0], [0, 0, 0], [4, 0, 0], [5, 6, 0]]), np.pi), ),
    )
)
def test_dihedral_phase(points, angle):
    angle_phase = angle + np.pi
    if angle_phase > np.pi:
        angle_phase -= 2 * np.pi
    if angle_phase < -np.pi:
        angle_phase += 2 * np.pi
    calc_angle = geometry.dihedral_phase(points)
    # +pi and -pi are the same angle; we normalize them to pi
    if np.allclose(calc_angle, -np.pi):
        calc_angle *= -1
    if np.allclose(angle_phase, -np.pi):
        angle_phase *= -1
    assert np.allclose(calc_angle, angle_phase)


def test_distance_matrix():
    # This array of coordinates was generated using:
    #    coordinates = (
    #        np.random.uniform(low=-2, high=2, size=(15, 3))
    #        .astype(np.float32)
    #        .round(2)
    #    )
    #    coordinates.tolist()
    coordinates = np.array([
        [-0.4099999964237213, 0.5699999928474426, -1.2000000476837158],
        [-0.05000000074505806, 1.6799999475479126, 1.0800000429153442],
        [1.649999976158142, -1.2699999809265137, -0.18000000715255737],
        [-0.4300000071525574, -0.5, -1.25],
        [1.75, -0.28999999165534973, -1.75],
        [1.399999976158142, 1.0399999618530273, -0.029999999329447746],
        [0.25999999046325684, -0.9399999976158142, -0.8899999856948853],
        [1.159999966621399, -0.10999999940395355, 0.07999999821186066],
        [0.009999999776482582, 0.5199999809265137, -0.8600000143051147],
        [-1.909999966621399, 1.5, -0.27000001072883606],
        [-1.5800000429153442, 0.5299999713897705, -0.6499999761581421],
        [1.9900000095367432, -1.7799999713897705, 1.7699999809265137],
        [1.6100000143051147, -0.03999999910593033, -0.8899999856948853],
        [0.10000000149011612, 1.1200000047683716, 0.17000000178813934],
        [0.41999998688697815, -0.5699999928474426, 0.33000001311302185],
    ])

    # The reference was built using MDAnalysis:
    #     mad.distance_array(coordinates[:6], coordinates[6:15]).tolist()
    reference = np.array([
        [1.6808033650982364, 2.1367498758730443, 0.5426785539106965,
         1.9949436426807041, 1.293445112450694, 4.483681543049364,
         2.1327447054976916, 1.5618898861478303, 2.0807210809989],
        [3.2926280559049035, 2.3807981156038913, 2.2611501818462068,
         2.3053199558207305, 2.5799806759724233, 4.075450898096163,
         3.0975635086704183, 1.0789809940089903, 2.417829605013437],
        [1.5953369295377542, 1.285807111097364, 2.5211306642537474,
         4.511607195619645, 3.727438788554226, 2.044064626898197,
         1.4207744414223977, 2.8700347089662435, 1.504327098906351],
        [0.8940357962136768, 2.109288939442685, 1.1773274598997576,
         2.674097996304839, 1.6563212819432889, 4.076174699496656,
         2.1219801748218843, 2.21849046119744, 1.7954944096661205],
        [1.8390758608686804, 1.931165501366316, 2.115608639381968,
         4.334754756955464, 3.601569024030028, 3.8298955467113966,
         0.9064767071466125, 2.8977576955983215, 2.4846930992457765],
        [2.4412292124092203, 1.1799152298407856, 1.7004117182714393,
         3.3504178337743813, 3.086243689319313, 3.397131067712101,
         1.3964597652032842, 1.31772526881778, 1.9188798060046086]
    ])

    matrix = geometry.distance_matrix(coordinates[:6], coordinates[6:15])
    assert matrix.shape == (6, 9)
    assert np.allclose(matrix, reference)
