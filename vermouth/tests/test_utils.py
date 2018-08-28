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

import string
import pytest
from hypothesis import given, strategies, example
import numpy as np
from numpy.testing import assert_allclose
from vermouth import utils


@pytest.mark.parametrize('iter_type', (list, tuple, iter, np.array))
@pytest.mark.parametrize('values, ref', (
    # All positive integers, one maximum
    ([1, 2, 3, 4], [4]),
    ([4, 2, 1, 3], [4]),
    # All positive integers, multiple maxima
    ([4, 1, 4, 2], [4] * 2),
    ([4, 4, 4, 4], [4] * 4),
    # All positive or negative integers, multiple maxima
    ([-1, 2, -4, 0, 2], [2] * 2),
    ([-1, -2, -3, -1], [-1] * 2),
    # All floats
    ([0.1, 0.4, 3.4, 2.1, 3.4], [3.4] * 2),
    # Strings
    (['z', 'aaa', 'bbb', 'z', 'z'], ['z'] * 3),
))
def test_maxes_simple_no_key(values, ref, iter_type):
    assert utils.maxes(iter_type(values)) == ref


@pytest.mark.parametrize('iter_type', (list, tuple, iter, np.array))
@pytest.mark.parametrize('values, ref', (
    # All positive integers, one maximum
    ([1, 2, 3, 4], [1]),
    ([4, 2, 1, 3], [1]),
    # All positive integers, multiple maxima
    ([4, 1, 1, 2], [1] * 2),
    ([4, 4, 4, 4], [4] * 4),
    # All positive or negative integers, multiple maxima
    ([-1, 2, -4, 0, -4], [-4] * 2),
    ([-3, -2, -3, -1], [-3] * 2),
    # All floats
    ([0.1, 0.4, 3.4, 0.1, 3.4], [0.1] * 2),
))
def test_maxes_key(values, ref, iter_type):
       assert utils.maxes(iter_type(values), key=lambda x: -x) == ref


@pytest.mark.parametrize('alpha', string.ascii_letters)
@pytest.mark.parametrize('template', (
    '{}bc', ' {}bc', '123{}WS', '@#$%{}*&^', '\U0001F62E{}',
))
def test_first_alpha(template, alpha):
    value = template.format(alpha)
    assert utils.first_alpha(value) == alpha


@pytest.mark.parametrize('value', ('', '0123', '\U0001F62E'))
def test_first_alpha_value_error(value):
    with pytest.raises(ValueError):
        utils.first_alpha(value)


@pytest.mark.parametrize('iter_type', (list, tuple, iter, np.array))
@pytest.mark.parametrize('values, ref', (
    # All equal
    ([0] * 5, True), ([True] * 4, True),
    ([None] * 5, True), ([1e-3] * 4, True), 
    # Not all equal
    ([0, 1] * 4, False), (['a', 'b', 'c'], False),
    # Empty
    ([], True),
    # One element
    ([1], True), ([None], True),
    # Should it be this way?
    ([0, False] * 3, True),
))
def test_are_all_equal(values, ref, iter_type):
    assert utils.are_all_equal(iter_type(values)) is ref


@pytest.mark.parametrize('values, ref', (
    ([list([0, 1, 2]) for _ in range(3)], True),   
    ([[0, 1, 2], [3, 4, 5], [6, 7, 8]], False),
))
def test_are_all_equal_multidim_list(values, ref):
    assert utils.are_all_equal(values) is ref


def test_are_all_equal_not_implemented():
    value = np.zeros((3, 4))
    with pytest.raises(NotImplementedError):
        utils.are_all_equal(value)


@strategies.composite
def vector_with_random_distance(draw):
    """
    Generate a vector with a random length and orientation.

    The vector is returned as the two points at its extremity.

    Returns
    -------
    point1: numpy.ndarray
    point2: numpy.ndarray
    distance: float
    """
    # Generate random polar coordinates and convert them to euclidean
    # coordinates.
    length = strategies.floats(min_value=0, max_value=100,
                               allow_nan=False, allow_infinity=False)
    angle = strategies.floats(min_value=0, max_value=2 * np.pi,
                              allow_nan=False, allow_infinity=False)
    distance = draw(length)
    theta = draw(angle)
    phi = draw(angle)
    shift = np.array([draw(length), draw(length), draw(length)])
    x = distance * np.sin(theta) * np.cos(phi)
    y = distance * np.sin(theta) * np.sin(phi)
    z = distance * np.cos(theta)
    return shift, np.array([x, y, z]) + shift, distance


@strategies.composite
def test_distance(vec_and_dist):
    vec_and_dist = draw(vector_with_random_distance)
    point1, point2, distance = vec_and_dist
    assert_allclose(utils.distance(point1, point2), distance)
