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
Tests for the `test_utils.py` module.
"""

import itertools
import string
import pytest
from hypothesis import strategies, given, example
import numpy as np
from numpy.testing import assert_allclose
from vermouth import utils

DIFFERENCE_USE_CASE = (
    '', [], 'a', 2, 3, True, False, None, float('nan'), float('inf'),
    [True], [False], [True, False], [False, False], [None], [None, None],
    [1.1, 1.1], [0.0, 1.1],
    [float('nan'), float('inf')],
    [float('inf'), float('nan')],
    ['a', 'b'], 'ab',
    np.array([1.2, 2.3, 4.5]),
    np.array([[1.2, 2.3, 4.5], [5.6, 7.8, 8.9]]),
    np.array([np.nan, np.inf, 3]),
    [[6, 8], [9, 10, 11]],
    {'a': 3, 2: 'tata'},
    {'a': 3, 2: np.array([2, 3, 4])},
    {'a': 4, 2: 'not tata'},
)



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
    """
    Test that :func:`utils.maxes` works in the simplest case.
    """
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
    """
    Test that :func:`utils.maxes` works when provided with a key.
    """
    assert utils.maxes(iter_type(values), key=lambda x: -x) == ref


@pytest.mark.parametrize('alpha', string.ascii_letters)
@pytest.mark.parametrize('template', (
    '{}bc', ' {}bc', '123{}WS', '@#$%{}*&^', '\U0001F62E{}',
))
def test_first_alpha(template, alpha):
    """
    Test that :func:`utils.first_alpha` works as expected.
    """
    value = template.format(alpha)
    assert utils.first_alpha(value) == alpha


@pytest.mark.parametrize('value', ('', '0123', '\U0001F62E'))
def test_first_alpha_value_error(value):
    """
    Make sure :func:`utils.first_alpha` raise a ValueError when expected to.
    """
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
    """
    Make sure that :func:`utils.are_all_equal` works as expected.
    """
    assert utils.are_all_equal(iter_type(values)) is ref


@pytest.mark.parametrize('values, ref', (
    ([list([0, 1, 2]) for _ in range(3)], True),
    ([[0, 1, 2], [3, 4, 5], [6, 7, 8]], False),
))
def test_are_all_equal_multidim_list(values, ref):
    """
    Test :func:`utils.are_all_equal` works on nested lists.
    """
    assert utils.are_all_equal(values) is ref


def test_are_all_equal_not_implemented():
    """
    Test that :func:`utils.are_all_equal` raises an exception on
    multidimensional arrays.
    """
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
    x = distance * np.sin(theta) * np.cos(phi)  # pylint: disable=invalid-name
    y = distance * np.sin(theta) * np.sin(phi)  # pylint: disable=invalid-name
    z = distance * np.cos(theta)  # pylint: disable=invalid-name
    return shift, np.array([x, y, z]) + shift, distance


@given(vector_with_random_distance())
@example((np.zeros((3,)), np.zeros((3,)), 0))
def test_distance(vec_and_dist):
    """
    Test the results of :func:`utils.distance`.
    """
    point1, point2, distance = vec_and_dist
    assert_allclose(utils.distance(point1, point2), distance)


@pytest.mark.parametrize(
    'left, right',
    itertools.combinations(DIFFERENCE_USE_CASE, 2)
)
def test_are_different(left, right):
    """
    Test that :func:`are_different` identify different values.
    """
    assert utils.are_different(left, right)


@pytest.mark.parametrize('left', DIFFERENCE_USE_CASE)
def test_not_are_different(left):
    """
    Test that :func:`are_different` returns False for equal values.
    """
    assert not utils.are_different(left, left)
