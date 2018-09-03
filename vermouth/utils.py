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
Provides several generic utility functions
"""

import string
import numpy as np


# Do not define in the except so the function can be tested.
def _distance(point_1, point_2):
    """
    .. autofunction:: scipy.spatial.distance.euclidean
    """
    return np.sqrt(np.sum((point_1 - point_2)**2))


try:
    from scipy.spatial.distance import euclidean as distance  # pylint: disable=unused-import
except ImportError:
    distance = _distance


def format_atom_string(node, **kwargs):
    node = node.copy()
    node.update(kwargs)
    return '{atomid}{chain}-{resname}{resid}:{atomname}'.format(**node)


def maxes(iterable, key=lambda x: x):
    """
    Analogous to ``max``, but returns a list of all maxima.

    >>> all(key(elem) == max(iterable, key=key) for elem in iterable)
    True

    Parameters
    ----------
    iterable: collections.abc.Iterable
        The iterable for which to find all maxima.
    key: collections.abc.Callable
        This callable will be called on each element of ``iterable`` to evaluate
        it to a value. Return values must support ``>`` and ``==``.

    Returns
    -------
    list
        A list of all maximal values.

    """
    max_key = None
    out = []
    for item in iterable:
        key_val = key(item)
        if max_key is None or key_val > max_key:
            out = [item]
            max_key = key_val
        elif key_val == max_key:
            out.append(item)
    return out


def first_alpha(search_string):
    """
    Returns the first ASCII letter.

    Parameters
    ----------
    string: str
        The string in which to look for the first ASCII letter.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        No ASCII letter was found in 'search_string'.
    """
    for elem in search_string:
        # str.isalpha catches all unicode charaters tagged as "letter"; it is a
        # very broad set of characters.
        if elem in string.ascii_letters:
            return elem
    raise ValueError('No alpha charecters in "{}".'.format(search_string))


def are_all_equal(iterable):
    """
    Returns ``True`` if and only if all elements in `iterable` are equal; and
    ``False`` otherwise.

    Parameters
    ----------
    iterable: collections.abc.Iterable
        The container whose elements will be checked.

    Returns
    -------
    bool
        ``True`` iff all elements in `iterable` compare equal, ``False``
        otherwise.
    """
    try:
        shape = iterable.shape
    except AttributeError:
        pass
    else:
        if len(shape) > 1:
            message = 'The function does not works on multidimension arrays.'
            raise NotImplementedError(message) from None

    iterator = iter(iterable)
    first = next(iterator, None)
    return all(item == first for item in iterator)


def are_different(left, right):
    """
    Return True if two values are different from one another.

    Values are considered different if they do not share the same type. In case
    of numerical value, the comparison is done with :func:`numpy.isclose` to
    account for rounding. In the context of this test, `nan` compares equal to
    itself, which is not the default behavior.
    """
    if not isinstance(left, right.__class__):
        return True

    left = np.asarray(left)
    right = np.asarray(right)

    if left.shape != right.shape:
        return True

    # For numbers, we want an approximate comparison to account for rounding
    # errors; it only works for numbers, though.
    try:
        return not np.all(np.isclose(left, right, equal_nan=True))
    except TypeError:
        return np.any(left != right)
