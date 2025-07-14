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

import collections.abc
import itertools
import numbers
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


class _Filler:
    """
    Utility class for :func:`are_different`.

    An instance of this class is used as filler when comparing iterables that
    may not have the same length.
    """
    pass


def format_atom_string(node, atomid='', chain='', resname='', resid='', atomname=''):
    defaults = dict(atomid=atomid, chain=chain, resname=resname, resid=resid,
                    atomname=atomname)
    defaults.update(node)
    return '{atomid}{chain}-{resname}{resid}:{atomname}'.format(**defaults)


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
    return all(np.all(item == first) for item in iterator)


def are_different(left, right):
    """
    Return True if two values are different from one another.

    Values are considered different if they do not share the same type. In case
    of numerical value, the comparison is done with :func:`numpy.isclose` to
    account for rounding. In the context of this test, `nan` compares equal to
    itself, which is not the default behavior.

    The order of mappings (dicts) is assumed to be irrelevant, so two
    dictionaries are not different if the only difference is the order of the
    keys.
    """
    if left.__class__ != right.__class__:
        return True

    # Because we know that `left` and `right` share the same type, we also know
    # that if `left` is `None`, then `right` is also `None`, so `left` and
    # `right` are NOT different. It is an easy and common case, so we treat it
    # early to avoid extra work.
    if left is None:
        return False

    if isinstance(left, numbers.Number):
        try:
            return not np.isclose(left, right, equal_nan=True)
        except TypeError:
            # Some things pretend to be numbers but cannot go through isclose.
            # It is the case of integers that overflow an int64 for instance.
            return left != right

    if isinstance(left, (str, bytes)):
        return left != right

    filler = _Filler()

    if isinstance(left, collections.abc.Mapping):
        left_key_set = set(left.keys())
        right_key_set = set(right.keys())
        if left_key_set != right_key_set:
            return True
        return any(are_different(left[key], right[key]) for key in left_key_set)

    if isinstance(left, collections.abc.Iterable):
        zipped = itertools.zip_longest(left, right, fillvalue=filler)
        return any(
            item_left is filler or
            item_right is filler or
            are_different(item_left, item_right)
            for item_left, item_right in zipped
        )

    return left != right
