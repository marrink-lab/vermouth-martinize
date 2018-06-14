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
Created on Tue Aug 22 11:38:17 2017

@author: Peter Kroon
"""

import numpy as np

try:
    from scipy.spatial.distance import euclidean as distance
except ImportError:
    def distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))


def maxes(iterable, key=lambda x: x):
    """
    Analogous to ``max``, but returns a list of all maxima.

    >>> all(key(elem) == max(iterable, key=key) for elem in iterable)
    True

    Parameters
    ----------
    iterable
        The iterable for which to find all maxima.
    key: callable
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


def first_alpha(string):
    """
    Returns the first character in ``string`` for which ``str.isalpha`` returns
    ``True``. If this is ``False`` for all characters in ``string``, returns the last
    character.

    Parameters
    ----------
    string: str
        The string in which to look for the first alpha character.

    Returns
    -------
    str
        The first element of ``string`` for which ``str.isalpha`` returns ``True``.

    Raises
    ------
    ValueError
        No alpha character was found in 'string'.
    """
    for elem in string:
        if elem.isalpha():
            return elem
    raise ValueError('No alpha charecters in "{}".'.format(string))


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
    iterator = iter(iterable)
    first = next(iterator, None)
    return all(item == first for item in iterator)
