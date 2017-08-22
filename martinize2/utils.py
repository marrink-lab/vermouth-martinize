# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:38:17 2017

@author: Peter Kroon
"""

import numpy as np

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
    """
    idx = 0
    while True:
        elem = string[idx]
        if elem.isalpha():
            break
        idx += 1
    return elem


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))

def edges_from_distance(molecule):
    # Note that this will blow up, eat your memory, and leave a sizable and
    # smoking crater if you have too many atoms in your molecule.
    # TODO: Read from data file
    # TODO: Try to use scipy spatial distance functions
    cov_radii = {'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'S': 1.05}
    positions = np.array([molecule.node[n]['position'] for n in molecule])
    elements = [molecule.node[n]['element'] for n in molecule]
    radii = np.array([cov_radii[element] for element in elements])
    thresholds = (radii[:, np.newaxis] + radii[np.newaxis, :]) * 1.1
    vecs = positions[:, np.newaxis] - positions[np.newaxis, :]
    distances = np.sqrt(np.sum(vecs**2, axis=2))
    # distances = np.linalg.norm(vecs, ord=2, axis=2)  # Is supposedly slow
    idxs = np.where((distances < thresholds) & (distances > 0.01))
    weights = 1/distances[idxs]
    molecule.add_weighted_edges_from(zip(idxs[0], idxs[1], weights))