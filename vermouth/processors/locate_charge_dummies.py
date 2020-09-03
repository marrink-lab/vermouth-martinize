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
Provides a processor that generates positions for every charge dummy.
"""
import itertools
import operator

import numpy as np

from .processor import Processor

DEFAULT_DUMMY_ATTRIBUTE = 'charge_dummy'


def fibonacci_sphere(n_samples):
    """
    Place points near-evenly distributed on a sphere.

    Use the Fibonacci sphere algorithm to place 'n_samples' points at the
    surface of a sphere of radius 1, centered on the origin.

    Parameters
    ----------
    n_samples: int
        Number of points to place.

    Returns
    -------
    numpy.ndarray
        3D coordinates of the points.
    """
    offset = 2 / n_samples
    increment = np.pi * (3 - np.sqrt(5))
    sample_idx = np.arange(n_samples)
    y = (sample_idx * offset - 1) + offset / 2  # pylint: disable=invalid-name
    r = np.sqrt(1 - y * y)  # pylint: disable=invalid-name
    phi = (sample_idx % n_samples) * increment
    x = np.cos(phi) * r  # pylint: disable=invalid-name
    z = np.sin(phi) * r  # pylint: disable=invalid-name
    return np.stack([x, y, z]).T


def colinear_pair():
    """
    Build two points on a line around the origin at a random orientation.
    """
    vector = np.random.rand(3)
    vector /= np.linalg.norm(vector)
    points = np.stack([np.zeros((3, )), vector])
    points -= vector/2
    return points


def find_anchor(molecule, node_key, attribute_tag=DEFAULT_DUMMY_ATTRIBUTE):
    """
    Find the non-dummy bead to which a charge dummy is anchored.

    Each charge dummy has to be attached to exactly one non-dummy atom. This
    function returns the node key for that non-dummy atom.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule to work on.
    node_key:
        The node key of the charge dummy.
    attribute_tag: str
        The name of the atom attribute used to describe charge dummies.

    Returns
    -------
    collections.abc.Hashable
        The node key of the anchor in the molecule graph.

    Raises
    ------
    ValueError
        Raised if there are no anchor, or more than one anchor, found. Raised
        also if the charge dummy is not a charge dummy.
    """
    dummy = molecule.nodes[node_key]
    if dummy.get(attribute_tag, None) is None:
        msg = 'Node "{}" is not a charge dummy. Check the "{}" node attribute.'
        raise ValueError(msg.format(node_key, attribute_tag))

    # There should be only one anchor, and that anchor is the (hopefully) only
    # neighbor that is not a dummy.
    potential_anchors = [
        neighbor
        for neighbor in molecule.neighbors(node_key)
        if molecule.nodes[neighbor].get(attribute_tag, None) is None
    ]
    if not potential_anchors:
        raise ValueError('No anchor found for dummy bead "{}": {}.'
                         .format(node_key, molecule.nodes[node_key]))
    elif len(potential_anchors) > 1:
        raise ValueError('Too many potential anchors found for dummy "{}" ({} found).'
                         .format(node_key, len(potential_anchors)))

    return potential_anchors[0]


def locate_dummy(molecule, anchor_key, dummy_keys, attribute_tag=DEFAULT_DUMMY_ATTRIBUTE):
    """
    Set the position of a group of charge dummies around a non-dummy anchor.

    The molecule is modified in-place.

    The charge dummies are placed at a distance to the anchor defined in nm by
    their charge dummy attribute, the name of which is given in the
    'attribute_tag' argument.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to work on.
    anchor_key:
        The key of the non-dummy anchor all the charge dummies are connected to.
    dummy_keys: collections.abc.Iterable
        A collection of atom keys for charge dummies to position.
    attribute_tag: str
        Name of the atom attribute that describe charge dummies.
    """
    anchor_position = molecule.nodes[anchor_key].get('position')
    if anchor_position is None:
        msg = 'The anchor of the "{}" dummy ("{}") does not have a position.'
        raise ValueError(msg.format(anchor_key, anchor_position[0]))

    distances = []
    distance_error_keys = []
    for dummy_key in dummy_keys:
        try:
            distances.append(float(molecule.nodes[dummy_key].get(attribute_tag)))
        except ValueError:
            distance_error_keys.append(dummy_key)
    if distance_error_keys:
        msg = ('The following charge dummies have an invalid for their {} '
               'attribute: {}. The values have to be numbers.'
               .format(attribute_tag, ', '.join(distance_error_keys)))
        raise ValueError(msg)
    distances = np.array(distances)

    if len(dummy_keys) == 2:
        points = colinear_pair()
    else:
        points = fibonacci_sphere(len(dummy_keys))
    points *= distances[:, None]

    for dummy_key, position in zip(dummy_keys, points):
        molecule.nodes[dummy_key]['position'] = position + anchor_position


def locate_all_dummies(molecule, attribute_tag=DEFAULT_DUMMY_ATTRIBUTE):
    """
    Set the position of all charge dummies of a molecule.

    The molecule is modified in-place.

    The charge dummies are placed at a distance to the anchor defined in nm by
    their charge dummy attribute, the name of which is given in the
    'attribute_tag' argument.

    Parameters
    ----------
    molecule: vermouth.molecule.Molecule
        The molecule to work on.
    attribute_tag: str
        Name of the atom attribute that describe charge dummies.
    """

    dummies = [
        (find_anchor(molecule, dummy_key, attribute_tag), dummy_key)
        for dummy_key in molecule.nodes
        if molecule.nodes[dummy_key].get(attribute_tag, None) is not None
    ]
    dummies.sort()  # Sort primarily on the anchor key as it appears first

    grouped_by_anchor = itertools.groupby(dummies, key=operator.itemgetter(0))
    for anchor_key, dummy_pairs in grouped_by_anchor:
        dummy_keys = [pair[1] for pair in dummy_pairs]
        locate_dummy(molecule, anchor_key, dummy_keys, attribute_tag)

class LocateChargeDummies(Processor):
    def __init__(self, attribute_tag=DEFAULT_DUMMY_ATTRIBUTE):
        super().__init__()
        self.attribute_tag = attribute_tag

    def run_molecule(self, molecule):
        locate_all_dummies(molecule, attribute_tag=self.attribute_tag)
        return molecule
