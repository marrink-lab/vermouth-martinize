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
Contains helper functions for tests.
"""
import operator
import os
import networkx.algorithms.isomorphism as iso


def make_into_set(iter_of_dict):
    """
    Convenience function that turns an iterator of dicts into a set of
    frozenset of the dict items.
    """
    return set(frozenset(dict_.items()) for dict_ in iter_of_dict)


def equal_graphs(g1, g2,
                 node_attrs=('resid', 'resname', 'atomname', 'chain', 'charge_group', 'atype'),
                 edge_attrs=()):
    """
    Parameters
    ----------
    g1: networkx.Graph
    g2: networkx.Graph
    node_attrs: collections.abc.Iterable or None
        Node attributes to consider. If `None`, the node attribute dicts must
        be equal.
    edge_attrs: collections.abc.Iterable or None
        Edge attributes to consider. If `None`, the edge attribute dicts must
        be equal.

    Returns
    -------
    bool
        True if `g1` and `g2` are isomorphic, False otherwise.
    """
    if node_attrs is None:
        node_equal = operator.eq
    else:
        node_equal = iso.categorical_node_match(node_attrs, [''] * len(node_attrs))
    if edge_attrs is None:
        edge_equal = operator.eq
    else:
        edge_equal = iso.categorical_node_match(edge_attrs, [''] * len(edge_attrs))
    matcher = iso.GraphMatcher(g1, g2, node_match=node_equal, edge_match=edge_equal)
    return matcher.is_isomorphic()


def find_in_path(names=('martinize2', 'martinize2.py')):
    """
    Finds and returns the location of one of `names` in PATH, and returns the
    first match.

    Parameters
    ----------
    names: collections.abc.Sequence
        Names to look for in PATH.

    Returns
    -------
    os.PathLike or None
    """
    for folder in os.getenv("PATH", '').split(os.pathsep):
        for name in names:
            fullpath = os.path.join(folder, name)
            if os.path.isfile(fullpath):
                return fullpath