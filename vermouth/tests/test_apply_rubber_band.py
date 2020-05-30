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
Test the ApplyRubberBand processor and the related functions.
"""
import pytest
import numpy as np
import networkx as nx
from vermouth.processors import apply_rubber_band


@pytest.fixture
def disconnected_graph():
    """
    A graph with two connected components.

    The node keys are integers from 0 to 15 (included). The name of each node
    is its key as a string. The first component is labelled as chain A while
    the second is labelled as chain B.

    0 - 1 - 2 - 3
                |
    4 - 5 - 6 - 7

     8 -  9 - 10 - 11
     |         |    |
    12 - 13 - 14 - 15
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(16))
    graph.add_edges_from([
        # First connected component
        [0, 1], [1, 2], [2, 3], [3, 7], [4, 5], [5, 6], [6, 7],
        # Second connected component
        [8, 9], [8, 12], [8, 9], [9, 10], [10, 14], [10, 11], [11, 15],
        [12, 13], [13, 14], [14, 15],
    ])
    for key, node in graph.nodes.items():
        node['name'] = str(key)
        if key < 8:
            node['chain'] = 'A'
        else:
            node['chain'] = 'B'
    return graph


@pytest.mark.parametrize('separation', (0, 1, 2, 3, 7))
@pytest.mark.parametrize('selection', (
    list(range(16)),  # Use all the nodes, explicitly
    list(range(8)),  # Only the first component
    list(range(8, 16)),  # Only the second component
    list(range(0, 16, 2)),  # Every other nodes
))
def test_build_connectivity_matrix(disconnected_graph, separation, selection):
    connectivity = apply_rubber_band.build_connectivity_matrix(
        disconnected_graph, separation, selection)
    other_component = [np.inf] * 8
    distance = np.array([
        #0  1  2  3  4  5  6  7    8 to 15
        [0, 0, 1, 2, 6, 5, 4, 3] + other_component,  # 0
        [0, 0, 0, 1, 5, 4, 3, 2] + other_component,  # 1
        [1, 0, 0, 0, 4, 3, 2, 1] + other_component,  # 2
        [2, 1, 0, 0, 3, 2, 1, 0] + other_component,  # 3
        [6, 5, 4, 3, 0, 0, 1, 2] + other_component,  # 4
        [5, 4, 3, 2, 0, 0, 0, 1] + other_component,  # 5
        [4, 3, 2, 1, 1, 0, 0, 0] + other_component,  # 6
        [3, 2, 1, 0, 2, 1, 0, 0] + other_component,  # 7
        # 0 to 7           8  9 10 11 12 13 14 15
        other_component + [0, 0, 1, 2, 0, 1, 2, 3],  # 8
        other_component + [0, 0, 0, 1, 1, 2, 1, 2],  # 9
        other_component + [1, 0, 0, 0, 2, 1, 0, 1],  # 10
        other_component + [2, 1, 0, 0, 3, 2, 1, 0],  # 11
        other_component + [0, 1, 2, 3, 0, 0, 1, 2],  # 12
        other_component + [1, 2, 1, 2, 0, 0, 0, 1],  # 13
        other_component + [2, 1, 0, 1, 1, 0, 0, 0],  # 14
        other_component + [3, 2, 1, 0, 2, 1, 0, 0],  # 15
    ])
    np.fill_diagonal(distance, np.inf)
    expected_connectivity = (distance <= separation)
    if selection is not None:
        expected_connectivity = expected_connectivity[:, selection][selection]
    assert np.all(connectivity == expected_connectivity)


@pytest.mark.parametrize('selection', (
        list(range(16)),  # Use all the nodes, explicitly
        list(range(8)),  # Only the first component
        list(range(8, 16)),  # Only the second component
        list(range(0, 16, 2)),  # Every other nodes
))
@pytest.mark.parametrize('extra_edges', ([], [(7, 10)]))
def test_build_pair_matrix(disconnected_graph, selection, extra_edges):
    """
    The creation of a pair matrix works as expected.

    Here, the criterion is ``True`` when nodes belong to the same domain.
    The graph is defined as having two domains: one per chain. The extra edges
    allow to make sure the connectivity does not impact the domain detection.
    """
    def have_same_chain(graph, left, right):
        return graph.nodes[left]['chain'] == graph.nodes[right]['chain']

    expected = np.zeros((16, 16), dtype=bool)
    expected[:8, :8] = True
    expected[8:, 8:] = True
    np.fill_diagonal(expected, False)
    expected = expected[:, selection][selection]

    disconnected_graph.add_edges_from(extra_edges)
    domains = apply_rubber_band.build_pair_matrix(
        disconnected_graph, have_same_chain, selection)
    assert np.all(domains == expected)

@pytest.mark.parametrize('selection', (
        None, # Use all nodes
        list(range(16)),  # Use all the nodes, explicitly
        list(range(8)),  # Only the first component
        list(range(8, 16)),  # Only the second component
        list(range(0, 16, 2)),  # Every other nodes
))
def test_apply_section(disconnected_graph, selection):
    """
    Test if the selection criterion and the default for
    None is applied correctly.
    """
    selected_nodes = apply_rubber_band._apply_selection(disconnected_graph, selection)
    if selection == None:
       selection = disconnected_graph.nodes
    assert set(selected_nodes) == set(selection)

@pytest.mark.parametrize('nodes, edges, outcome', (
    ([1, 2, 3],
     [(1, 2), (2, 3)],
     True),
    ([1, 2, 3],
     [(2, 3)],
     False)
))
def test_are_connected(nodes, edges, outcome):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    assert are_connected(graph, 1, 2, 1) == outcome

@pytest.mark.parametrize('nodes, edges, chain, outcome', (
    ([1, 2, 3],
     {1:"A", 2:"A", 3:"C"},
     [(1, 2), (2, 3)],
     True),
    ([1, 2, 3],
     {1:"A", 2:"B", 3:"C"},
     [(1, 2), (2, 3)],
     False)
))
def test_same_chain(nodes, edges, chain, outcome):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    nx.set_node_attributes(graph, chain, "chain")
    assert same_chain(graph, 1, 1) == outcome

def test_compute_force_constants():
     compute_force_constants(distance_matrix, lower_bound, upper_bound,
                             decay_factor, decay_power, base_constant,
                            minimum_force)


def test_self_distance_matrix(coordinates):
     self_distance_matrix(coordinates)


def test_compute_decay():
    compute_decay(distance, shift, rate, power)
