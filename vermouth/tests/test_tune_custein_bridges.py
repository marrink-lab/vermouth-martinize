#!/usr/bin/env python3
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

import copy
import pytest
import numpy as np
import networkx as nx
import vermouth.processors.tune_cystein_bridges as tune_cystein_bridges


@pytest.fixture
def molecule_for_pruning():
    graph = nx.Graph([
        ['A', 'B'],
        ['C', 'A'],
        ['A', 'D'],
        ['B', 'D'],
        ['C', 'D'],
        ['E', 'B'],
        ['D', 'F'],
        ['E', 'F'],
        ['G', 'A'],
        ['G', 'C'],
    ])
    return graph


@pytest.fixture
def molecule_pruned(molecule_for_pruning):
    graph = copy.deepcopy(molecule_for_pruning)
    selection_a = ['A', 'B', 'G']
    selection_b = ['C', 'D', 'G']
    tune_cystein_bridges.prune_edges_between_selections(
        graph, selection_a, selection_b
    )
    return graph


@pytest.mark.parametrize('edge', [
    ['A', 'C'], ['A', 'D'], ['B', 'D']
])
def test_prune_edges_between_selections_removed(molecule_pruned, edge):
    assert edge not in molecule_pruned.edges


@pytest.mark.parametrize('edge', [
    ['A', 'B'],  # Both in selection_a
    ['C', 'D'],  # Both in selection_b
    ['B', 'E'],  # E not in selections
    ['D', 'F'],  # F not in selections
    ['E', 'F'],  # None of E and F in selections
])
def test_prune_edges_between_selections_kept(molecule_pruned, edge):
    assert edge in molecule_pruned.edges


@pytest.fixture
def simple_protein():
    graph = nx.Graph()
    graph.add_nodes_from((
        (0, {'atomname': 'BB', 'resname': 'CYS'}),
        (1, {'atomname': 'SG', 'resname': 'CYS'}),
        (2, {'atomname': 'BB', 'resname': 'OTHER'}),
        (3, {'atomname': 'SG', 'resname': 'OTHER'}),
        (4, {'atomname': 'BB', 'resname': 'CYS'}),
        (5, {'atomname': 'SG', 'resname': 'CYS'}),
        (6, {'atomname': 'BB', 'resname': 'CYS'}),
        (7, {'atomname': 'SG', 'resname': 'CYS'}),
        (8, {'atomname': 'BB', 'resname': 'CYS'}),
        (9, {'atomname': 'SG', 'resname': 'CYS'}),
        (10, {'atomname': 'BB', 'resname': 'CYS'}),
        (11, {'atomname': 'SG', 'resname': 'CYS'}),
        (12, {'atomname': 'BB', 'resname': 'CYS'}),
        (13, {'atomname': 'SG', 'resname': 'CYS'}),
        (14, {'atomname': 'BB', 'resname': 'OTHER'}),
        (15, {'atomname': 'SG', 'resname': 'OTHER'}),
        (16, {'atomname': 'BB', 'resname': 'OTHER'}),
        (17, {'atomname': 'SG', 'resname': 'OTHER'}),
    ))
    graph.add_edges_from((
        # Chain edges connecting the backbone and the side chains nodes
        (0, 1), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9),
        (8, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15),
        (14, 16), (16, 17),
        # Bridges, including that should stay
        (1, 17), (3, 15), (5, 13), (7, 11),
    ))
    return graph


@pytest.fixture
def simple_protein_pruned(simple_protein):
    graph = copy.deepcopy(simple_protein)
    tune_cystein_bridges.remove_cystein_bridge_edges(graph)
    return graph


@pytest.mark.parametrize('edge', ((5, 13), (7, 11)))
def test_remove_cystein_bridge_edges_remove(simple_protein_pruned, edge):
    assert edge not in simple_protein_pruned.edges


@pytest.mark.parametrize('edge', (
    (0, 1), (2, 3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9),
    (8, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15),
    (14, 16), (16, 17), (1, 17), (3, 15),
))
def test_remove_cystein_bridge_edges_kept(simple_protein_pruned, edge):
    assert edge in simple_protein_pruned.edges


@pytest.fixture
def protein_with_coords():
    # This array of coordinates was generated using:
    #    coordinates = (
    #        np.random.uniform(low=-2, high=2, size=(15, 3))
    #        .astype(np.float32)
    #        .round(2)
    #    )
    #    coordinates.tolist()
    coordinates = np.array([
        [-0.4099999964237213, 0.5699999928474426, -1.2000000476837158],   #  0
        [-0.05000000074505806, 1.6799999475479126, 1.0800000429153442],   #  1
        [1.649999976158142, -1.2699999809265137, -0.18000000715255737],   #  2
        [-0.4300000071525574, -0.5, -1.25],                               #  3
        [1.75, -0.28999999165534973, -1.75],                              #  4
        [1.399999976158142, 1.0399999618530273, -0.029999999329447746],   #  5
        [0.25999999046325684, -0.9399999976158142, -0.8899999856948853],  #  6
        [1.159999966621399, -0.10999999940395355, 0.07999999821186066],   #  7
        [0.009999999776482582, 0.5199999809265137, -0.8600000143051147],  #  8
        [-1.909999966621399, 1.5, -0.27000001072883606],                  #  9
        [-1.5800000429153442, 0.5299999713897705, -0.6499999761581421],   # 10
        [1.9900000095367432, -1.7799999713897705, 1.7699999809265137],    # 11
        [1.6100000143051147, -0.03999999910593033, -0.8899999856948853],  # 12
        [0.10000000149011612, 1.1200000047683716, 0.17000000178813934],   # 13
        [0.41999998688697815, -0.5699999928474426, 0.33000001311302185],  # 14
    ])
    graph = nx.Graph()
    graph.add_nodes_from([
        (index, {'coord': position})
        for index, position in enumerate(coordinates)
    ])
    tune_cystein_bridges.add_edges_at_distance(
        graph, 2.0, range(6), range(6, 15), attribute='coord'
    )
    return graph


@pytest.mark.parametrize('edge', [
    (0, 6), (0, 8), (0, 9), (0, 10), (0, 13), (1, 13), (2, 6), (2, 7),
    (2, 12), (2, 14), (3, 6), (3, 8), (3, 10), (3, 14), (4, 6), (4, 7),
    (4, 12), (5, 7), (5, 8), (5, 12), (5, 13), (5, 14)
])
def test_add_edges_at_distance(protein_with_coords, edge):
    assert edge in protein_with_coords.edges


def test_add_edges_at_distance(protein_with_coords):
    assert len(protein_with_coords.edges) == 22
