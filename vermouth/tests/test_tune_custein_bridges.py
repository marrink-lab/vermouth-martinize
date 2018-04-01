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
