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
