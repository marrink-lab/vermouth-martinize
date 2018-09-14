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
import vermouth
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
def coordinate_array():
    # This array of coordinates was generated using:
    #    coordinates = (
    #        np.random.uniform(low=-2, high=2, size=(36, 3))
    #        .astype(np.float32)
    #        .round(2)
    #    )
    #    coordinates.tolist()
    return np.array([
        [0.5400000214576721, 1.690000057220459, 1.590000033378601],
        [1.7799999713897705, -0.10000000149011612, 0.41999998688697815],
        [1.9700000286102295, 1.409999966621399, 0.6000000238418579],
        [0.7300000190734863, 1.059999942779541, 1.1100000143051147],
        [1.3700000047683716, -2.0, 0.7799999713897705],
        [0.5899999737739563, 1.4299999475479126, 1.809999942779541],
        [-1.7400000095367432, -0.5400000214576721, 0.8500000238418579],
        [0.4300000071525574, -1.309999942779541, -1.7899999618530273],
        [-1.3300000429153442, 1.5399999618530273, -0.25],
        [0.8799999952316284, -1.9299999475479126, 1.2899999618530273],
        [1.7799999713897705, 1.5299999713897705, 1.690000057220459],
        [-1.7100000381469727, 0.20999999344348907, 0.8199999928474426],
        [0.07999999821186066, -1.1200000047683716, 1.399999976158142],
        [-0.3199999928474426, -0.49000000953674316, 0.9599999785423279],
        [0.7400000095367432, -1.8200000524520874, 1.1799999475479126],
        [1.0399999618530273, -1.7000000476837158, -1.6399999856948853],
        [0.4000000059604645, -1.0700000524520874, 1.25],
        [1.2200000286102295, 1.440000057220459, -0.6399999856948853],
        [-0.47999998927116394, -1.3799999952316284, 1.3200000524520874],
        [-0.46000000834465027, -0.07000000029802322, -0.5199999809265137],
        [0.05000000074505806, 0.5799999833106995, 1.659999966621399],
        [1.090000033378601, -0.17000000178813934, -0.8600000143051147],
        [1.159999966621399, 0.6499999761581421, 0.15000000596046448],
        [-0.3400000035762787, 0.9200000166893005, -0.07000000029802322],
        [0.17000000178813934, 1.2300000190734863, 0.23999999463558197],
        [1.559999942779541, 1.5800000429153442, -0.3100000023841858],
        [1.059999942779541, -0.6700000166893005, -0.4300000071525574],
        [-1.059999942779541, -0.3499999940395355, 1.9299999475479126],
        [-1.6299999952316284, 0.5299999713897705, -0.5299999713897705],
        [1.3799999952316284, -1.2699999809265137, 0.7400000095367432],
        [-1.4900000095367432, 1.6799999475479126, 0.41999998688697815],
        [1.159999966621399, -1.809999942779541, 1.2000000476837158],
        [-0.17000000178813934, -0.28999999165534973, 1.7899999618530273],
        [-1.9500000476837158, -1.25, 1.2200000286102295],
        [-0.05000000074505806, 0.8899999856948853, -1.350000023841858],
        [1.5099999904632568, 1.3300000429153442, 0.11999999731779099],
    ])


@pytest.fixture
def protein_with_coords(coordinate_array):
    graph = nx.Graph()
    graph.add_nodes_from([
        (index, {'coord': position})
        for index, position in enumerate(coordinate_array)
    ])
    tune_cystein_bridges.add_edges_at_distance(
        graph, 2.0, range(6), range(6, 37), attribute='coord'
    )
    return graph


@pytest.mark.parametrize('edge', [
    (0, 10), (0, 20), (0, 22), (0, 24), (0, 35), (1, 16), (1, 17), (1, 21),
    (1, 22), (1, 25), (1, 26), (1, 29), (1, 31), (1, 35), (2, 10), (2, 17),
    (2, 22), (2, 24), (2, 25), (2, 35), (3, 10), (3, 13), (3, 17), (3, 20),
    (3, 22), (3, 23), (3, 24), (3, 25), (3, 32), (3, 35), (4, 9), (4, 12),
    (4, 14), (4, 16), (4, 26), (4, 29), (4, 31), (5, 10), (5, 20), (5, 22),
    (5, 24), (5, 32), (5, 35)
])
def test_add_edges_at_distance(protein_with_coords, edge):
    assert edge in protein_with_coords.edges


def test_add_edges_at_distance(protein_with_coords):
    assert len(protein_with_coords.edges) == 43


@pytest.fixture
def multi_molecules(coordinate_array):
    molecules = []
    for i in range(6):
        molecule = vermouth.molecule.Molecule()
        molecule.add_nodes_from([(idx, {'resid': 1}) for idx in range(6)])
        molecules.append(molecule)
    iter_nodes = (
        node
        for molecule in molecules
        for node in molecule.nodes.values()
    )
    for node, coords in zip(iter_nodes, coordinate_array):
        node['coords'] = coords
    return molecules


@pytest.fixture
def multi_molecules_linked(multi_molecules):
    edges = [
        ((0, 1), (4, 2)),
        ((4, 3), (5, 0)),
        ((2, 0), (3, 1)),
        ((3, 2), (2, 4)),
        ((1, 1), (1, 2)),
    ]
    return tune_cystein_bridges.add_inter_molecule_edges(multi_molecules, edges)


def test_add_inter_molecule_edges_nmols(multi_molecules_linked):
    print(multi_molecules_linked)
    assert len(multi_molecules_linked) == 3


@pytest.mark.parametrize('mol, edge', (
    (0, (1, 8)),
    (0, (9, 12)),
    (2, (0, 7)),
    (2, (8, 4)),
    (1, (1, 2)),
))
def test_add_inter_molecule_edges_edges(multi_molecules_linked, mol, edge):
    assert edge in multi_molecules_linked[mol].edges


@pytest.fixture
def pair_selected(multi_molecules):
    selection = [
        [0, 1],
        [0, 2],
        [1, 4],
        [2, 5],
        [3, 1],
        [5, 4],
    ]
    return tune_cystein_bridges.pairs_under_threshold(
        multi_molecules, 2.0, selection, selection, attribute='coords'
    )


@pytest.mark.parametrize('edge', [
    ([0, 1], [0, 2]), ([0, 2], [1, 4]),
    ([0, 2], [2, 5]), ([3, 1], [5, 4]),
    ([0, 1], [2, 5]), ([2, 5], [5, 4]),
])
def test_pairs_under_threshold_symetric(pair_selected, edge):
    assert edge in list(pair_selected)


def test_pairs_under_threshold_symetric_nedges(pair_selected):
    # Each pair is yielded twice. Indeed, both selections are the same leading
    # to symetric pairs.
    assert len(list(pair_selected)) == 12


@pytest.fixture
def assymetric_pair_selected(multi_molecules):
    selection_a = [
        [0, 1],
        [1, 4],
        [3, 1],
    ]
    selection_b = [
        [0, 2],
        [2, 5],
        [5, 4],
    ]
    return tune_cystein_bridges.pairs_under_threshold(
        multi_molecules, 2.0, selection_a, selection_b, attribute='coords'
    )


@pytest.mark.parametrize('edge', [
    ([0, 1], [0, 2]), ([1, 4], [0, 2]),
    ([3, 1], [5, 4]), ([0, 1], [2, 5]),
])
def test_pairs_under_threshold_assymetric(assymetric_pair_selected, edge):
    assert edge in list(assymetric_pair_selected)


def test_pairs_under_threshold_assymetric_nedges(assymetric_pair_selected):
    assert len(list(assymetric_pair_selected)) == 4
