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

"""
Unit tests for the :mod:`vermouth.edge_tuning` module.
"""

# The redefined-outer-name check from pylint wrongly catches the use of pytest
# fixtures.
# pylint: disable=redefined-outer-name

import copy
import pytest
import numpy as np
import networkx as nx
import vermouth
from vermouth import edge_tuning
from vermouth.molecule import Choice
from vermouth.utils import distance


@pytest.fixture
def molecule_for_pruning():
    """
    Build arbitrary graph to be pruned.
    """
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
    for key, value in graph.nodes.items():
        value['name'] = key
    return graph


@pytest.fixture
def simple_protein():
    """
    Build a protein-like molecule graph with possible cystein bridges.

    The molecule does not have coordinates.
    """
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
def coordinate_array():
    """
    Build an array of coordinates for 36 points.

    The coordinates are random, but preset in the sense that they were rolled
    once and the same array is always returned.
    """
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
def multi_molecules(coordinate_array):
    """
    Build 6 molecules with 6 atoms each. Coordinates are populated unter the
    `coords` attribute key with the coordinates produced by
    :func:`coordinate_array`.
    """
    molecules = []
    for _ in range(6):
        molecule = vermouth.molecule.Molecule()
        molecule.add_nodes_from([(idx, {'atomid': idx, 'resid': 1})
                                 for idx in range(6)])
        molecules.append(molecule)
    iter_nodes = (
        node
        for molecule in molecules
        for node in molecule.nodes.values()
    )
    for idx, (node, coords) in enumerate(zip(iter_nodes, coordinate_array)):
        node['coords'] = coords
        node['serial'] = idx
    return molecules


class TestPruneEdgesBetweenSelection:
    """
    Tests for :func:`edge_tuning.prune_edges_between_selections`.
    """
    @staticmethod
    @pytest.fixture
    def molecule_pruned(molecule_for_pruning):
        """
        Graph with edges pruned by
        :func:`edge_tuning.prune_edges_between_selections`.
        """
        graph = copy.deepcopy(molecule_for_pruning)
        selection_a = ['A', 'B', 'G']
        selection_b = ['C', 'D', 'G']
        edge_tuning.prune_edges_between_selections(
            graph, selection_a, selection_b
        )
        return graph

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ['A', 'C'], ['A', 'D'], ['B', 'D']
    ])
    def test_prune_edges_between_selections_removed(molecule_pruned, edge):
        """
        Make sure that the edges pruned by
        :func:`edge_tuning.prune_edges_between_selections` are not present
        in the final graph.
        """
        assert edge not in molecule_pruned.edges

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ['A', 'B'],  # Both in selection_a
        ['C', 'D'],  # Both in selection_b
        ['B', 'E'],  # E not in selections
        ['D', 'F'],  # F not in selections
        ['E', 'F'],  # None of E and F in selections
    ])
    def test_prune_edges_between_selections_kept(molecule_pruned, edge):
        """
        Make sure edges that should not be pruned by
        edge_tuning.prune_edges_between_selections` are still present in
        the final graph.
        """
        assert edge in molecule_pruned.edges


class TestPrudeEdgesWithSelectors:
    """
    Tests for :func:`edge_tuning.prune_edges_with_selectors`.
    """

    @staticmethod
    @pytest.fixture
    def molecule_pruned_one_selector(molecule_for_pruning):
        """
        Graph with edges pruned by :func:`edge_tuning.prune_edges_with_selectors`
        called with only :func:`dummy_selector_a`.
        """
        edge_tuning.prune_edges_with_selectors(molecule_for_pruning, dummy_selector_a)
        return molecule_for_pruning

    @staticmethod
    @pytest.fixture
    def molecule_pruned_two_selectors(molecule_for_pruning):
        """
        Graph with edges pruned by :func:`edge_tuning.prune_edges_with_selectors`
        called with both :func:`dummy_selector_a` and :func:`dummy_selector_b`.
        """
        edge_tuning.prune_edges_with_selectors(
            molecule_for_pruning, dummy_selector_a, dummy_selector_b
        )
        return molecule_for_pruning

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ['A', 'B'], ['A', 'G'],
    ])
    def test_prune_edges_with_one_selector_removed(molecule_pruned_one_selector, edge):
        """
        Make sure that the edges pruned by
        :func:`edge_tuning.prune_edges_with_selectors` with a single selector
        provided are not present in the final graph.
        """
        assert edge not in molecule_pruned_one_selector.edges

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ['A', 'C'], ['A', 'D'], ['B', 'D'],
    ])
    def test_prune_edges_with_two_selectors_removed(molecule_pruned_two_selectors, edge):
        """
        Make sure that the edges pruned by
        :func:`edge_tuning.prune_edges_with_selectors` with two selectors provided
        are not present in the final graph.
        """
        assert edge not in molecule_pruned_two_selectors.edges

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ['A', 'B'],  # Both in selection_a
        ['C', 'D'],  # Both in selection_b
        ['B', 'E'],  # E not in selections
        ['D', 'F'],  # F not in selections
        ['E', 'F'],  # None of E and F in selections
    ])
    def test_prune_edges_with_two_selectors_kept(molecule_pruned_two_selectors, edge):
        """
        Make sure edges that should not be pruned by
        edge_tuning.prune_edges_with_selectors` with two selectors are still
        present in the final graph.
        """
        assert edge in molecule_pruned_two_selectors.edges


class TestAddEdgesAtDistance:
    """
    Tests for :func:`edge_tuning.add_edges_at_distance`.
    """
    @staticmethod
    @pytest.fixture
    def protein_with_coords(coordinate_array):
        """
        Build a graph with coordinates and edges placed by
        :func:`edge_tuning.add_edges_at_distance`.
        """
        graph = nx.Graph()
        graph.add_nodes_from([
            (index, {'coord': position})
            for index, position in enumerate(coordinate_array)
        ])
        edge_tuning.add_edges_at_distance(
            graph, 2.0, range(6), range(6, 37), attribute='coord'
        )
        return graph

    @staticmethod
    @pytest.mark.parametrize('edge', [
        (0, 10), (0, 20), (0, 22), (0, 24), (0, 35), (1, 16), (1, 17), (1, 21),
        (1, 22), (1, 25), (1, 26), (1, 29), (1, 31), (1, 35), (2, 10), (2, 17),
        (2, 22), (2, 24), (2, 25), (2, 35), (3, 10), (3, 13), (3, 17), (3, 20),
        (3, 22), (3, 23), (3, 24), (3, 25), (3, 32), (3, 35), (4, 9), (4, 12),
        (4, 14), (4, 16), (4, 26), (4, 29), (4, 31), (5, 10), (5, 20), (5, 22),
        (5, 24), (5, 32), (5, 35)
    ])
    def test_add_edges_at_distance(protein_with_coords, edge):
        """
        Make sure all the expected edges are added by
        :func:`edge_tuning.add_edges_at_distance`.
        """
        assert edge in protein_with_coords.edges

    @staticmethod
    def test_add_edges_at_distance_num(protein_with_coords):
        """
        Make sure that :func:`edge_tuning.add_edges_at_distance` adds the
        expected number of edges.

        This test passing is only relevant if :func:`test_add_edges_at_distance`
        passes as well.
        """
        assert len(protein_with_coords.edges) == 43


class TestAddInterMoleculeEdges:
    """
    Tests for :func:`add_inter_molecule_edges`.
    """
    @staticmethod
    @pytest.fixture
    def multi_molecules_linked(multi_molecules):
        """
        Merge the molecules from :func:`multi_molecules` using
        :func:`edge_tuning.add_inter_molecule_edges`.
        """
        edges = [
            ((0, 1), (4, 2)),
            ((4, 3), (5, 0)),
            ((2, 0), (3, 1)),
            ((3, 2), (2, 4)),
            ((1, 1), (1, 2)),
        ]
        return edge_tuning.add_inter_molecule_edges(multi_molecules, edges)

    @staticmethod
    def test_add_inter_molecule_edges_nmols(multi_molecules_linked):
        """
        Test that :func:`edge_tuning.add_inter_molecule_edges` produces
        the expected number of molecules.
        """
        print(multi_molecules_linked)
        assert len(multi_molecules_linked) == 3

    @staticmethod
    @pytest.mark.parametrize('mol, edge', (
        (0, (1, 8)),
        (0, (9, 12)),
        (2, (0, 7)),
        (2, (8, 4)),
        (1, (1, 2)),
    ))
    def test_add_inter_molecule_edges_edges(multi_molecules_linked, mol, edge):
        """
        Test that :func:`edge_tuning.add_inter_molecule_edges` creates the
        expected edges.
        """
        assert edge in multi_molecules_linked[mol].edges


class TestPairsUnderThreshold:
    """
    Tests for :func:`edge_tuning.pairs_under_threshold`.
    """
    @staticmethod
    @pytest.fixture
    def pair_selected(multi_molecules):
        """
        Call :func:`edge_tuning.pairs_under_threshold` with twice the same
        selection.
        """
        selection = [
            [0, 1],
            [0, 2],
            [1, 4],
            [2, 5],
            [3, 1],
            [5, 4],
        ]
        return edge_tuning.pairs_under_threshold(
            multi_molecules, 2.0, selection, selection, attribute='coords'
        )

    @staticmethod
    @pytest.fixture
    def assymetric_pair_selected(multi_molecules):
        """
        Call :func:`edge_tuning.pairs_under_threshold` with two different
        selections.
        """
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
        return edge_tuning.pairs_under_threshold(
            multi_molecules, 2.0, selection_a, selection_b, attribute='coords'
        )

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ([0, 1], [0, 2]), ([0, 2], [1, 4]),
        ([0, 2], [2, 5]), ([3, 1], [5, 4]),
        ([0, 1], [2, 5]), ([2, 5], [5, 4]),
    ])
    def test_pairs_under_threshold_symetric(pair_selected, edge):
        """
        Test that :func:`edge_tuning.pairs_under_threshold` select the
        expected pairs when provided twice the same selection.
        """
        assert edge in list(pair[:2] for pair in pair_selected)

    @staticmethod
    def test_pairs_under_threshold_symetric_nedges(pair_selected):
        """
        Test that :func:`edge_tuning.pairs_under_threshold` select the
        expected number of pairs when provided twice the same selection.
        """
        # Each pair is yielded twice. Indeed, both selections are the same leading
        # to symetric pairs.
        assert len(list(pair_selected)) == 12

    @staticmethod
    def test_distance_under_threshold(multi_molecules, assymetric_pair_selected):
        distances = []
        found_distances = []
        for ((mol_a, key_a), (mol_b, key_b), dist) in assymetric_pair_selected:
            node_a = multi_molecules[mol_a].nodes[key_a]
            node_b = multi_molecules[mol_b].nodes[key_b]
            distances.append(distance(node_a['coords'], node_b['coords']))
            found_distances.append(dist)
        assert all(d <= 2.0 for d in distances)
        assert np.allclose(found_distances, distances)

    @staticmethod
    @pytest.mark.parametrize('edge', [
        ([0, 1], [0, 2]), ([1, 4], [0, 2]),
        ([3, 1], [5, 4]), ([0, 1], [2, 5]),
    ])
    def test_pairs_under_threshold_assymetric(assymetric_pair_selected, edge):
        """
        Test that :func:`edge_tuning.pairs_under_threshold` select the
        expected pairs when provided two different selections.
        """
        assert edge in list(pair[:2] for pair in assymetric_pair_selected)

    @staticmethod
    def test_pairs_under_threshold_assymetric_nedges(assymetric_pair_selected):
        """
        Test that :func:`edge_tuning.pairs_under_threshold` select the
        expected number of pairs when provided twice the same selection.
        """
        assert len(list(assymetric_pair_selected)) == 4

    @staticmethod
    def test_empty_selection(multi_molecules):
        """
        Make sure :func:`edge_tuning.pairs_under_threshold` is not failing on
        empty selections.
        """
        assert not list(edge_tuning.pairs_under_threshold(
            multi_molecules, 2.0, [], [], attribute='coords'
        ))


class TestAddEdgesThreshold:
    """
    Tests for :func:`edge_tuning.add_edges_threshold`.
    """
    @staticmethod
    @pytest.fixture
    def multi_molecules_with_edges(multi_molecules):
        """
        Creates multiple molecules connected using
        :func:`edge_tuning.add_edges_threshold`.
        """
        templates_a = [{'serial': Choice([1, 10, 19])}, {'name': 'not there'}]
        templates_b = [{'serial': Choice([2, 17])}, {'serial': 34}]
        return edge_tuning.add_edges_threshold(
            multi_molecules, 2.0, templates_a, templates_b, attribute='coords'
        )

    @staticmethod
    def test_add_edges_threshold_nmol(multi_molecules_with_edges):
        """
        Test that :func:`edge_tuning.add_edges_threshold` generate the expected
        number of molecules.
        """
        assert len(multi_molecules_with_edges) == 3

    @staticmethod
    def test_add_edges_threshold_nedges(multi_molecules_with_edges):
        """
        Test that :func:`edge_tuning.add_edges_threshold` generate the expected
        number of edges.
        """
        total_nedges = sum(len(molecule.edges) for molecule in multi_molecules_with_edges)
        assert total_nedges == 4


def test_select_nodes_multi(multi_molecules):
    """
    Test the output of :func:`edge_tuning.select_nodes_multi`.
    """
    selected = list(edge_tuning.select_nodes_multi(multi_molecules, dummy_selector))
    expected = [(molid, atomid) for molid in range(6) for atomid in range(0, 6, 2)]
    assert selected == expected


def dummy_selector_a(node):
    """
    Dummy node selector.
    """
    return node['name'] in ('A', 'B', 'G')


def dummy_selector_b(node):
    """
    Dummy node selector.
    """
    return node['name'] in ('C', 'D', 'G')


def dummy_selector(node):
    """
    Dummy selector that selects nodes with an even atomid.
    """
    return node['atomid'] % 2 == 0
