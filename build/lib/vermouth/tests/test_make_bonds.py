# Copyright 2019 University of Groningen
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
Test the MakeBonds processor.
"""

import pytest
from vermouth.forcefield import get_native_force_field
from vermouth.molecule import Molecule
from vermouth.system import System
from vermouth.processors import MakeBonds


@pytest.mark.parametrize('nodes, edges, expected_edges', (
    [
        # No molecules
        [],
        [],
        []
    ],
    [
        # Single molecule with single node
        [[{'element': 'H', 'position': [0, 0, 0]}], ],
        [[], ],
        [{}, ],
    ],
    [
        # Single molecule with two nodes that should be connected, except
        # they're both hydrogens
        [[{'element': 'H', 'position': [0, 0, 0]},
          {'element': 'H', 'position': [0, 0, 0.12]}], ],
        [[], ],
        [{}],
    ],
    [
        # Two molecule with one node each that should be connected, except
        # they're both hydrogens
        [[{'element': 'H', 'position': [0, 0, 0]}, ],
         [{'element': 'H', 'position': [0, 0, 0.12]}], ],
        [[], []],
        [{}, {}],
    ],
[
        # Single molecule with two nodes that should be connected
        [[{'element': 'C', 'position': [0, 0, 0]},
          {'element': 'H', 'position': [0, 0, 0.12]}], ],
        [[], ],
        [{(0, 1): {'distance': 0.12}}, ],
    ],
    [
        # Two molecule with one node each that should be connected, except that
        # one of the atoms is a H.
        [[{'element': 'C', 'position': [0, 0, 0]}, ],
         [{'element': 'H', 'position': [0, 0, 0.12]}], ],
        [[], []],
        [{}, {}],
    ],
    [
        # Two molecule with one node each that should be connected
        [[{'element': 'C', 'position': [0, 0, 0]}, ],
         [{'element': 'C', 'position': [0, 0, 0.12]}], ],
        [[], []],
        [{(0, 1): {'distance': 0.12}}],
    ],
    [
        # Single molecule with two nodes that should not be connected
        [[{'element': 'H', 'position': [0, 0, 0]},
          {'element': 'H', 'position': [0, 0, 0.2]}], ],
        [[], ],
        [{}],
    ],
    [
        # Two molecule with one node each that should not be connected
        [[{'element': 'H', 'position': [0, 0, 0]}, ],
         [{'element': 'H', 'position': [0, 0, 0.2]}], ],
        [[], []],
        [{}, {}],
    ],
    [
        # Two molecule with one node each that should not be connected because
        # the element is not known
        [[{'element': 'H', 'position': [0, 0, 0]}, ],
         [{'element': 'X', 'position': [0, 0, 0]}], ],
        [[], []],
        [{}, {}],
    ],
    [
        # Single molecule with two nodes that should remain connected
        [[{'element': 'H', 'position': [0, 0, 0]},
          {'element': 'H', 'position': [0, 0, 0.2]}], ],
        [[(0, 1, {'attr': True})], ],
        [{(0, 1): {'attr': True}}],
    ],
    [
        # Single molecule with two nodes that should remain connected
        [[{'element': 'H', 'position': [0, 0, 0]},
          {'element': 'H', 'position': [0, 0, 0.2]}], ],
        [[(0, 1, {})], ],
        [{(0, 1): {}}],
    ],
    [
        # Single molecule with four nodes that should be connected
        [[
            {'element': 'H', 'position': [0, 0, 0]},
            {'element': 'C', 'position': [0, 0, 0.145]},
            {'element': 'C', 'position': [0, 0, 0.315]},
            {'element': 'H', 'position': [0, 0, 0.460]},
        ], ],
        [[], ],
        [{(0, 1): {'distance': pytest.approx(0.145)},
          (1, 2): {'distance': pytest.approx(0.170)},
          (2, 3): {'distance': pytest.approx(0.145)}}],
    ],
    [
        # Single molecule with four nodes that should be connected
        [[
            {'element': 'H', 'position': [0, 0, 0]},
            {'element': 'C', 'position': [0, 0, 0.145]},
            {'element': 'C', 'position': [0, 0, 0.315]},
            {'element': 'H', 'position': [0, 0, 0.460]},
        ], ],
        [[(1, 2, {})], ],
        [{(0, 1): {'distance': pytest.approx(0.145)},
          (1, 2): {},
          (2, 3): {'distance': pytest.approx(0.145)}}],
    ],
    [
        # Single molecule with four nodes that should not be connected despite
        # being close enough because they're not connected in the block
        [[
            {'atomname': 'H1', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0]},
            {'atomname': 'C', 'element': 'C', 'resname': 'GLY', 'position': [0, 0, 0.145]},
            {'atomname': 'N', 'element': 'N', 'resname': 'GLY', 'position': [0, 0, 0.315]},
            {'atomname': 'H2', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0.460]},
        ], ],
        [[(0, 1, {}), (2, 3, {})], ],
        [{(0, 1): {},
          (2, 3): {}}],
    ],
    [
        # Single molecule with four nodes that should be connected despite
        # being far away because they're connected in the block
        [[
            {'atomname': 'H1', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0]},
            {'atomname': 'C', 'element': 'C', 'resname': 'GLY', 'position': [0, 0, 0.145]},
            {'atomname': 'CA', 'element': 'C', 'resname': 'GLY', 'position': [0, 0, 1.315]},
            {'atomname': 'H2', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0.460]},
        ], ],
        [[(0, 1, {}), (2, 3, {})], ],
        [{(0, 1): {},
          (1, 2): {'distance': pytest.approx(1.17)},
          (2, 3): {}}],
    ],
    [
        # Single molecule with four nodes that have a resname, but not atomnames
        [[
            {'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0]},
            {'element': 'C', 'resname': 'GLY', 'position': [0, 0, 0.145]},
            {'element': 'N', 'resname': 'GLY', 'position': [0, 0, 0.315]},
            {'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0.460]},
        ], ],
        [[(0, 1, {}), (2, 3, {})], ],
        [{(0, 1): {},
          (1, 2): {'distance': 0.17},
          (2, 3): {}}],
    ],
    [
        # Single molecule with four nodes that have a resname and duplicate
        # atomnames
        [[
            {'atomname': 'H', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0]},
            {'atomname': 'C', 'element': 'C', 'resname': 'GLY', 'position': [0, 0, 0.145]},
            {'atomname': 'N', 'element': 'N', 'resname': 'GLY', 'position': [0, 0, 0.315]},
            {'atomname': 'H', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0.460]},
        ], ],
        [[(0, 1, {}), (2, 3, {})], ],
        [{(0, 1): {},
          (1, 2): {'distance': 0.17},
          (2, 3): {}}],
    ],
))
def test_make_bonds(nodes, edges, expected_edges):
    """
    Test to make sure that make_bonds makes the bonds it is expected to, and not
    too many.
    nodes is a List[List[Dict]], allowing for multiple molecules
    edges is a List[List[Tuple[Int, Int, Dict]]], allowing for
        multiple molecules
    expected_edges is a List[Dict[Tuple[Int, Int], Dict]], allowing for
        multiple molecules
    """
    system = System(force_field=get_native_force_field('charmm'))
    for node_set, edge_set in zip(nodes, edges):
        mol = Molecule()
        mol.add_nodes_from(enumerate(node_set))
        mol.add_edges_from(edge_set)
        system.add_molecule(mol)
    MakeBonds().run_system(system)
    # Make sure number of connected components is the same
    assert len(system.molecules) == len(expected_edges)
    # Make sure that for every molecule found, the edges are correct
    for found_mol, ref_edges in zip(system.molecules, expected_edges):
        assert dict(found_mol.edges) == ref_edges

@pytest.mark.parametrize('nodes, edges, logtype', [
    [
        # Single molecule with two nodes that have an unknown resname
        [
            {'atomname': 'H', 'element': 'H', 'resname': 'XXX', 'position': [0, 0, 0]},
            {'atomname': 'C', 'element': 'C', 'resname': 'XXX', 'position': [0, 0, 0.145]},
        ],
        [(0, 1, {}),],
        'unknown-residue'
    ],
    [
        # Single molecule with two nodes that have duplicate atoms
        [
            {'atomname': 'X', 'element': 'H', 'resname': 'GLY', 'position': [0, 0, 0]},
            {'atomname': 'X', 'element': 'C', 'resname': 'GLY', 'position': [0, 0, 0.145]},

        ],
        [(0, 1, {}),],
        'inconsistent-data'
    ],

])
def test_make_bonds_logs(caplog, nodes, edges, logtype):
    system = System(force_field=get_native_force_field('charmm'))
    mol = Molecule()
    mol.add_nodes_from(enumerate(nodes))
    mol.add_edges_from(edges)
    system.add_molecule(mol)

    MakeBonds().run_system(system)
    for record in caplog.records:
        assert record.type == logtype
