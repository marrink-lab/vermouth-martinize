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
Contains unittests for vermouth.processors.annotate_mut_mod.
"""

import networkx as nx
import pytest
from vermouth.molecule import Molecule
from vermouth.forcefield import ForceField
from vermouth.processors.annotate_mut_mod import (
    parse_residue_spec,
    _subdict,
    _terminal_matches,
    annotate_modifications,
    AnnotateMutMod
)
from vermouth.tests.datafiles import (
    FF_UNIVERSAL_TEST,
)

# pylint: disable=redefined-outer-name

@pytest.fixture
def example_mol():
    mol = Molecule(force_field=ForceField(FF_UNIVERSAL_TEST))
    nodes = [
        {'chain': 'A', 'resname': 'GLY', 'resid': 1},  # 0, R1
        {'chain': 'A', 'resname': 'GLY', 'resid': 2},  # 1, R2
        {'chain': 'A', 'resname': 'GLY', 'resid': 2},  # 2, R2
        {'chain': 'A', 'resname': 'PHE', 'resid': 2},  # 3, R3
        {'chain': 'B', 'resname': 'GLY', 'resid': 1},  # 4, R4
        {'chain': 'B', 'resname': 'GLY', 'resid': 2},  # 5, R5
        {'chain': 'B', 'resname': 'GLY', 'resid': 2},  # 6, R5
        {'chain': 'B', 'resname': 'PHE', 'resid': 2},  # 7, R6
        {'chain': 'A', 'resname': 'CYS', 'resid': 3},  # 8, R7
        {'chain': 'C', 'resname': 'not_protein', 'resid': 4}, # 9, R8
    ]
    mol.add_nodes_from(enumerate(nodes))
    mol.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (4, 7), (3, 8), (3, 9)])
    return mol


@pytest.mark.parametrize('spec,expected', [
    ('', {}),
    ('-', {'chain': ''}),
    ('#', {}),
    ('-#', {'chain': ''}),
    ('A-ALA1', {'chain': 'A', 'resname': 'ALA', 'resid': 1}),
    ('A-ALA#1', {'chain': 'A', 'resname': 'ALA', 'resid': 1}),
    ('ALA1', {'resname': 'ALA', 'resid': 1}),
    ('A-ALA', {'chain': 'A', 'resname': 'ALA'}),
    ('ALA', {'resname': 'ALA'}),
    ('2', {'resid': 2}),
    ('#2', {'resid': 2}),
    ('PO4#3', {'resname': 'PO4', 'resid': 3}),
    ('PO43', {'resname': 'PO', 'resid': 43}),
    ('A-B-C#D#1', {'chain': 'A', 'resid': 1, 'resname': 'B-C#D'}),
    ('ter', {'resname': 'ter'})
])
def test_parse_residue_spec(spec, expected):
    found = parse_residue_spec(spec)
    assert found == expected


@pytest.mark.parametrize('dict1,dict2,expected', [
    ({}, {}, True),
    ({1: 1}, {}, False),
    ({}, {1: 1}, True),
    ({1: 1}, {1: 1}, True),
    ({1: 1}, {1: 1, 2: 2}, True),
    ({1: 1, 2: 2}, {1: 1}, False),
    ({1: 1}, {1: 3, 2: 2}, False),
    ({1: 1, 2: 2}, {1: 1, 2: 3}, False),
])
def test_subdict(dict1, dict2, expected):
    found = _subdict(dict1, dict2)
    assert found == expected


@pytest.mark.parametrize('mutations,expected_mut', [
    ([], {}),
    ([({'chain': 'A', 'resname': 'GLY', 'resid': 1}, 'ALA')], {0: {'mutation': ['ALA']}}),
    (
        [({'resname': 'GLY'}, 'ALA')],
        {0: {'mutation': ['ALA']},
         1: {'mutation': ['ALA']},
         2: {'mutation': ['ALA']},
         4: {'mutation': ['ALA']},
         5: {'mutation': ['ALA']},
         6: {'mutation': ['ALA']},}
    ),
    (
        [({'resid': 2, 'chain': 'B'}, 'ALA')],
        {5: {'mutation': ['ALA']},
         6: {'mutation': ['ALA']},
         7: {'mutation': ['ALA']},}
    ),
    (
        [({'resname': 'cter'}, 'ALA')],
        {5: {'mutation': ['ALA']},
         6: {'mutation': ['ALA']},
         7: {'mutation': ['ALA']},
         8: {'mutation': ['ALA']}},
    ),
    (
        [({'resid': 2, 'chain': 'B'}, 'none')],  # none is not an existing modification...
        {5: {'mutation': ['none']},
         6: {'mutation': ['none']},
         7: {'mutation': ['none']},}
    ),
])
@pytest.mark.parametrize('modifications,expected_mod', [
    ([], {}),
    ([({'chain': 'A', 'resname': 'GLY', 'resid': 1}, 'C-ter')], {0: {'modification': ['C-ter']}}),
    (
        [({'chain': 'A', 'resname': 'GLY', 'resid': 1}, 'C-ter'),
         ({'chain': 'A', 'resname': 'GLY', 'resid': 1}, 'HSD')],
        {0: {'modification': ['C-ter', 'HSD']}}
    ),
    ([({'resname': 'PHE', 'resid': 1}, 'C-ter'),], {}),
    (
        [({'resname': 'PHE', 'resid': 2}, 'C-ter'),],
        {3: {'modification': ['C-ter']},
         7: {'modification': ['C-ter']},}
    ),
    ([({'resname': 'PHE', 'resid': 1}, 'C-ter'),], {}),
    (
        [({'resname': 'nter'}, 'C-ter')],
        {0: {'modification': ['C-ter']},},
    ),
    (
        [({'resname': 'cter', 'chain': 'B'}, 'C-ter'), ({'resname': 'CYS'}, 'HSD')],
        {7: {'modification': ['C-ter']},
         5: {'modification': ['C-ter']},
         6: {'modification': ['C-ter']},
         8: {'modification': ['HSD']}}  # Not a C-ter mod
    )
])
def test_annotate_modifications(example_mol, modifications, mutations, expected_mod, expected_mut):
    annotate_modifications(example_mol, modifications, mutations)
    for node_idx, mods in expected_mod.items():
        assert _subdict(mods, example_mol.nodes[node_idx])
    for node_idx, mods in expected_mut.items():
        assert _subdict(mods, example_mol.nodes[node_idx])


def test_single_residue_mol():
    mol = Molecule(force_field=ForceField(FF_UNIVERSAL_TEST))
    nodes = [
        {'chain': 'A', 'resname': 'A', 'resid': 2},
        {'chain': 'A', 'resname': 'A', 'resid': 2},
    ]
    mol.add_nodes_from(enumerate(nodes))
    mol.add_edges_from([(0, 1)])

    modification = [({'resname': 'A', 'resid': 2}, 'C-ter'),]
    annotate_modifications(mol, modification, [])

    assert mol.nodes[0] == {'modification': ['C-ter'], 'resname': 'A', 'resid': 2, 'chain': 'A'}
    assert mol.nodes[1] == {'modification': ['C-ter'], 'resname': 'A', 'resid': 2, 'chain': 'A'}


@pytest.mark.parametrize('modifications,mutations', [
    ([({'chain': 'A'}, 'M')], []),  # unknown residue name
    ([], [({'resid': 1}, 'M')]),  # unknown modification name
])
def test_annotate_modifications_error(example_mol, modifications, mutations):
    with pytest.raises(NameError):
        annotate_modifications(example_mol, modifications, mutations)


def test_unknown_terminus_match():
    resname = 'xter'
    graph = nx.Graph()
    graph.add_edge(0, 1)
    with pytest.raises(KeyError):
        _terminal_matches(resname, graph, 0)


@pytest.mark.parametrize('mutations,expected_mut', [
    ([], {}),
    ([('A-GLY1', 'ALA')], {0: {'mutation': ['ALA']}}),
    (
        [('GLY', 'ALA')],
        {0: {'mutation': ['ALA']},
         1: {'mutation': ['ALA']},
         2: {'mutation': ['ALA']},
         4: {'mutation': ['ALA']},
         5: {'mutation': ['ALA']},
         6: {'mutation': ['ALA']},}
    ),
    (
        [('B-2', 'ALA')],
        {5: {'mutation': ['ALA']},
         6: {'mutation': ['ALA']},
         7: {'mutation': ['ALA']},}
    )
])
@pytest.mark.parametrize('modifications,expected_mod', [
    ([], {}),
    ([('A-GLY1', 'C-ter')], {0: {'modification': ['C-ter']}}),
    (
        [('A-GLY1', 'C-ter'),
         ('A-GLY1', 'HSD')],
        {0: {'modification': ['C-ter', 'HSD']}}
    ),
    ([('PHE1', 'C-ter'),], {}),
    (
        [('PHE2', 'C-ter'),],
        {3: {'modification': ['C-ter']},
         7: {'modification': ['C-ter']},}
    ),
    ([('PHE1', 'C-ter'),], {}),
])
def test_annotate_mutmod_processor(example_mol, modifications, mutations, expected_mod, expected_mut):
    AnnotateMutMod(modifications, mutations).run_molecule(example_mol)
    for node_idx, mods in expected_mod.items():
        assert _subdict(mods, example_mol.nodes[node_idx])
    for node_idx, mods in expected_mut.items():
        assert _subdict(mods, example_mol.nodes[node_idx])



@pytest.mark.parametrize('node_data, edge_data, expected', [
    (
        [
            {'resname': 'ALA', 'resid': 1},
            {'resname': 'ALA', 'resid': 2},
            {'resname': 'ALA', 'resid': 3}
        ],
        [(0, 1), (1, 2)],
        {1: ['N-ter'], 3: ['C-ter']}
    ),
    (
        [
            {'resname': 'XXX', 'resid': 1},
            {'resname': 'ALA', 'resid': 2},
            {'resname': 'ALA', 'resid': 3}
        ],
        [(0, 1), (1, 2)],
        {3: ['C-ter']}
    ),
    (
        [
            {'resname': 'ALA', 'resid': 1},
            {'resname': 'ALA', 'resid': 2},
            {'resname': 'XXX', 'resid': 3}
        ],
        [(0, 1), (1, 2)],
        {1: ['N-ter']}
    ),
    (
        [
            {'resname': 'ALA', 'resid': 1},
            {'resname': 'XXX', 'resid': 2},
            {'resname': 'ALA', 'resid': 3}
        ],
        [(0, 1), (1, 2)],
        {1: ['N-ter'], 3: ['C-ter']}
    ),
(
        [
            {'resname': 'XXX', 'resid': 1},
            {'resname': 'ALA', 'resid': 2},
            {'resname': 'XXX', 'resid': 3}
        ],
        [(0, 1), (1, 2)],
        {}
    ),

])
def test_nter_cter_modifications(node_data, edge_data, expected):
    """
    Tests that 'nter' and 'cter' specifications only match protein N and C
    termini
    """
    mol = Molecule(force_field=ForceField(FF_UNIVERSAL_TEST))
    mol.add_nodes_from(enumerate(node_data))
    mol.add_edges_from(edge_data)
    modification = [({'resname': 'cter'}, 'C-ter'), ({'resname': 'nter'}, 'N-ter')]

    annotate_modifications(mol, modification, [])

    found = {}
    for node_idx in mol:
        node = mol.nodes[node_idx]
        if 'modification' in node:
            found[node['resid']] = node['modification']

    assert found == expected
