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
Test graph reparation and related operations.
"""
import copy
import logging
import networkx as nx
import numpy as np
import pytest
import vermouth
from vermouth.molecule import Link
import vermouth.forcefield
from .datafiles import PDB_TRI_ALANINE

# pylint: disable=redefined-outer-name

def build_forcefield_with_mods():
    """
    Build a force field that describe some modifications.

    Returns
    -------
    vermouth.ForceField
    """
    nter = Link(name='N-ter')
    nter.add_nodes_from((
        (0, {'atomname': 'CA', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'N', 'PTM_atom': False, 'element': 'N'}),
        (2, {'atomname': 'HN', 'PTM_atom': False, 'element': 'H'}),
        (3, {'atomname': 'H', 'PTM_atom': True, 'element': 'H'}),
    ))
    nter.add_edges_from([[0, 1], [1, 2], [1, 3]])

    cter = Link(name='C-ter')
    cter.add_nodes_from((
        (0, {'atomname': 'C', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'O', 'PTM_atom': False, 'element': 'O'}),
        (2, {'atomname': 'OXT', 'PTM_atom': True, 'element': 'O'}),
    ))
    cter.add_edges_from([[0, 1], [0, 2]])

    gluh = Link(name='GLU-H')
    gluh.add_nodes_from((
        (0, {'atomname': 'CD', 'PTM_atom': False, 'element': 'C'}),
        (1, {'atomname': 'OE1', 'PTM_atom': False, 'element': 'O'}),
        (2, {'atomname': 'OE2', 'PTM_atom': False, 'element': 'O'}),
        (3, {'atomname': 'HE1', 'PTM_atom': True, 'element': 'H'}),
    ))
    gluh.add_edges_from([[0, 1], [0, 2], [1, 3]])

    forcefield = copy.copy(vermouth.forcefield.get_native_force_field('charmm'))
    forcefield.modifications = {}
    for mod in [nter, gluh, cter]:
        forcefield.modifications[mod.name] = mod
    forcefield.renamed_residues[('GLU', ('GLU-H', 'N-ter'))] = 'GLU0'
    return forcefield


def build_system_mod(force_field):
    """
    Build a system with a molecule that comprises modifications.

    Parameters
    ----------
    force_field: vermouth.ForceField
        A force field based on "universal" that desribed the "GLU-H", "C-ter"
        and "N-ter" modifications.

    Returns
    -------
    vermouth.System

    See Also
    --------
    build_forcefield_with_mods
    """
    molecule = vermouth.molecule.Molecule()
    # When the name must be fixed, the expected name is stored under 'expected'.
    # If more than one value is acceptable, then 'expected' is a tuple.
    molecule.add_nodes_from((
        # Residue 1 is a N-terminus (atom 14, H) and has a protonated
        # side chain (atom 13, HE1)
        (0, {'resid': 1, 'resname': 'GLU', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (1, {'resid': 1, 'resname': 'GLU', 'atomname': 'CA', 'chain': 'A', 'element': 'C'}),
        (2, {'resid': 1, 'resname': 'GLU', 'atomname': 'C', 'chain': 'A', 'element': 'C'}),
        (3, {'resid': 1, 'resname': 'GLU', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        (4, {'resid': 1, 'resname': 'GLU', 'atomname': 'CB', 'chain': 'A', 'element': 'C'}),
        (5, {'resid': 1, 'resname': 'GLU', 'atomname': 'HB1', 'chain': 'A', 'element': 'H'}),
        (6, {'resid': 1, 'resname': 'GLU', 'atomname': 'HB2', 'chain': 'A', 'element': 'H'}),
        (7, {'resid': 1, 'resname': 'GLU', 'atomname': 'CG', 'chain': 'A', 'element': 'C'}),
        (8, {'resid': 1, 'resname': 'GLU', 'atomname': 'HG1', 'chain': 'A', 'element': 'H'}),
        (9, {'resid': 1, 'resname': 'GLU', 'atomname': 'HG2', 'chain': 'A', 'element': 'H'}),
        (10, {'resid': 1, 'resname': 'GLU', 'atomname': 'CD', 'chain': 'A', 'element': 'C'}),
        (11, {'resid': 1, 'resname': 'GLU', 'atomname': 'OE2', 'chain': 'A', 'element': 'O'}),
        (12, {'resid': 1, 'resname': 'GLU', 'atomname': 'OE1', 'chain': 'A', 'element': 'O'}),
        (13, {'resid': 1, 'resname': 'GLU', 'atomname': 'HE1', 'chain': 'A', 'element': 'H'}),
        (14, {'resid': 1, 'resname': 'GLU', 'atomname': 'H', 'chain': 'A', 'element': 'H'}),
        (15, {'resid': 1, 'resname': 'GLU', 'atomname': 'HA', 'chain': 'A', 'element': 'H'}),
        (16, {'resid': 1, 'resname': 'GLU', 'atomname': 'HN', 'chain': 'A', 'element': 'H'}),
        # Residue 2 has missing atoms (HA1 and HA2, the hydrogen of the C
        # alpha; HN the hydrogen on the N-ter nitrogen and O, the oxygen at the
        # C-ter). 'common' is an arbitrary attribute name with identical values,
        # which should be propagated to reconstructed atoms.
        (17, {'resid': 2, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N', 'common': 'a'}),
        (18, {'resid': 2, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'C', 'common': 'a'}),
        (19, {'resid': 2, 'resname': 'GLY', 'atomname': 'C', 'chain': 'A', 'element': 'C', 'common': 'a'}),
        # Residue 3 has only N and C specified, everything else is missing.
        (20, {'resid': 3, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (21, {'resid': 3, 'resname': 'GLY', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        # Residue 4 has shuffled and missing names
        (22, {'resid': 4, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'N', 'expected': 'N'}),
        (23, {'resid': 4, 'resname': 'GLY', 'chain': 'A', 'element': 'C', 'expected': 'CA'}),
        (24, {'resid': 4, 'resname': 'GLY', 'atomname': 'HA1', 'chain': 'A', 'element': 'C', 'expected': 'C'}),
        (25, {'resid': 4, 'resname': 'GLY', 'atomname': 'HN', 'chain': 'A', 'element': 'O', 'expected': 'O'}),
        (26, {'resid': 4, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'H', 'expected': ('HA1', 'HA2')}),
        (27, {'resid': 4, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'H', 'expected': ('HA1', 'HA2')}),
        (28, {'resid': 4, 'resname': 'GLY', 'atomname': '', 'chain': 'A', 'element': 'H', 'expected': 'HN'}),
        # Residue 5 has a wrong name in a modification, and a wrong name for
        # the anchor.
        (29, {'resid': 5, 'resname': 'GLY', 'atomname': 'N', 'chain': 'A', 'element': 'N'}),
        (30, {'resid': 5, 'resname': 'GLY', 'atomname': 'CA', 'chain': 'A', 'element': 'C'}),
        # Should be C
        (31, {'resid': 5, 'resname': 'GLY', 'atomname': 'WRONG', 'chain': 'A', 'element': 'C', 'expected': 'C'}),
        (32, {'resid': 5, 'resname': 'GLY', 'atomname': 'O', 'chain': 'A', 'element': 'O'}),
        (33, {'resid': 5, 'resname': 'GLY', 'atomname': 'HN', 'chain': 'A', 'element': 'H'}),
        (34, {'resid': 5, 'resname': 'GLY', 'atomname': 'HA1', 'chain': 'A', 'element': 'H'}),
        (35, {'resid': 5, 'resname': 'GLY', 'atomname': 'HA2', 'chain': 'A', 'element': 'H'}),
        # Should be OXT
        (36, {'resid': 5, 'resname': 'GLY', 'atomname': 'O2', 'chain': 'A', 'element': 'O', 'expected': 'OXT'}),
    ))
    molecule.add_edges_from([
        [0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [4, 6], [4, 7], [7, 8],
        [7, 9], [7, 10], [10, 11], [10, 12], [12, 13], [14, 0], [15, 1], [16, 0],
        [2, 17], [17, 18], [18, 19],
        [19, 20], [21, 22], [22, 23], [23, 24], [24, 25],
        [22, 28], [23, 26], [23, 27], [24, 29], [29, 30], [30, 31], [31, 32],
        [31, 36], [29, 33], [30, 34], [30, 35],
    ])
    for node in molecule:
        molecule.nodes[node]['atomid'] = node
    system = vermouth.System()
    system.molecules = [molecule]
    system.force_field = force_field
    return system


@pytest.fixture
def forcefield_with_mods():
    return build_forcefield_with_mods()


@pytest.fixture
def system_mod(forcefield_with_mods):
    return build_system_mod(forcefield_with_mods)


@pytest.fixture(params=(True, False))
def repaired_graph(request, system_mod):
    vermouth.RepairGraph(include_graph=request.param).run_system(system_mod)
    return system_mod


@pytest.fixture
def canonicalized_graph(repaired_graph):
    vermouth.CanonicalizeModifications().run_system(repaired_graph)
    return repaired_graph


@pytest.fixture
def renamed_graph(canonicalized_graph):
    vermouth.RenameModifiedResidues().run_system(canonicalized_graph)
    return canonicalized_graph


@pytest.mark.parametrize('node_key', (13, 14, 36))
def test_PTM_atom_true(repaired_graph, node_key):
    molecule = repaired_graph.molecules[0]
    assert molecule.nodes[node_key].get('PTM_atom', False)


@pytest.mark.parametrize('node_key', (
    node_key for node_key in range(16) if node_key not in (13, 14)
))
def test_PTM_atom_false(repaired_graph, node_key):
    molecule = repaired_graph.molecules[0]
    assert not molecule.nodes[node_key].get('PTM_atom', False)


def test_uniq_names_repaired(repaired_graph):
    assert len(repaired_graph.molecules) == 1
    molecule = repaired_graph.molecules[0]
    # Only for non-PTM atoms, since those names are not touched by repair graph
    atoms = [(node['resid'], node['atomname'])
             for node in molecule.nodes.values()
             if not node.get('PTM_atom', False)]
    deduplicated = set(atoms)
    assert len(atoms) == len(deduplicated)


def test_uniq_names_canonicalize(canonicalized_graph):
    assert len(canonicalized_graph.molecules) == 1
    molecule = canonicalized_graph.molecules[0]
    atoms = [(node['resid'], node['atomname'])
             for node in molecule.nodes.values()]
    deduplicated = set(atoms)
    assert len(atoms) == len(deduplicated)


def _list_expected_names():
    molecule = build_system_mod(build_forcefield_with_mods()).molecules[0]
    for key, node in molecule.nodes.items():
        if 'expected' in node:
            expected = node['expected']
        else:
            expected = node['atomname']
        if not isinstance(expected, tuple):
            expected = (expected, )
        yield key, expected


@pytest.mark.parametrize('key, expected_names', _list_expected_names())
def test_name_repaired(repaired_graph, key, expected_names):
    molecule = repaired_graph.molecules[0]
    if not molecule.nodes[key].get('PTM_atom', False):
        print(key, molecule.nodes[key])
        assert molecule.nodes[key]['atomname'] in expected_names


@pytest.mark.parametrize('key, expected_names', _list_expected_names())
def test_name_canonicalized(canonicalized_graph, key, expected_names):
    molecule = canonicalized_graph.molecules[0]
    assert molecule.nodes[key]['atomname'] in expected_names


@pytest.mark.parametrize('resid, expected_attrs', [
    (2, {'resid': 2, 'resname': 'GLY', 'chain': 'A', 'common': 'a'})
])
def test_common_attributes(repaired_graph, resid, expected_attrs):
    """Test that attributes that are common to all nodes in a residue end up in
    reconstructed nodes."""
    mol = repaired_graph.molecules[0]
    for node_idx in mol:
        node = mol.nodes[node_idx]
        if node.get('resid') == resid:
            for key, val in expected_attrs.items():
                assert key in node
                assert node[key] == val


def test_renaming(renamed_graph):
    for node in renamed_graph.molecules[0].nodes.values():
        if node['resid'] == 1:
            assert node['resname'] == 'GLU0'
        else:
            assert node['resname'] == 'GLY'


@pytest.mark.parametrize('resid,mutations,modifications,atomnames', [
    (1, ['ALA'], [], 'O C CA HA N HN CB HB1 HB2 HB3'),  # The glutamate chain and N-ter are removed
    (1, [], ['N-ter'], 'O C CA HA N H HN CB HB1 HB2 CG HG1 HG2 CD OE1 OE2'),  # HE1 got removed
    (2, ['ALA'], ['N-ter', 'C-ter'], 'O OXT C CA HA N H HN CB HB1 HB2 HB3'),
    (2, ['GLU'], [], 'O C CA HA N HN CB HB1 HB2 CG HG1 HG2 CD OE1 OE2'),  # Added glutamate sidechain
    (5, ['GLY'], ['none'], 'N CA C O HN HA1 HA2'),  # Remove O2 from C-ter mod
])
def test_repair_graph_with_mutation_modification(system_mod, resid, mutations,
                                                 modifications, atomnames):
    mol = system_mod.molecules[0]
    # Let's mutate res1 to ALA
    for node_idx in mol:
        if mol.nodes[node_idx].get('resid') == resid:
            if mutations:
                mol.nodes[node_idx]['mutation'] = mutations
            if modifications:
                mol.nodes[node_idx]['modification'] = modifications
    mol = vermouth.RepairGraph().run_molecule(mol)
    resid1_atomnames = set()
    for node_idx in mol:
        if mol.nodes[node_idx].get('resid') == resid:
            if mutations:
                assert mol.nodes[node_idx]['resname'] == mutations[0]
            if modifications:
                assert mol.nodes[node_idx].get('modification') == modifications
            resid1_atomnames.add(mol.nodes[node_idx]['atomname'])
    assert resid1_atomnames == set(atomnames.split())


@pytest.mark.parametrize('resid,mutations,modifications', [
    (2, [], ['GLU-H']),  # The glutamate chain and N-ter are removed
    (2, ['ALA', 'LEU'], [])
])
def test_repair_graph_with_mutation_modification_error(system_mod, caplog,
                                                       resid, mutations,
                                                       modifications):
    mol = system_mod.molecules[0]
    # Let's mutate res1 to ALA
    for node_idx in mol:
        if mol.nodes[node_idx].get('resid') == resid:
            if mutations:
                mol.nodes[node_idx]['mutation'] = mutations
            if modifications:
                mol.nodes[node_idx]['modification'] = modifications
    with pytest.raises(ValueError), caplog.at_level(logging.WARNING):
        assert not caplog.records
        mol = vermouth.RepairGraph().run_molecule(mol)
        assert len(caplog.records) == 1


@pytest.mark.parametrize('known_mod_names', [
    [],
    ['C-ter'],
    ['C-ter', 'N-ter'],
    ['GLU-H', 'N-ter'],
])
def test_unknown_mods_removed(caplog, repaired_graph, known_mod_names):
    """
    Tests that atoms that are part of modifications, but are not recognized, get
    removed from the graph by CanonicalizeModifications
    """
    caplog.set_level(logging.WARNING)
    ff = copy.copy(repaired_graph.force_field)
    for mod_name in known_mod_names:
        assert mod_name in ff.modifications  # Purely defensive

    removed_mods = []
    for name, mod in dict(ff.modifications).items():
        if name not in known_mod_names:
            del ff.modifications[name]
            removed_mods.append(mod)

    repaired_graph.force_field = ff
    mol = repaired_graph.molecules[0]

    assert not caplog.records
    assert len(mol) == 46
    vermouth.CanonicalizeModifications().run_system(repaired_graph)

    assert caplog.records

    for record in caplog.records:
        assert record.levelname == 'WARNING'

    assert len(mol) < 46
    atomnames = dict(mol.nodes(data='atomname')).values()
    for mod in removed_mods:
        for node_key in mod.nodes:
            node = mod.nodes[node_key]
            if node['PTM_atom']:
                assert node['atomname'] not in atomnames

    for node_key in mol.nodes:
        node = mol.nodes[node_key]
        if node.get('PTM_atom'):
            contained_by = [mod for mod in ff.modifications.values()
                            if node.get('expected', node['atomname']) in
                               dict(mod.nodes(data='atomname')).values()]
            assert len(contained_by) == 1


def test_tri_alanine_termini():
    """Test that repair_graph preferentially picks non-modification nodes
    This is a good thing, since not all proteins come with their C-term atoms,
    and you *must* pick O/OXT such that O gets the coordinates from the PDB file
    since DSSP can't recognize OXT.
    Prevents recurrence of #317.
    """
    ff = vermouth.forcefield.get_native_force_field('charmm')
    mol = vermouth.pdb.read_pdb(PDB_TRI_ALANINE)[0]
    system = vermouth.system.System(force_field=None)
    system.add_molecule(mol)
    system.force_field = ff
    vermouth.processors.AnnotateMutMod(modifications=[('cter', 'C-ter')]).run_system(system)
    vermouth.processors.RepairGraph().run_system(system)

    mol = system.molecules[0]
    idx_O = next(mol.find_atoms(atomname='O'))
    idx_OXT = next(mol.find_atoms(atomname='OXT'))
    assert not np.any(np.isnan(mol.nodes[idx_O]['position']))
    assert np.all(np.isnan(mol.nodes[idx_OXT].get('position', [np.nan]*3)))


def test_reference_with_resid(system_mod):
    system = system_mod
    ff = copy.deepcopy(system.force_field)
    gly = ff.blocks['GLY']
    for n_idx in gly:
        gly.nodes[n_idx]['resid'] = 42
    ff.blocks['GLY'] = gly
    system.force_field = ff

    vermouth.processors.RepairGraph().run_system(system)
    assert set(nx.get_node_attributes(system.molecules[0], 'resid').values()) == set(range(1, 6))
