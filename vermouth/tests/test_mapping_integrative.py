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
Contains integrative tests for the Mapping parser and the DoMapping processor.
"""

from collections import defaultdict
import pytest

from vermouth.processors import (
    DoMapping, RepairGraph, CanonicalizeModifications, PDBInput, MakeBonds
)
from vermouth import System
from vermouth import Molecule
from vermouth.map_input import read_mapping_directory
from vermouth.forcefield import ForceField
from vermouth.tests.datafiles import (
    PDB_ALA5,
    PDB_ALA5_CG,
    FF_UNIVERSAL_TEST,
    FF_PEPPLANE,
    FF_MARTINI_TEST,
    MAP_UNIVERSAL_TEST_PEPPLANE
)
from vermouth.tests.helper_functions import equal_graphs


# Pylint doesn't like pytest fixtures
# pylint: disable=redefined-outer-name


@pytest.fixture(scope='session')
def force_fields():
    """
    Read a bunch of force fields fit for testing
    """
    ffs = {}
    ffs['universal-test'] = ForceField(FF_UNIVERSAL_TEST)
    ffs['pepplane'] = ForceField(FF_PEPPLANE)
    ffs['martini-test'] = ForceField(FF_MARTINI_TEST)
    assert set(ffs.keys()) == {'universal-test', 'pepplane', 'martini-test'}
    assert set(ffs['universal-test'].modifications.keys()) >= {'C-ter', 'N-ter'}
    assert set(ffs['universal-test'].blocks.keys()) >= {'ALA',}
    return ffs


@pytest.fixture(scope='session')
def mappings(force_fields):
    """
    Read a bunch of mappings fit for testing
    """
    maps = read_mapping_directory(MAP_UNIVERSAL_TEST_PEPPLANE, force_fields)
    assert set(maps.keys()) == {'universal-test',}
    assert set(maps['universal-test'].keys()) == {'pepplane', 'martini-test'}
    assert set(maps['universal-test']['pepplane'].keys()) == {('AA1', 'AA2'), ('C-ter',), ('N-ter',)}
    return maps


@pytest.fixture()
def ala5_aa(force_fields):
    """
    Read an atomistic alanine oligomer from a PDB file
    """
    system = System()
    PDBInput(PDB_ALA5).run_system(system)
    system.force_field = force_fields['universal-test']
    MakeBonds().run_system(system)
    RepairGraph().run_system(system)
    CanonicalizeModifications().run_system(system)
    return system


@pytest.fixture()
def ala5_cg(force_fields):
    """
    Read a CG alanine oligomer from a PDB file
    """
    system = System()
    PDBInput(PDB_ALA5_CG).run_system(system)
    system.force_field = force_fields['pepplane']
    return system


@pytest.fixture()
def ser3_aa(force_fields):
    """
    Provide a serine oligomer without coordinates
    """
    mol = Molecule(force_field=force_fields['universal-test'])
    ser_block = force_fields['universal-test'].blocks['SER']
    for _ in range(3):
        mol.merge_molecule(ser_block)
    mol.add_edges_from([(10, 12), (21, 23)])
    sys = System(force_field=force_fields['universal-test'])
    assert sys.force_field == force_fields['universal-test']
    sys.add_molecule(mol)
    return sys


def test_pepplane_mapping(force_fields, mappings, ala5_aa, ala5_cg):
    """
    Test a mapping that crosses residue boundaries and is not based on blocks.
    """
    ala5_aa = ala5_aa.copy()
    ala5_cg = ala5_cg.copy()
    DoMapping(mappings, force_fields['pepplane'], attribute_keep=('chain', 'resid'),
              attribute_must=('resname',)).run_system(ala5_aa)

    assert ala5_aa.force_field == ala5_cg.force_field
    assert len(ala5_aa.molecules) == len(ala5_cg.molecules) == 1
    aa_mol = ala5_aa.molecules[0]
    cg_mol = ala5_cg.molecules[0]
    attrnames = ('atomname', 'resname', 'resid')
    for attrname in attrnames:
        print(aa_mol.nodes(data=attrname))
        print(cg_mol.nodes(data=attrname))
    print(aa_mol.edges)
    print(cg_mol.edges)
    assert equal_graphs(aa_mol, cg_mol, node_attrs=attrnames)


def test_spawned_atoms(force_fields, ser3_aa, mappings):
    """
    Make sure that nodes that are not explicitly mapped to are still added, and
    have the right mapping_weights
    """
    DoMapping(mappings, force_fields['martini-test'],
              attribute_keep=('chain', 'resid'),
              attribute_must=('resname',)).run_system(ser3_aa)
    found = ser3_aa.molecules[0]
    assert found.force_field == force_fields['martini-test']
    assert len(found) == 6
    mapped_atoms_per_resid = defaultdict(set)
    for n_idx in found.nodes:
        node = found.nodes[n_idx]
        if node['atomname'] == 'BB':
            mapped_atoms_per_resid[node['resid']].update(node['mapping_weights'].keys())
    for n_idx in found.nodes:
        node = found.nodes[n_idx]
        if node['atomname'] == 'SC1':
            assert node['mapping_weights'] == {idx: 0 for idx in
                                               mapped_atoms_per_resid[node['resid']]}
