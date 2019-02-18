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

import networkx as nx
import pytest

from vermouth.processors import DoMapping, RepairGraph, CanonicalizeModifications, PDBInput, MakeBonds
from vermouth.pdb.pdb import read_pdb
from vermouth import System
from vermouth.map_input import read_mapping_directory
from vermouth.forcefield import ForceField
from vermouth.tests.datafiles import (
    PDB_ALA5,
    PDB_ALA5_CG,
    FF_UNIVERSAL_TEST,
    FF_PEPPLANE,
    MAP_UNIVERSAL_TEST_PEPPLANE
)
from vermouth.tests.helper_functions import equal_graphs


@pytest.fixture(scope='session')
def force_fields():
    ffs = {}
    ffs['universal-test'] = ForceField(FF_UNIVERSAL_TEST)
    ffs['pepplane'] = ForceField(FF_PEPPLANE)
    assert set(ffs.keys()) == {'universal-test', 'pepplane'}
    assert set(ffs['universal-test'].modifications.keys()) >= {'C-ter', 'N-ter'}
    assert set(ffs['universal-test'].blocks.keys()) >= {'ALA',}
    return ffs


@pytest.fixture(scope='session')
def mappings(force_fields):
    maps = read_mapping_directory(MAP_UNIVERSAL_TEST_PEPPLANE, force_fields)
    assert set(maps.keys()) == {'universal-test',}
    assert set(maps['universal-test'].keys()) == {'pepplane',}
    assert set(maps['universal-test']['pepplane'].keys()) == {('AA1', 'AA2'), ('C-ter',), ('N-ter',)}
    return maps


@pytest.fixture(scope='session')
def ala5_aa(force_fields):
    system = System()
    PDBInput(PDB_ALA5).run_system(system)
    system.force_field = force_fields['universal-test']
    MakeBonds().run_system(system)
    RepairGraph().run_system(system)
    CanonicalizeModifications().run_system(system)
    return system


@pytest.fixture(scope='session')
def ala5_cg(force_fields):
    system = System()
    PDBInput(PDB_ALA5_CG).run_system(system)
    system.force_field = force_fields['pepplane']
    return system


def test_pepplane_mapping(force_fields, mappings, ala5_aa, ala5_cg):
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
    assert equal_graphs(aa_mol, cg_mol, node_attrs=attrnames)
