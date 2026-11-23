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
Test the writing of TOP file.
"""
import os
import pytest
import vermouth
from itertools import product
from string import ascii_letters
from vermouth.file_writer import DeferredFileWriter
from vermouth.gmx.topology import (sigma_epsilon_to_C6_C12,
                                   write_atomtypes,
                                   write_nonbond_params,
                                   write_gmx_topology,
                                   remove_duplicate_parameters,
                                   Atomtype,
                                   NonbondParam)
from vermouth.tests.gmx.test_itp import dummy_molecule

@pytest.mark.parametrize('atomtypes, expected, C6C12',
        # simple atomtype
        #'Atomtype', 'molecule node sigma epsilon meta
        ((
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n"], False),
        (
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 254.62172908 37693.15402307 \n"], True),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"node":1, "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 \n",
         "B 72 0 A 0.47000000 2.50000000 \n"], False),
        (
        [{"node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {"comment": ["comment"]}}],
        ["[ atomtypes ]\n", 
         "A 72 0 A 0.43000000 2.30000000 ;comment\n"], False),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"group": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"group": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '; g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n'], False
        ),
        (
        [{"node":0, "sigma": 0.43, "epsilon": 2.3, "meta": {"ifdef": "g1"}},
         {"node":1, "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"node":2, "sigma": 0.47, "epsilon": 2.5, "meta": {"ifdef": "g1"}}],
        ['[ atomtypes ]\n', 
         'B 72 0 A 0.44000000 3.30000000 \n', 
         '#ifdef g1\n', 
         'A 72 0 A 0.43000000 2.30000000 \n', 
         'C 72 0 A 0.47000000 2.50000000 \n',
         '#endif'], False
        ),
        ))
def test_atomtypes(tmp_path, dummy_molecule, atomtypes, expected, C6C12):
    """
    Test that the atomtypes directive is properly written.
    """
    dummy_sys = vermouth.system.System("")
    for atype in atomtypes:
        # somewhat hacky workaround because pytest doesn't
        # allow the use of fixtures in parametrize
        atype['molecule'] = dummy_molecule
        dummy_sys.gmx_topology_params['atomtypes'].append(Atomtype(**atype))

    outpath = tmp_path / 'out.itp'
    write_atomtypes(dummy_sys, outpath, C6C12=C6C12)
    DeferredFileWriter().write()

    with open(str(outpath), encoding='utf-8') as infile:
        for line, ref_line in zip(infile, expected):
            assert line == ref_line


@pytest.mark.parametrize('nbparams, expected, C6C12',
        # simple atomtype
        #'Atomtype', 'molecule node sigma epsilon meta
        ((
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 \n"], False),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 254.62172908 37693.15402307 \n"], True),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"atoms": ("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 \n",
         "A C 1 0.47000000 2.50000000 \n"], False),
        # self interaction
        (
        [{"atoms": ("A",), "sigma": 0.43, "epsilon": 2.3, "meta": {}},
         {"atoms": ("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {}}],
        ["[ nonbond_params ]\n", 
         "A A 1 0.43000000 2.30000000 \n",
         "A C 1 0.47000000 2.50000000 \n"], False),
        (
        [{"atoms": ("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"comment": ["comment"]}}],
        ["[ nonbond_params ]\n", 
         "A B 1 0.43000000 2.30000000 ;comment\n"], False),
        (
        [{"atoms":("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"group": "g1"}},
         {"atoms":("B", "C"), "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"atoms":("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {"group": "g1"}}],
        ['[ nonbond_params ]\n', 
         'B C 1 0.44000000 3.30000000 \n', 
         '; g1\n', 
         'A B 1 0.43000000 2.30000000 \n', 
         'A C 1 0.47000000 2.50000000 \n'], False
        ),
        (
        [{"atoms":("A", "B"), "sigma": 0.43, "epsilon": 2.3, "meta": {"ifdef": "g1"}},
         {"atoms":("B", "C"), "sigma": 0.44, "epsilon": 3.3, "meta": {}},
         {"atoms":("A", "C"), "sigma": 0.47, "epsilon": 2.5, "meta": {"ifdef": "g1"}}],
        ['[ nonbond_params ]\n', 
         'B C 1 0.44000000 3.30000000 \n', 
         '#ifdef g1\n', 
         'A B 1 0.43000000 2.30000000 \n', 
         'A C 1 0.47000000 2.50000000 \n',
         '#endif'], False
        ),
        ))
def test_nonbond_params(tmp_path, nbparams, expected, C6C12):
    """
    Test that the atomtypes directive is properly written.
    """
    dummy_sys = vermouth.system.System("")
    for nbparam in nbparams:
        dummy_sys.gmx_topology_params['nonbond_params'].append(NonbondParam(**nbparam))

    outpath = tmp_path / 'out.itp'
    write_nonbond_params(dummy_sys, outpath, C6C12=C6C12)
    DeferredFileWriter().write()

    with open(str(outpath), encoding='utf-8') as infile:
        for line, ref_line in zip(infile, expected):
            assert line == ref_line

@pytest.mark.parametrize('moltypes, includes, molecules', (
    (['molecule_0'], ['molecule_0'], ['molecule_0    1']),
    (['molecule_0', 'molecule_0', 'molecule_1'], ['molecule_0', 'molecule_1'],
     ['molecule_0    2', 'molecule_1    1']),
    # interleaved moltypes keep their order, but each itp is included once
    (['molecule_0', 'molecule_1', 'molecule_0', 'molecule_1'], ['molecule_0', 'molecule_1'],
     ['molecule_0    1', 'molecule_1    1', 'molecule_0    1', 'molecule_1    1']),
))
def test_toplevel_topology(tmp_path, dummy_molecule, moltypes, includes, molecules):
    """
    Make sure the toplevel topology file
    is correctly written. Note that the individual
    molecule itp files are seperately checked in
    gmx itp module or in the previous tests.
    """
    os.chdir(tmp_path)
    system = vermouth.System()
    for moltype in moltypes:
        molecule = dummy_molecule.copy()
        molecule.meta['moltype'] = moltype
        system.add_molecule(molecule)
    # "node": 0, "sigma": 0.43, "epsilon": 2.3, "meta": {}}
    system.gmx_topology_params['atomtypes'].append(Atomtype(node=0,
                                                   molecule=system.molecules[0],
                                                   sigma=0.43,
                                                   meta={},
                                                   epsilon=2.3))
    system.gmx_topology_params['nonbond_params'].append(NonbondParam(atoms=("A", "B"),
                                                                     sigma=0.43,
                                                                     epsilon=2.3,
                                                                     meta={}))
    system.meta["header"] = ['first header comment', 'second header comment']
    outpath = tmp_path / 'out.itp'
    atompath = tmp_path / 'atomtypes.itp'
    nbpath = tmp_path / 'nonbond_params.itp'
    write_gmx_topology(system,
                       outpath,
                       defines=('random', ),
                       itp_paths={"atomtypes": atompath, "nonbond_params": nbpath},
                       # at this level C6C12 doesn't matter; it gets
                       # checked in previous texts
                       C6C12=False)
    DeferredFileWriter().write()

    ref_lines = ['#define random', '#include "martini.itp"',
                 *(f'#include "{moltype}.itp"' for moltype in includes),
                 '', '[ system ]', 'Title of the system', '', '[ molecules ]', *molecules]
    with open(str(outpath), encoding='utf-8') as infile:
        assert [line.strip() for line in infile] == ref_lines
    assert sorted(path.name for path in tmp_path.glob('molecule_*.itp')) == [f'{moltype}.itp' for moltype in includes]

@pytest.mark.parametrize('command, expected',
    (
        [ascii_letters, {"length": len(ascii_letters), "string": ascii_letters}],
        [(ascii_letters*100), {"length": 4004, "string": (ascii_letters*100)[:4000]+" ..."}]

    ))
def test_gromacs_cmd_len(dummy_molecule, tmp_path, command, expected):
    os.chdir(tmp_path)
    system = vermouth.System()
    system.add_molecule(dummy_molecule)
    dummy_molecule.meta['moltype'] = "molecule_0"
    system.meta['header'].extend([command])

    outpath = tmp_path / 'out.itp'

    write_gmx_topology(system,
                       outpath,
                       defines=('random', ),
                       C6C12=False)
    DeferredFileWriter().write()

    with open(str(tmp_path / 'molecule_0.itp'), encoding='utf-8') as infile:
        expected_line = infile.readlines()[0].strip()[2:]

    assert len(expected_line) == expected["length"]

    assert expected_line == expected["string"]


def test_remove_duplicate_parameters():
    """
    Repeated atomtypes and nonbond_params are removed; entries with other
    parameters are kept.
    """
    system = vermouth.System()
    for _ in range(2):
        molecule = vermouth.molecule.Molecule()
        molecule.add_nodes_from([(0, {'atype': 'A'}), (1, {'atype': 'B'})])
        system.add_molecule(molecule)
    system.gmx_topology_params['atomtypes'] = [
        Atomtype(molecule=molecule, node=node, sigma=0.0, epsilon=0.0, meta={})
        for molecule in system.molecules for node in molecule.nodes]
    system.gmx_topology_params['nonbond_params'] = [NonbondParam(('A', 'B'), 0.47, 1.0, {}),
                                                    NonbondParam(('A', 'B'), 0.47, 1.0, {}),
                                                    NonbondParam(('A', 'B'), 0.47, 2.0, {})]

    remove_duplicate_parameters(system)

    atomtypes = [atomtype.molecule.nodes[atomtype.node]['atype']
                 for atomtype in system.gmx_topology_params['atomtypes']]
    assert atomtypes == ['A', 'B']
    assert [param.epsilon for param in system.gmx_topology_params['nonbond_params']] == [1.0, 2.0]
