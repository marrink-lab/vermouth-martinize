# Copyright 2020 University of Groningen
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
Runs more elaborate integration tests
"""

from collections import OrderedDict
from pathlib import Path
import shlex
import subprocess
import sys
import numpy as np

import pytest
import vermouth
from vermouth.forcefield import ForceField

from .. import datafiles
from ..helper_functions import find_in_path, parse_gofiles


INTEGRATION_DATA = Path(datafiles.TEST_DATA/'integration_tests')

PATTERN = '{path}/{tier}/{protein}/martinize2/'


MARTINIZE2 = find_in_path()


def assert_equal_blocks(block1, block2, blocknames_equal=True):
    """
    Asserts that two blocks are equal to gain the pytest rich comparisons,
    which is lost when doing `assert block1 == block2`
    """
    if blocknames_equal:
        assert block1.name == block2.name
    assert block1.nrexcl == block2.nrexcl
    assert block1.force_field == block2.force_field  # Set to be equal
    # Assert the order to be equal as well...
    # assert list(block1.nodes) == list(block2.nodes)
    # ... as the attributes
    nodes2 = OrderedDict(block2.nodes(data=True))
    for n_idx, attrs in nodes2.items():
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                nodes2[n_idx][k] = pytest.approx(v, abs=1e-3)
    assert OrderedDict(block1.nodes(data=True)) == nodes2
    edges1 = {frozenset(e[:2]): e[2] for e in block1.edges(data=True)}
    edges2 = {frozenset(e[:2]): e[2] for e in block2.edges(data=True)}
    for e, attrs in edges2.items():
        for k, v in attrs.items():
            if isinstance(v, float):
                attrs[k] = pytest.approx(v, abs=1e-3) # PDB precision is 1e-3

    assert edges1 == edges2
    for inter_type, interactions in block1.interactions.items():
        block2_interactions = block2.interactions.get(inter_type, [])
        assert sorted(interactions, key=lambda i: i.atoms) == sorted(block2_interactions, key=lambda i: i.atoms)


def compare_itp(filename1, filename2):
    """
    Asserts that two itps are functionally identical
    """
    dummy_ff = ForceField(name='dummy')
    with open(filename1) as fn1:
        vermouth.gmx.read_itp(fn1, dummy_ff)
    dummy_ff2 = ForceField(name='dummy')
    with open(filename2) as fn2:
        vermouth.gmx.read_itp(fn2, dummy_ff2)
    for block in dummy_ff2.blocks.values():
        block._force_field = dummy_ff
    assert set(dummy_ff.blocks.keys()) == set(dummy_ff2.blocks.keys())
    for name in dummy_ff.blocks:
        block1 = dummy_ff.blocks[name]
        block2 = dummy_ff2.blocks[name]
        assert_equal_blocks(block1, block2)


def compare_pdb(filename1, filename2):
    """
    Asserts that two pdbs are functionally identical
    """
    pdb1 = vermouth.pdb.read_pdb(filename1)
    pdb2 = vermouth.pdb.read_pdb(filename2)
    assert len(pdb1) == len(pdb2)
    for mol1, mol2 in zip(pdb1, pdb2):
        for mol in (mol1, mol2):
            for n_idx in mol:
                node = mol.nodes[n_idx]
                if 'position' in node and node['atomname'] in ('SCN', 'SCP'):
                    # Charge dummies get placed randomly, which complicated
                    # comparisons to no end.
                    # These will be caught by the distances in the edges instead.
                    del node['position']

        assert_equal_blocks(mol1, mol2)


COMPARERS = {'.itp': compare_itp,
             '.pdb': compare_pdb}

def compare_nbparams(fileref, filecomp):
    """
    Asserts that two go_nbparams.itp files are functionally identical.
    """
    ref = parse_gofiles(fileref)
    compare = parse_gofiles(filecomp)
    
    assert set(ref.keys()) == set(compare.keys())  # assert correct nb pairs

    for key in ref.keys(): 
        assert compare[key] == pytest.approx(ref[key])  # assert correct sigma and eps


def compare_goatomtypes(fileref, filecomp):
    """
    Asserts that two go_atomtypes.itp files are functionally identical.
    """
    ref = parse_gofiles(fileref, atomtypes=True)
    compare = parse_gofiles(filecomp, atomtypes=True)
    
    assert set(ref.keys()) == set(compare.keys())  # assert correct atomtypes

    for key in ref.keys(): 
        assert ref[key] == compare[key]  #assert correct atom definition string

def compare_written_contact_map(fileref, filecomp):
    with open(fileref) as f0:
        ref_lines = f0.readlines()

    with open(filecomp) as f1:
        comp_lines = f1.readlines()

    # compare from line 1 otherwise we compare martinize version
    for (ref, comp) in zip(ref_lines[1:], comp_lines[1:]):
        assert ref == comp

GOCOMPARERS = {'go_nbparams.itp': compare_nbparams,
               'virtual_sites_nonbond_params.itp': compare_nbparams,
               'go_atomtypes.itp': compare_goatomtypes,
               'virtual_sites_atomtypes.itp': compare_goatomtypes,
               'martinize_contact_map.out': compare_written_contact_map}

def _interaction_equal(interaction1, interaction2):
    """
    Returns True if interaction1 == interaction2, ignoring rounding errors in
    interaction parameters.
    """
    p1 = list(map(float, interaction1.parameters))
    try:
        p2 = list(map(float, interaction2.parameters))
    # position restraints are now defined using #ifndef statements, which do not map to floats.
    # i.e. interaction2.parameters = ['1', 'POSRES_FC', 'POSRES_FC', 'POSRES_FC']
    # except the error and write out the interaction explicitly iff the current interaction is a position restraint
    # TODO parse define statements in itps properly
    except ValueError:
        if interaction2.parameters[1] == 'POSRES_FC':
            p2 = list(map(float, ['1', '1000', '1000', '1000']))

    return interaction1.atoms == interaction2.atoms \
           and interaction1.meta == interaction2.meta \
           and pytest.approx(p1) == p2


@pytest.mark.parametrize("tier, protein", [
    ['tier-0', 'mini-protein1_betasheet'],
    ['tier-0', 'mini-protein2_helix'],
    ['tier-0', 'mini-protein3_trp-cage'],
    ['tier-0', 'dipro-termini'],
    ['tier-1', 'bpti'],
    ['tier-1', 'lysozyme'],
    ['tier-1', 'lysozyme_prot'],
    ['tier-1', 'lysozyme_GO'],
    ['tier-1', 'lysozyme_GObias'],
    ['tier-1', 'lysozyme_ENbias'],
    ['tier-1', 'villin'],
    ['tier-1', '3i40'],
    ['tier-1', '6LFO_gap'],
    ['tier-1', '1mj5'],
    ['tier-1', '1mj5-charmm'],
    ['tier-1', 'EN_chain'],
    ['tier-1', 'EN_region'],
    ['tier-1', 'hst5'],
    ['tier-1', '1UBQ'],
    ['tier-1', 'lysozyme_GO_internal']
#   ['tier-2', 'barnase_barstar'],
#   ['tier-2', 'dna'],
#   ['tier-2', 'gpa_dimer'],
])
def test_integration_protein(tmp_path, monkeypatch, tier, protein):
    """
    Runs integration tests by executing the contents of the file `command` in
    the folder tier/protein, and tests whether the contents of the produced
    files are the same as the reference files. The comparison of the files is
    governed by `COMPARERS`.

    Parameters
    ----------
    tmp_path
    tier: str
    protein: str
    """
    monkeypatch.chdir(tmp_path)

    data_path = Path(PATTERN.format(path=INTEGRATION_DATA, tier=tier, protein=protein))
    with open(str(data_path / 'command')) as cmd_file:
        command = cmd_file.read().strip()
    assert command  # Defensive
    command = shlex.split(command)
    result = [sys.executable]
    for token in command:
        if token.startswith('martinize2'):  # Could be martinize2.py
            result.append(str(MARTINIZE2))
        elif token.startswith('.'):
            result.append(str(data_path / token))
        else:
            result.append(token)
    command = result

    # read the citations that are expected
    citations = []
    with open(str(data_path/'citation')) as cite_file:
        for line in cite_file:
            citations.append(line.strip())
    print(command)
    proc = subprocess.run(command, cwd='.', timeout=60, check=False,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)
    exit_code = proc.returncode
    if exit_code:
        print(proc.stdout)
        print(proc.stderr)
        assert not exit_code

    # check if strdout has citations in string
    for citation in citations:
        assert citation in proc.stderr

    files = list(tmp_path.iterdir())

    assert files
    assert list(tmp_path.glob('*.itp')), files
    assert list(tmp_path.glob('*.pdb')), files

    for new_file in tmp_path.iterdir():
        filename = new_file.name
        reference_file = data_path/filename
        assert reference_file.is_file()
        if filename in GOCOMPARERS:
            # compare the extra go/vs model files
            GOCOMPARERS[filename](str(reference_file), str(new_file))
        else:
            ext = new_file.suffix.lower()
            if ext in COMPARERS:
                with monkeypatch.context() as m:
                    # Compare Interactions such that rounding errors in the
                    # parameters are OK.
                    m.setattr(vermouth.molecule.Interaction, '__eq__', _interaction_equal)
                    COMPARERS[ext](str(reference_file), str(new_file))
