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

import os
from pathlib import Path
import subprocess
import sys

import pytest
import vermouth

from .. import datafiles


INTEGRATION_DATA = Path(datafiles.TEST_DATA/'integration_tests')

PATTERN = '{path}/{tier}/{protein}/martinize2/'


def find_in_path(name='martinize2.py'):
    """
    Finds the first location of `name` in PATH.

    Parameters
    ----------
    name: str
        The filename to find

    Returns
    -------
    pathlib.Path
        The full path of the first found matching file
    """
    for folder in os.getenv('PATH', '').split(';'):
        trialpath = Path(folder) / Path(name)
        if trialpath.exists():
            return trialpath
    raise OSError('File not found in {}'.format(sys.path))

MARTINIZE2 = find_in_path('martinize2.py')  # TODO: turn into pytest cli argument


def assert_equal_blocks(block1, block2):
    """
    Asserts that two blocks are equal to gain the pytest rich comparisons,
    which is lost when doing `assert block1 == block2`
    """
    assert block1.name == block2.name
    assert block1.nrexcl == block2.nrexcl
    # assert block1.force_field == block2.force_field  # Set to be equal
    # Assert the order to be equal as well...
    assert list(block1.nodes) == list(block2.nodes)
    # ... as the attributes
    assert dict(block1.nodes(data=True)) == dict(block2.nodes(data=True))
    edges1 = {frozenset(e[:2]): e[2] for e in block1.edges(data=True)}
    edges2 = {frozenset(e[:2]): e[2] for e in block2.edges(data=True)}
    assert edges1 == edges2
    assert block1.interactions == block2.interactions


def compare_itp(filename1, filename2):
    """
    Asserts that two itps are functionally identical
    """
    dummy_ff = vermouth.forcefield.ForceField(name='dummy')
    with open(filename1) as fn1:
        vermouth.gmx.read_itp(fn1, dummy_ff)
    dummy_ff2 = vermouth.forcefield.ForceField(name='dummy')
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
    assert pdb1 == pdb2


COMPARERS = {'.itp': compare_itp,
             '.pdb': compare_pdb}


@pytest.mark.parametrize("tier, protein", [
    ['tier-0', 'mini-protein1_betasheet'],
    ['tier-0', 'mini-protein2_helix'],
    ['tier-0', 'mini-protein3_trp-cage'],
    # ['tier-1', 'bpti'],
    # ['tier-1', 'lysozyme'],
    # ['tier-1', 'villin'],
    # ['tier-2', 'barnase_barstar'],
    # ['tier-2', 'dna'],
    # ['tier-2', 'gpa_dimer'],
])
def test_integration_protein(tmp_path, tier, protein):
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
    data_path = Path(PATTERN.format(path=INTEGRATION_DATA, tier=tier, protein=protein))

    with open(data_path / 'command') as cmd_file:
        command = cmd_file.read().strip()
    assert command
    command = command.format(inpath=data_path, martinize2=MARTINIZE2)
    command = command.replace('\n', ' ')
    command = '{python} {cmd}'.format(python=sys.executable, cmd=command)

    print(command)

    proc = subprocess.run(command, cwd=tmp_path, shell=True, timeout=60,
                          capture_output=True, text=True, check=False)
    exit_code = proc.returncode
    assert exit_code == 0, (proc.stdout, proc.stderr)

    for new_file in tmp_path.iterdir():
        filename = new_file.name
        reference_file = data_path/filename
        assert reference_file.is_file()
        ext = new_file.suffix.lower()
        if ext in COMPARERS:
            COMPARERS[ext](reference_file, new_file)
