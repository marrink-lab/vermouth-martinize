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
Test the writing of ITP file.
"""

import pytest
import vermouth
from vermouth.gmx.itp import write_molecule_itp


@pytest.fixture
def dummy_molecule():
    """
    A virtually empty molecule.
    """
    molecule = vermouth.Molecule()
    molecule.add_nodes_from((
        (0, {
            'atype': 'A', 'resid': 1, 'resname': 'X', 'atomname': 'A',
            'charge_group': 1, 'charge': 0, 'mass': 72
        }),
        (1, {
            'atype': 'A', 'resid': 1, 'resname': 'X', 'atomname': 'A',
            'charge_group': 1, 'charge': 0, 'mass': 72
        }),
    ))
    molecule.meta['moltype'] = 'TEST'
    molecule.nrexcl = 1
    return molecule


def test_no_header(tmpdir, dummy_molecule):
    """
    Test that no header is written if none is provided.
    """
    outpath = tmpdir / 'out.itp'
    with open(str(outpath), 'w') as outfile:
        write_molecule_itp(dummy_molecule, outfile)

    with open(str(outpath)) as infile:
        assert next(infile) == '[ moleculetype ]\n'


def test_header(tmpdir, dummy_molecule):
    """
    Test that the header is written.
    """
    header = (
        'This is a header.',
        'It contains more than one line.',
    )
    expected = (
        '; This is a header.\n',
        '; It contains more than one line.\n',
        '\n',
    )
    outpath = tmpdir / 'out.itp'
    with open(str(outpath), 'w') as outfile:
        write_molecule_itp(dummy_molecule, outfile, header=header)

    with open(str(outpath)) as infile:
        for line, expected_line in zip(infile, expected):
            assert line == expected_line
