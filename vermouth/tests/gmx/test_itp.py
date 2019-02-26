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

# Pylint issues false warnings because of pytest's fixtures.
# pylint: disable=redefined-outer-name
# Some of the expected outputs do contain trailing whitespaces.
# pylint: disable=trailing-whitespace

import io
import textwrap
import pytest
import vermouth
from vermouth.gmx.itp import write_molecule_itp
from vermouth.molecule import Interaction, Molecule


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


@pytest.mark.parametrize('pre_meta', (True, False))
@pytest.mark.parametrize('post_meta', (True, False))
def test_pre_post_section_lines(pre_meta, post_meta):
    """
    :func:`vermouth.gmx.itp.write_molecule_itp` writes pre and post section lines.

    If `pre_meta` or `post_meta` is `True`, then "pre_section_lines" or
    "post_section_lines", respectively, is read from `molecule.meta`, otherwise
    it is passed as an argument.
    """
    molecule = Molecule(nrexcl=1)
    molecule.add_nodes_from((
        (0, {
            'atype': 'Q5', 'resid': 1, 'resname': 'VAL', 'atomname': 'BB',
            'charge_group': 1, 'charge': 1,
        }),
        (1, {
            'atype': 'Q5', 'resid': 1, 'resname': 'VAL', 'atomname': 'BB',
            'charge_group': 1, 'charge': 1,
        }),
        (2, {
            'atype': 'Q5', 'resid': 1, 'resname': 'VAL', 'atomname': 'BB',
            'charge_group': 1, 'charge': 1,
        }),
    ))
    # The molecule has 6 interaction sections. The sections are names
    # "interaction_<joint>_<part>", where <joint> can be "with" or "only", and
    # <part> can be "pre", "post", or "pre_post". The <part> indicates what
    # pre_section_lines and post_section_lines are defined for the section.
    # When <joint> in "with", then the section contains 2 interactions in
    # addition to the pre- and post- lines. When <joint> is "only", then the
    # section only has the pre/post lines defined, and no interactions.
    parts = ('pre', 'post', 'pre_post')
    molecule.interactions = {}
    for part in parts:
        name = 'interaction_with_{}'.format(part)
        molecule.interactions[name] = [
            Interaction(atoms=[0, 1], parameters=[name], meta={}),
            Interaction(atoms=[1, 2], parameters=[name], meta={}),
        ]

    pre_section_lines = {
        'atoms': ['test_atoms_0', 'test_atoms_1'],
        'interaction_with_pre': ['test_pre_0', 'test_pre_1', 'test_pre_2'],
        'interaction_with_pre_post': ['test_pre_post_0'],
        'interaction_only_pre': ['test_pre_only_0', 'test_pre_only_1'],
        'interaction_only_pre_post': ['test_prepost_only_0', 'test_prepost_only_1'],
    }
    post_section_lines = {
        'atoms': ['after_atoms_0'],
        'interaction_with_post': ['after_post_0', 'after_post_1'],
        'interaction_with_pre_post': ['after_pre_post_0'],
        'interaction_only_post': ['after_post_only_0', 'after_post_only_1'],
        'interaction_only_pre_post': ['after_prepost_only_0'],
    }
    expected_segments = [
        """
        [ atoms ]
        test_atoms_0
        test_atoms_1
        1 Q5 1 VAL BB 1 1 
        2 Q5 1 VAL BB 1 1 
        3 Q5 1 VAL BB 1 1 
        after_atoms_0
        """,
        """
        [ interaction_with_pre ]
        test_pre_0
        test_pre_1
        test_pre_2
        1 2 interaction_with_pre
        2 3 interaction_with_pre
        """,
        """
        [ interaction_with_post ]
        1 2 interaction_with_post
        2 3 interaction_with_post
        """,
        """
        [ interaction_with_pre_post ]
        test_pre_post_0
        1 2 interaction_with_pre_post
        2 3 interaction_with_pre_post
        after_pre_post_0
        """,
        """
        [ interaction_only_pre ]
        test_pre_only_0
        test_pre_only_1
        """,
        """
        [ interaction_only_post ]
        after_post_only_0
        after_post_only_1
        """,
        """
        [ interaction_only_pre_post ]
        test_prepost_only_0
        test_prepost_only_1
        after_prepost_only_0
        """,
    ]

    if post_meta:
        arg_post = post_section_lines
        molecule.meta['post_section_lines'] = 'invalid'
    else:
        arg_post = None
        molecule.meta['post_section_lines'] = post_section_lines
    if pre_meta:
        arg_pre = pre_section_lines
        molecule.meta['pre_section_lines'] = 'invalid'
    else:
        arg_pre = None
        molecule.meta['pre_section_lines'] = pre_section_lines

    outfile = io.StringIO()
    write_molecule_itp(molecule, outfile, moltype='test',
                       post_section_lines=arg_post, pre_section_lines=arg_pre)
    itp_content = outfile.getvalue()
    # This could be a assert all(...), but it makes it more difficult to
    # understant what is happening in case of a failure.
    for segment in expected_segments:
        assert textwrap.dedent(segment)[:-1] in itp_content
