#!/usr/bin/env python3
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
Unit tests for the mapping file parser and its utilities.
"""

# We call "private" methods and functions as part of the tests, which pylint
# does not like. Because we want to test these methods and functions, we have
# to call them outside of there normal "scope".
# pylint: disable=protected-access

# Pylint wrongly complains about the use of pytest fixtures.
# pylint: disable=redefined-outer-name

import collections
import itertools
import textwrap
from pathlib import Path

import pytest

import vermouth
import vermouth.map_input
import vermouth.forcefield

Reference = collections.namedtuple('Reference',
                                   'string name from_ff to_ff mapping weights extra')

# Test system 1: simple case
# from: a-b-c-d-e-f
#       ----- -----
# to:     A     B
SYSTEM_BASIC = Reference(
    string="""\
system1

[atoms]
1 a A
2 b A
3 c A
4 d B
5 e B
6 f B
""",
    name='system1',
    from_ff=['universal'],  # default
    to_ff=['martini22'],  # default
    mapping={
        (0, 'a'): [(0, 'A')],
        (0, 'b'): [(0, 'A')],
        (0, 'c'): [(0, 'A')],
        (0, 'd'): [(0, 'B')],
        (0, 'e'): [(0, 'B')],
        (0, 'f'): [(0, 'B')],
    },
    weights={
        (0, 'A'): {(0, 'a'): 1, (0, 'b'): 1, (0, 'c'): 1},
        (0, 'B'): {(0, 'd'): 1, (0, 'e'): 1, (0, 'f'): 1},
    },
    extra=[],
)

# Test system 1: shared atoms
# from: a-b-c-d-e-f
#       -------
#           -------
# to:     A     B
# c has a greater weight in A, while d has a greater weight in B
SYSTEM_SHARED = Reference(
    string="""\
system2

[atoms]
1 a A
2 b A
3 c A B A  ; more on A (also comment)
4 d A B B  ; more on B
5 e B
6 f B
""",
    name='system2',
    from_ff=['universal'],  # default
    to_ff=['martini22'],  # default
    mapping={
        (0, 'a'): [(0, 'A')],
        (0, 'b'): [(0, 'A')],
        (0, 'c'): [(0, 'A'), (0, 'B'), (0, 'A')],
        (0, 'd'): [(0, 'A'), (0, 'B'), (0, 'B')],
        (0, 'e'): [(0, 'B')],
        (0, 'f'): [(0, 'B')],
    },
    weights={
        (0, 'A'): {(0, 'a'): 1, (0, 'b'): 1, (0, 'c'): 2 / 3, (0, 'd'): 1 / 3},
        (0, 'B'): {(0, 'c'): 1 / 3, (0, 'd'): 2 / 3, (0, 'e'): 1, (0, 'f'): 1},
    },
    extra=[],
)

# Test system 3: unmapped particles
# Same as system 1, but C, D, and E do not have underlying atoms
SYSTEM_EXTRA = Reference(
    string=SYSTEM_BASIC.string + """
[extra]
C ; alone on its line
D E  ; two particles on the same line
""",
    name=SYSTEM_BASIC.name,
    from_ff=SYSTEM_BASIC.from_ff,
    to_ff=SYSTEM_BASIC.to_ff,
    mapping=SYSTEM_BASIC.mapping,
    weights=SYSTEM_BASIC.weights,
    extra=['C', 'D', 'E'],
)

# Test system 4: null-weights
# from: a-b-c-d-e-f
#       -------
#           -------
# to:     A     B
# c as a weight of 0 on B, and d has a weight of 0 on A
SYSTEM_NULL = Reference(
    string="""\
system4

[atoms]
1 a A
2 b A
3 c A !B
4 d !A B
5 e B
6 f B
""",
    name='system4',
    from_ff=['universal'],  # default
    to_ff=['martini22'],  # default
    mapping={
        (0, 'a'): [(0, 'A')],
        (0, 'b'): [(0, 'A')],
        (0, 'c'): [(0, 'A'), (0, 'B')],
        (0, 'd'): [(0, 'A'), (0, 'B')],
        (0, 'e'): [(0, 'B')],
        (0, 'f'): [(0, 'B')],
    },
    weights={
        (0, 'A'): {(0, 'a'): 1, (0, 'b'): 1, (0, 'c'): 1, (0, 'd'): 0},
        (0, 'B'): {(0, 'c'): 0, (0, 'd'): 1, (0, 'e'): 1, (0, 'f'): 1},
    },
    extra=[],
)

# Test system 5: origin and target force fields
# Same as system 1, but with origin and target force field explicitly
# specified.
SYSTEM_FROM_TO = Reference(
    string=SYSTEM_BASIC.string + """

[from]
from_ff
from_ff_2
from_ff_3 from_ff_4

[mapping]
from_ff_5 from_ff_6
from_ff_7

[to]
to_ff
to_ff_2 to_ff_3

""",
    name=SYSTEM_BASIC.name,
    from_ff=['from_ff', 'from_ff_2', 'from_ff_3',
             'from_ff_4', 'from_ff_5', 'from_ff_6', 'from_ff_7'],
    to_ff=['to_ff', 'to_ff_2', 'to_ff_3'],
    mapping=SYSTEM_BASIC.mapping,
    weights=SYSTEM_BASIC.weights,
    extra=SYSTEM_BASIC.extra,
)

# Test system 6: unused field from Backward
# Same as system 1, but with a section that has to be ignored.
SYSTEM_IGNORE = Reference(
    string=SYSTEM_BASIC.string + """
[ chiral ]
A B C D
""",
    name=SYSTEM_BASIC.name,
    from_ff=SYSTEM_BASIC.from_ff,
    to_ff=SYSTEM_BASIC.to_ff,
    mapping=SYSTEM_BASIC.mapping,
    weights=SYSTEM_BASIC.weights,
    extra=SYSTEM_BASIC.extra,
)


@pytest.mark.parametrize(
    'case',
    [SYSTEM_BASIC, SYSTEM_SHARED, SYSTEM_EXTRA, SYSTEM_NULL, SYSTEM_FROM_TO, SYSTEM_IGNORE]
)
def test_read_mapping_partial(case):
    """
    Test that regular mapping files are read as expected.
    """
    full_mapping = vermouth.map_input._read_mapping_partial(case.string.split('\n'), 1)
    name, from_ff, to_ff, mapping, weights, extra, _ = full_mapping
    assert name == case.name
    assert from_ff == case.from_ff
    assert to_ff == case.to_ff
    assert mapping == case.mapping
    assert extra == case.extra
    assert weights == case.weights


@pytest.mark.parametrize('content', (
    """
dummy

[ atoms ]
0 X1 !A A B
    """,  # Inconsistent bead weight
    """
[ molecule
dummy
    """,  # Incomplete section line
    """
dummy

[ atoms ]
0 X1 A B
1 X2 C D
2 X1 Y U
    """,  # Multiple difinitions for the same atom
    """
[ atoms ]
0 A B
    """,  # no molecule name
    """
Just a pile of garbage.
Clearly, this is not a mapping file.
Not even a partial one.
    """,
    # If a file contains two consecutive [ molecule ] lines, it means the first
    # of the molecules is empty.
    """
[ molecule ]
    """,
))
def test_read_mapping_errors(content):
    """
    Test that syntax error are caught when reading a mapping.
    """
    with pytest.raises(IOError):
        vermouth.map_input._read_mapping_partial(content.split('\n'), 1)


@pytest.mark.parametrize(
    'case',
    [SYSTEM_BASIC, SYSTEM_SHARED, SYSTEM_EXTRA, SYSTEM_NULL, SYSTEM_FROM_TO, SYSTEM_IGNORE]
)
def test_read_mapping_file(case):
    """
    Test that regular mapping files are read as expected.
    """
    reference = collections.defaultdict(lambda: collections.defaultdict(dict))
    for from_ff, to_ff in itertools.product(case.from_ff, case.to_ff):
        reference[from_ff][to_ff][case.name] = (
            case.mapping, case.weights, case.extra
        )
    reference = vermouth.map_input._default_to_dict(reference)

    mappings = vermouth.map_input.read_mapping_file(
        ['[ molecule ]'] + case.string.split('\n')
    )

    assert mappings == reference


@pytest.fixture
def reference_multi():
    """
    Build a reference file and mapping collection with multiple molecules.
    """
    content = textwrap.dedent("""
        ; Some line before the first molecule.
        ; Just because.

        [ molecule ]
        dummy_0

        [ atoms ]
        0 X1 A
        1 X2 B

        ; Some mess between two molecules.

        [ molecule ]
        ; A comment just after the molecule section.
        dummy_1

        [ atoms ]
        0 X2 C
        1 X3 D
    """).split('\n')
    reference = {'universal': {'martini22': {
        'dummy_0': (
            {(0, 'X1'): [(0, 'A')], (0, 'X2'): [(0, 'B')]},
            {(0, 'A'): {(0, 'X1'): 1.0}, (0, 'B'): {(0, 'X2'): 1.0}},
            [],
        ),
        'dummy_1': (
            {(0, 'X2'): [(0, 'C')], (0, 'X3'): [(0, 'D')]},
            {(0, 'C'): {(0, 'X2'): 1.0}, (0, 'D'): {(0, 'X3'): 1.0}},
            [],
        ),
    }}}
    return content, reference


def test_read_mapping_file_multiple(reference_multi):
    """
    Test that read_mapping_file can read more than one molecule.
    """
    content, reference = reference_multi
    mappings = vermouth.map_input.read_mapping_file(content)
    assert mappings == reference


@pytest.fixture(scope='session')
def ref_mapping_directory(tmpdir_factory):
    """
    Build a file tree with mapping files.
    """
    basedir = tmpdir_factory.mktemp('data')
    mapdir = basedir.mkdir('mappings')

    template = textwrap.dedent("""
        [ molecule ]
        dummy_{0}

        [ from ]
        {1}

        [ to ]
        {2}

        [ atoms ]
        0 X1{0} A{0} B{0}
        1 X2{0} C{0} D{0}
    """)

    mappings = collections.defaultdict(lambda: collections.defaultdict(dict))

    force_fields_from = ['ff{}'.format(i) for i in range(4)]
    force_fields_to = force_fields_from + ['only_to']
    force_fields_from = force_fields_from + ['only_from']
    iterate_on = itertools.product(force_fields_from, force_fields_to, range(3))
    for idx, (from_ff, to_ff, _) in enumerate(iterate_on):
        mapfile = mapdir / 'file{}.map'.format(idx)
        with open(str(mapfile), 'w') as outfile:
            outfile.write(template.format(idx, from_ff, to_ff))

        mapping = {
            (0, 'X1{}'.format(idx)): [(0, 'A{}'.format(idx)), (0, 'B{}'.format(idx))],
            (0, 'X2{}'.format(idx)): [(0, 'C{}'.format(idx)), (0, 'D{}'.format(idx))],
        }
        weights = {
            (0, 'A{}'.format(idx)): {(0, 'X1{}'.format(idx)): 0.5},
            (0, 'B{}'.format(idx)): {(0, 'X1{}'.format(idx)): 0.5},
            (0, 'C{}'.format(idx)): {(0, 'X2{}'.format(idx)): 0.5},
            (0, 'D{}'.format(idx)): {(0, 'X2{}'.format(idx)): 0.5},
        }
        extra = []
        mappings[from_ff][to_ff]['dummy_{}'.format(idx)] = (mapping, weights, extra)

    mappings = {from_ff: dict(to_ff) for from_ff, to_ff in mappings.items()}

    return Path(str(basedir)), mappings


def test_read_mapping_directory(ref_mapping_directory):
    """
    Test that mapping files from a directory are propely found and read.
    """
    dirpath, ref_mappings = ref_mapping_directory
    mappings = vermouth.map_input.read_mapping_directory(dirpath)
    assert mappings == ref_mappings


def test_read_mapping_directory_not_dir():
    """
    Test that :func:`vermouth.map_input.read_mapping_directory` fails when
    the input is not a directory.
    """
    with pytest.raises(NotADirectoryError):
        vermouth.map_input.read_mapping_directory('not a directory')


def test_read_mapping_directory_error(tmpdir):
    """
    Test that :func:`vermouth.map_input.read_mapping_directory` raises an
    exception when a file could not be read.
    """
    mapdir = Path(str(tmpdir.mkdir('mappings')))
    with open(str(mapdir / 'valid.map'), 'w') as outfile:
        outfile.write(textwrap.dedent("""
            [ molecule ]
            valid
            [ atoms ]
            0 A B
        """))
    with open(str(mapdir / 'not_valid.map'), 'w') as outfile:
        outfile.write('invalid content')
    with pytest.raises(IOError):
        vermouth.map_input.read_mapping_directory(mapdir)


def test_generate_self_mapping():
    """
    Test that :func:`vermouth.map_input.generate_self_mappings` works as
    expected.
    """
    # Build the input blocks
    blocks = {
        'A0': vermouth.molecule.Block([['AA', 'BBB'], ['BBB', 'CCCC']]),
        'B1': vermouth.molecule.Block([['BBB', 'CCCC'], ['BBB', 'E']]),
    }
    for name, block in blocks.items():
        block.name = name
        for atomname, node in block.nodes.items():
            node['atomname'] = atomname
    # Build the expected output
    ref_mappings = {
        'A0': (
            # mapping
            {(0, 'AA'): [(0, 'AA')], (0, 'BBB'): [(0, 'BBB')], (0, 'CCCC'): [(0, 'CCCC')]},
            # weights
            {(0, 'AA'): {(0, 'AA'): 1}, (0, 'BBB'): {(0, 'BBB'): 1}, (0, 'CCCC'): {(0, 'CCCC'): 1}},
            # extra
            [],
        ),
        'B1': (
            # mapping
            {(0, 'BBB'): [(0, 'BBB')], (0, 'CCCC'): [(0, 'CCCC')], (0, 'E'): [(0, 'E')]},
            # weights
            {(0, 'BBB'): {(0, 'BBB'): 1}, (0, 'CCCC'): {(0, 'CCCC'): 1}, (0, 'E'): {(0, 'E'): 1}},
            # extra
            [],
        ),
    }
    # Actually test
    mappings = vermouth.map_input.generate_self_mappings(blocks)
    assert mappings.keys() == ref_mappings.keys()
    assert mappings == ref_mappings


def test_generate_all_self_mappings():
    """
    Test that :func:`vermouth.map_input.generate_all_self_mappings` generate
    the expected entries.
    """
    force_fields = []
    expected = []
    for idx in range(3):
        idx_str = str(idx)
        ff_name = 'ff_' + idx_str
        force_field = vermouth.forcefield.ForceField(name=ff_name)
        force_field.blocks = {
            'A' + idx_str: vermouth.molecule.Block([['AA', 'BBB'], ['BBB', 'CCCC']]),
            'B' + idx_str: vermouth.molecule.Block([['BBB', 'CCCC'], ['BBB', 'E']]),
        }
        for name, block in force_field.blocks.items():
            block.name = name
            for atomname, node in block.nodes.items():
                node['atomname'] = atomname
        force_fields.append(force_field)

        expected.append((ff_name, (ff_name, )))

    mappings = vermouth.map_input.generate_all_self_mappings(force_fields)
    found = [(from_ff, tuple(to_ff.keys())) for from_ff, to_ff in mappings.items()]

    # In python <= 3.5, dicts are not ordered.
    assert set(found) == set(expected)


@pytest.fixture
def base_mappings():
    """
    Build a basic (empty) mapping collection to modify.
    """
    return {
        'from_1': {
            'to_1': {
                'mol_1': ({}, {}, []),
                'mol_2': ({}, {}, []),
            },
        },
    }


@pytest.mark.parametrize('partial_mappings, expected', (
    (  # new force field from and to
        # partial
        {'from_2': {'to_2': {'mol_1': ({}, {}, [])}}},
        # expected
        {
            'from_1': {'to_1': {'mol_1': ({}, {}, []), 'mol_2': ({}, {}, [])}},
            'from_2': {'to_2': {'mol_1': ({}, {}, [])}}
        }
    ),
    (  # Add a target to an existing force field
        # partial
        {'from_1': {'to_2': {'mol_1': ({}, {}, [])}}},
        # expected
        {
            'from_1': {
                'to_1': {'mol_1': ({}, {}, []), 'mol_2': ({}, {}, [])},
                'to_2': {'mol_1': ({}, {}, [])}
            }
        }
    ),
    (  # Add a molecule to an existing force field pair
        # partial
        {'from_1': {'to_1': {'mol_3': ({}, {}, [])}}},
        # expected
        {'from_1': {'to_1': {
            'mol_1': ({}, {}, []),
            'mol_2': ({}, {}, []),
            'mol_3': ({}, {}, []),
        }}}
    ),
    (  # Replace a molecule from an existing force field pair
        # partial
        {'from_1': {'to_1': {'mol_1': ({}, {}, ['modified'])}}},
        # expected
        {'from_1': {'to_1': {
            'mol_1': ({}, {}, ['modified']),
            'mol_2': ({}, {}, []),
        }}}
    ),
))
def test_combine_mappings(base_mappings, partial_mappings, expected):
    """
    Test that :func:`vermouth.map_input.combine_mappings` works as expected.
    """
    vermouth.map_input.combine_mappings(base_mappings, partial_mappings)
    assert base_mappings == expected
