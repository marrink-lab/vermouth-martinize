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

import pytest
import collections
import itertools
import textwrap
from pathlib import Path
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
[molecule]
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
[molecule]
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
[molecule]
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
    string="""\
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

""" + SYSTEM_BASIC.string,
    name=SYSTEM_BASIC.name,
    from_ff=['from_ff', 'from_ff_2', 'from_ff_3',
             'from_ff_4', 'from_ff_5', 'from_ff_6', 'from_ff_7'],
    to_ff=['to_ff', 'to_ff_2', 'to_ff_3'],
    mapping=SYSTEM_BASIC.mapping,
    weights=SYSTEM_BASIC.weights,
    extra=SYSTEM_BASIC.extra,
)


@pytest.mark.parametrize(
    'case',
    [SYSTEM_BASIC, SYSTEM_SHARED, SYSTEM_EXTRA, SYSTEM_NULL, SYSTEM_FROM_TO]
)
def test_read_mapping(case):
    name, from_ff, to_ff, mapping, weights, extra = vermouth.map_input.read_mapping(case.string.split('\n'))
    assert name == case.name
    assert from_ff == case.from_ff
    assert to_ff == case.to_ff
    assert mapping == case.mapping
    assert extra == case.extra


@pytest.mark.parametrize('content', (
    """
[ molecule ]
dummy

[ atoms ]
0 X1 !A A B
    """,  # Inconsistent bead weight
    """
[ molecule
dummy
    """,  # Incomplete section line
    """
[ molecule ]
dummy

[ atoms ]
0 X1 A B
1 X2 C D
2 X1 Y U
    """,  # Multiple difinitions for the same atom
    """
no initial context
    """,
    """
[ molecule ]
[ atoms ]
0 A B
    """,  # no molecule name
))

def test_read_mapping_errors(content):
    with pytest.raises(IOError):
        vermouth.map_input.read_mapping(content.split('\n'))


@pytest.fixture(scope='session')
def ref_mapping_directory(tmpdir_factory):
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
    force_fields_from += ['only_from']
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
            (0, 'A{}'.format(idx)): {(0, 'X1{}'.format(idx)): 1},
            (0, 'B{}'.format(idx)): {(0, 'X1{}'.format(idx)): 1},
            (0, 'C{}'.format(idx)): {(0, 'X2{}'.format(idx)): 1},
            (0, 'D{}'.format(idx)): {(0, 'X2{}'.format(idx)): 1},
        }
        extra = []
        mappings[from_ff][to_ff]['dummy_{}'.format(idx)] = (mapping, weights, extra)

    mappings = {from_ff: dict(to_ff) for from_ff, to_ff in mappings.items()}

    return Path(str(basedir)), mappings


def test_read_mapping_directory(ref_mapping_directory):
    dirpath, ref_mappings = ref_mapping_directory
    mappings = vermouth.map_input.read_mapping_directory(dirpath)
    assert mappings == ref_mappings


def test_read_mapping_directory_not_dir():
    with pytest.raises(NotADirectoryError):
        vermouth.map_input.read_mapping_directory('not a directory')


def test_read_mapping_directory_error(tmpdir):
    mapdir = tmpdir.mkdir('mappings')
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
    assert ref_mappings == mappings


def test_generate_all_self_mappings():
    force_fields = []
    expected = []
    for idx in range(3):
        idx_str = str(idx)
        ff_name = 'ff_' + idx_str
        ff = vermouth.forcefield.ForceField(name=ff_name)
        ff.blocks = {
            'A' + idx_str: vermouth.molecule.Block([['AA', 'BBB'], ['BBB', 'CCCC']]),
            'B' + idx_str: vermouth.molecule.Block([['BBB', 'CCCC'], ['BBB', 'E']]),
        }
        for name, block in ff.blocks.items():
            block.name = name
            for atomname, node in block.nodes.items():
                node['atomname'] = atomname
        force_fields.append(ff)

        expected.append((ff_name, (ff_name, )))

    mappings = vermouth.map_input.generate_all_self_mappings(force_fields)
    found = [(from_ff,  tuple(to_ff.keys())) for from_ff, to_ff in mappings.items()]

    assert found == expected


@pytest.fixture
def base_mappings():
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
    vermouth.map_input.combine_mappings(base_mappings, partial_mappings)
    assert base_mappings == expected
