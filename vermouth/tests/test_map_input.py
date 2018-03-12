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
import vermouth.map_input

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
