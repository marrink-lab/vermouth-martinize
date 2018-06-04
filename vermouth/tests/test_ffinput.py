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
from vermouth import ffinput
from vermouth.molecule import Choice
import numpy as np


CHOICE = Choice(['A', 'B'])


@pytest.mark.parametrize('key, ref_prefix, ref_base', (
    ('BB', '', 'BB'),
    ('+XX', '+', 'XX'),
    ('--UNK', '--', 'UNK'),
    ('>>>PLOP', '>>>', 'PLOP'),
    ('<<KEY', '<<', 'KEY'),
    ('++ONE+TWO', '++', 'ONE+TWO'),
    ('*****NODE', '*****', 'NODE'),
))
def test_split_node_key(key, ref_prefix, ref_base):
    prefix, base = ffinput._split_node_key(key)
    assert prefix == ref_prefix
    assert base == ref_base


@pytest.mark.parametrize('key', (
    '',  # empty key
    '++',  # no base
    '++-BB',  # mixed prefix
))
def test_split_node_key_error(key):
    with pytest.raises(IOError):
        prefix, base = ffinput._split_node_key(key)
        print('prefix: "{}"; base: "{}"'.format(prefix, base))


@pytest.mark.parametrize('attributes, ref_prefix, ref_order', (
    ({}, '', None),
    ({'order': None}, '', None),
    ({'arbitrary': 'something'}, '', None),
    ({'atomname': 'BB'}, '', None),
    ({'order': 4}, '++++', 4),
    ({'order': -2}, '--', -2),
    ({'order': 0}, '', 0),
    ({'order': np.int16(4)}, '++++', 4),
    ({'order': '>'}, '>', '>'),
    ({'order': '>>'}, '>>', '>>'),
    ({'order': '<'}, '<', '<'),
    ({'order': '<<<'}, '<<<', '<<<'),
    ({'order': '*'}, '*', '*'),
    ({'order': '****'}, '****', '****'),
))
def test_get_order_and_prefix_from_attributes(attributes, ref_prefix, ref_order):
    (result_prefix,
     result_order) = ffinput._get_order_and_prefix_from_attributes(attributes)
    assert result_prefix == ref_prefix
    assert result_order == ref_order


@pytest.mark.parametrize('attributes', (
    {'order': ''},
    {'order': 4.5},
    {'order': True},
    {'order': False},
    {'order': np.True_},
    {'order': '++'},
    {'order': '-'},
    {'order': '><'},
    {'order': 'invalid'},
))
def test_get_order_and_prefix_from_attributes_error(attributes):
    with pytest.raises(IOError):
        result = ffinput._get_order_and_prefix_from_attributes(attributes)
        print(result)


@pytest.mark.parametrize('prefix, ref_prefix, ref_order', (
    ('', None, 0),
    ('+', '+', 1),
    ('++', '++', 2),
    ('-', '-', -1),
    ('---', '---', -3),
    ('>', '>', '>'),
    ('>>>>', '>>>>', '>>>>'),
    ('<', '<', '<'),
    ('<<<', '<<<', '<<<'),
    ('*', '*', '*'),
    ('**', '**', '**'),
))
def test_get_order_and_prefix_from_prefix(prefix, ref_prefix, ref_order):
    result_prefix, result_order = ffinput._get_order_and_prefix_from_prefix(prefix)
    assert result_prefix == ref_prefix
    assert result_order == ref_order


@pytest.mark.parametrize('key, attributes, ref_prefixed, ref_attributes', (
    # No attribute given
    ('BB', {}, 'BB', {'atomname': 'BB', 'order': 0}),
    ('+BB', {}, '+BB', {'atomname': 'BB', 'order': 1}),
    ('--BB', {}, '--BB', {'atomname': 'BB', 'order': -2}),
    ('>>XX', {}, '>>XX', {'atomname': 'XX', 'order': '>>'}),
    ('<<<YYY', {}, '<<<YYY', {'atomname': 'YYY', 'order': '<<<'}),
    ('****Z', {}, '****Z', {'atomname': 'Z', 'order': '****'}),
    # Order given by the attribute
    ('BB', {'order': 3}, '+++BB', {'atomname': 'BB', 'order': 3}),
    ('XX', {'order': -4}, '----XX', {'atomname': 'XX', 'order': -4}),
    ('YYY', {'order': '>'}, '>YYY', {'atomname': 'YYY', 'order': '>'}),
    ('ZZZZ', {'order': '<<<'}, '<<<ZZZZ', {'atomname': 'ZZZZ', 'order': '<<<'}),
    ('A', {'order': '**'}, '**A', {'atomname': 'A', 'order': '**'}),
    # Order given both with prefix and attribute
    ('BB', {'order': 0}, 'BB', {'atomname': 'BB', 'order': 0}),
    ('++BB', {'order': 2}, '++BB', {'atomname': 'BB', 'order': 2}),
    ('--BB', {'order': -2}, '--BB', {'atomname': 'BB', 'order': -2}),
    ('>>BB', {'order': '>>'}, '>>BB', {'atomname': 'BB', 'order': '>>'}),
    ('<<BB', {'order': '<<'}, '<<BB', {'atomname': 'BB', 'order': '<<'}),
    ('**BB', {'order': '**'}, '**BB', {'atomname': 'BB', 'order': '**'}),
    # atomname given in attribute
    ('BB', {'atomname': 'BB'}, 'BB', {'atomname': 'BB', 'order': 0}),
    ('>>XX', {'atomname': 'BB'}, '>>XX', {'atomname': 'BB', 'order': '>>'}),
    ('YYY', {'atomname': 'BB', 'order': '**'}, '**YYY', {'atomname': 'BB', 'order': '**'}),
    # CHOICE is `Choice(['A', 'B'])`, it is an instance of
    # `molecule.LinkPredicate`. It is defined beforehand so the two references
    # are made to the same instance. Te equality of link predicates is based on
    # their identity since they do not have an `__eq__` method.
    ('XX', {'atomname': CHOICE}, 'XX', {'atomname': CHOICE, 'order': 0}),
))
def test_treat_atom_prefix(key, attributes,
                           ref_prefixed, ref_attributes):
    (result_prefixed,
     result_attributes) = ffinput._treat_atom_prefix(key, attributes)
    assert result_prefixed == ref_prefixed
    assert result_attributes == ref_attributes
    assert result_attributes is not attributes  # We do a shallow copy
