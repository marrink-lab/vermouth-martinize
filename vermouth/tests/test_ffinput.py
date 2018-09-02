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

import collections
import pytest
from vermouth import ffinput
from vermouth.molecule import Choice, Link
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
    """
    Test _split_node_key works as expected.
    """
    prefix, base = ffinput._split_node_key(key)
    assert prefix == ref_prefix
    assert base == ref_base


@pytest.mark.parametrize('key', (
    '',  # empty key
    '++',  # no base
    '++-BB',  # mixed prefix
))
def test_split_node_key_error(key):
    """
    Test _split_node_key raises an error when expected.
    """
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
    """
    Test _get_order_and_prefix_from_attributes works as expected.
    """
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
    """
    Test _get_order_and_prefix_from_attributes fails when expected.
    """
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
    """
    Test _get_order_and_prefix_from_prefix works as expected.
    """
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
    # CHOICE is `Choice(['A', 'B'])`, it is an instance of `molecule.LinkPredicate`.
    ('XX', {'atomname': CHOICE}, 'XX', {'atomname': CHOICE, 'order': 0}),
))
def test_treat_atom_prefix(key, attributes,
                           ref_prefixed, ref_attributes):
    """
    Test _treat_atom_prefix works as expected.
    """
    (result_prefixed,
     result_attributes) = ffinput._treat_atom_prefix(key, attributes)
    assert result_prefixed == ref_prefixed
    assert result_attributes == ref_attributes
    assert result_attributes is not attributes  # We do a shallow copy


def test_treat_atom_prefix_error():
    """
    Test _treat_atom_prefix fails when order is inconsistent.
    """
    with pytest.raises(IOError):
        ffinput._treat_atom_prefix('+A', {'order': -1})


@pytest.mark.parametrize('line, tokens', (
    ('2  3  1  0.2  1000', ['2', '3', '1', '0.2', '1000']),
    ('PO4  GL1  1 0.2  1000', ['PO4', 'GL1', '1', '0.2', '1000']),
    ('PO4  GL1  --  1 0.2  1000', ['PO4', 'GL1', '--', '1', '0.2', '1000']),
    (
        "BB {'resname': 'ALA', 'secstruc': 'H'} BB "
        "{'resname': 'LYS', 'secstruc': 'H', 'order': +1} 1 0.2 1000",
        ['BB', "{'resname': 'ALA', 'secstruc': 'H'}", 'BB',
         "{'resname': 'LYS', 'secstruc': 'H', 'order': +1}",
         '1', '0.2', '1000']
    ),
    (
        "BB {'resname': 'ALA', 'secstruc': 'H'} "
        "+BB {'resname': 'LYS', 'secstruc': 'H'} -- 1 0.2 1000",
        ['BB', "{'resname': 'ALA', 'secstruc': 'H'}",
         '+BB', "{'resname': 'LYS', 'secstruc': 'H'}",
         '--', '1', '0.2', '1000']
    ),
    ('ATOM1{attributes}ATOM2', ['ATOM1', '{attributes}', 'ATOM2']),
    ('ATOM1 {attributes} ATOM2', ['ATOM1', '{attributes}', 'ATOM2']),
    ('{} {{}} {{}{{}}}', ['{}', '{{}}', '{{}{{}}}']),
    ('{}{{}}{{}{{}}}', ['{}', '{{}}', '{{}{{}}}']),
    ('', []),
    ('    ', []),
))
def test_tokenize(line, tokens):
    """
    Test that _tokenize works as expected.
    """
    found = ffinput._tokenize(line)
    assert found == tokens


@pytest.mark.parametrize('token', (
    '{{{}}',  # missing closing bracket
    '{{}}}',  # missing openning bracket
))
def test_tokenize_bracket_error(token):
    """
    Test that _tokenize recognizes missing brackets.
    """
    with pytest.raises(IOError):
        # One closing bracket is missing.
        ffinput._tokenize(token)


@pytest.mark.parametrize('line, macros, expected', (
    (  # base case
        'content $macro other content',
        {'macro': 'plop'},
        'content plop other content',
    ),
    (  # non-separated macros
        'content $macro$other end',
        {'macro': 'plop', 'other': 'toto'},
        'content ploptoto end',
    ),
    ('hop $macro{toto}', {'macro': 'plop'}, 'hop plop{toto}'),
    ('hop $macro\n', {'macro': 'plop'}, 'hop plop\n'),
    ('hop $macro\t', {'macro': 'plop'}, 'hop plop\t'),
    ('hop $macro', {'macro': 'plop'}, 'hop plop'),
    (  # nothing to replace
        'hop hop hop',
        {'macro': 'plop'},
        'hop hop hop',
    ),
    (  # empty string
        '',
        {'macro': 'plop'},
        '',
    ),
    (  # multiple times the same
        'content $macro $macro content',
        {'macro': 'plop'},
        'content plop plop content',
    ),
    ('$macro', {'macro': 'plop'}, 'plop'),  # only the macro
    ('start $macro', {'macro': 'plop'}, 'start plop'),  # end with macro
))
def test_substitute_macros(line, macros, expected):
    """
    Test _substitute_macros works as expected.
    """
    found = ffinput._substitute_macros(line, macros)
    assert found == expected


def test_substitute_macros_missing():
    """
    Test _substitute_macros fails when a macro is missing.
    """
    with pytest.raises(KeyError):
        ffinput._substitute_macros('hop $missing end', {'macro': 'plop'})


@pytest.mark.parametrize('tokens, atoms, natoms, expected', (
    (['A', 'B', 'C'], [], 2, True),
    (['A', 'B', 'C'], [], 0, False),
    (['A', 'B', 'C'], ['X', 'Y'], 3, True),
    (['A', 'B', 'C'], ['X', 'Y'], 2, False),
    (['A', 'B', '--', 'C'], [], None, True),
    (['--', 'A', 'B', 'C'], [], None, False),
    ([], [], None, False),
))
def test_some_atoms_left(tokens, atoms, natoms, expected):
    """
    Test _some_atoms_left works as expected.
    """
    tokens = collections.deque(tokens)
    found = ffinput._some_atoms_left(tokens, atoms, natoms)
    assert found == expected


@pytest.mark.parametrize('token, expected', (
    ('{}', {}),
    ('{"atomname": "PO4"}', {'atomname': 'PO4'}),
    ('{"a": "abc", "b": "def"}', {'a': 'abc', 'b': 'def'}),
    ('{"a": "123"}', {'a': '123'}),
    ('{"a": 123}', {'a': 123}),
    ('{"a": null}', {'a': None}),
    ('{"a": true}', {'a': True}),
    ('{"a": false}', {'a': False}),
    ('{"a": "A|B|C"}', {'a': Choice(['A', 'B', 'C'])}),
    ('{"a": {"b": "123"}}', {'a': {'b': '123'}}),
))
def test_parse_atom_attributes(token, expected):
    """
    Test _parse_atom_attributes works as expected.
    """
    found = ffinput._parse_atom_attributes(token)
    assert found == expected


@pytest.mark.parametrize('token', (
    '[1, 2]', '1', 'true', 'false', 'null',  # not dict-like
    "{'a': 123}", '{"a", "b"}',
))
def test_parse_atom_attributes_error(token):
    """
    Test _parse_atom_attributes fails as expected.
    """
    with pytest.raises(ValueError):
        ffinput._parse_atom_attributes(token)


@pytest.mark.parametrize('tokens, natoms, expected', (
    (['A', 'B', 'C'], None, [['A', {}], ['B', {}], ['C', {}]]),
    (['A', 'B'], 2, [['A', {}], ['B', {}]]),
    (['A', '{"a": "abc"}', 'B'], 2, [['A', {'a': 'abc'}], ['B', {}]]),
    (['1', '2', '1', '0.31', '7500'], 2, [['1', {}], ['2', {}]]),
    (
        ['BB', '{"replace": {"atype": "Qd", "charge": 1}}'],
        1,
        [['BB', {'replace': {'atype': 'Qd', 'charge': 1}}]],
    ),
    (
        ['BB', '++BB', '1', '0.640', '2500', '{"g": "Short", "edge": false}'],
        2,
        [['BB', {}], ['++BB', {}]],
    ),
    (
        ['-BB', 'BB', '+BB', '{"c": "C"}'],
        None,
        [['-BB', {}], ['BB', {}], ['+BB', {'c': 'C'}]],
    ),
    (
        ['-BB', '{"a": 1}', 'BB', '{"b": 2}', '+BB', '{"c": 3}'],
        None,
        [['-BB', {'a': 1}], ['BB', {'b': 2}], ['+BB', {'c': 3}]],
    ),
    (
        ['4', '6', '5', '--', '2'],
        None,
        [['4', {}], ['6', {}], ['5', {}]],
    ),
))
def test_get_atoms(tokens, natoms, expected):
    """
    Test _get_atoms works as expected.
    """
    tokens = collections.deque(tokens)
    remaining = collections.deque(tokens)
    found = ffinput._get_atoms(tokens, natoms)
    assert found == expected
    # Make sure '--' is consumed if needed
    if tokens:
        assert not tokens[0] == '--'


@pytest.mark.parametrize('tokens, natoms', (
    (['{"a": "abc"}', 'A'], None),  # atom attributes without an atom
))
def test_get_atoms_errors(tokens, natoms):
    """
    Test _get_atoms fails when expected.
    """
    tokens = collections.deque(tokens)
    with pytest.raises(IOError):
        ffinput._get_atoms(tokens, natoms)


@pytest.mark.parametrize('atoms, apply_to_all, existing, expected', (
    ((), {}, {}, {}),
    (
        (('A', {}), ),
        {},
        {},
        {'A': {'atomname': 'A', 'order': 0}}
    ),
    (
        (('A', {}), ('B', {}), ),
        {},
        {},
        {'A': {'atomname': 'A', 'order': 0}, 'B': {'atomname': 'B', 'order':0}},
    ),
    (
        (('+A', {}), ),
        {},
        {},
        {'+A': {'atomname': 'A', 'order': 1}},
    ),
    (
        (('A', {}),  ('B', {}), ),
        {'attr': 0},
        {},
        {
            'A': {'atomname': 'A', 'order': 0, 'attr': 0},
            'B': {'atomname': 'B', 'order': 0, 'attr': 0},
        }
    ),
    (
        (('A', {'exist': 'hello'}), ('B', {}), ),
        {},
        {'A': {'new': 'world'}},
        {
            'A': {'atomname': 'A', 'order': 0, 'exist': 'hello', 'new': 'world'},
            'B': {'atomname': 'B', 'order': 0},
        },
    ),
    (
        (('A', {'exist': 'hello'}), ('B', {}), ),
        {'attr': 'plop'},
        {'A': {'new': 'world'}},
        {
            'A': {
                'atomname': 'A', 'order': 0,
                'exist': 'hello', 'new': 'world',
                'attr': 'plop',
            },
            'B': {'atomname': 'B', 'order': 0, 'attr': 'plop'},
        },
    ),
))
def test_treat_link_interaction_atoms(atoms, apply_to_all, existing, expected):
    """
    Test that _treat_link_interaction_atoms works as expected.
    """
    context = Link()
    if apply_to_all:
        context._apply_to_all_nodes = apply_to_all
    context.add_nodes_from(existing.items())
    ffinput._treat_link_interaction_atoms(atoms, context, 'section')
    found = dict(context.nodes.items())
    assert found == expected


def test_treat_link_interaction_atoms_conflix():
    """
    Test that _treat_link_interaction_atoms fails when there is a conflict
    between the atoms to add ans the existing atoms.
    """
    context = Link()
    context.add_nodes_from({
        'A': {'exist': 'before'},
    }.items())
    atoms = (('A', {'exist': 'after'}), )
    with pytest.raises(IOError):
        ffinput._treat_link_interaction_atoms(atoms, context, 'section')


@pytest.mark.parametrize('token, expected', (
    ('', False),
    ('(something)', False),
    ('before(inside)after', False),
    ('something()', True),
    ('something(inside)', True),
    ('partial(end', False),
))
def test_is_param_effector(token, expected):
    found = ffinput._is_param_effector(token)
    assert found == expected
