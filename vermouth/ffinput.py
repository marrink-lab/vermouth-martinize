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
Read .ff files.

The FF file format describes molecule components for a given force field. It is
a test format devised for quick prototyping.

The format is built on top of a subset of the ITP format. Describing a block
is done in the same way an ITP file describes a molecule.
"""

import collections
import copy
import numbers
import json
from .molecule import (
    Block, Link, Modification,
    Interaction, DeleteInteraction,
    Choice, NotDefinedOrNot,
    ParamDistance, ParamAngle, ParamDihedral, ParamDihedralPhase,
)
from .parser_utils import (
    SectionLineParser, _tokenize, _substitute_macros, _parse_macro
)

from .log_helpers import StyleAdapter, get_logger
import networkx as nx

# Python 3.4 does not raise JSONDecodeError but ValueError.

try:
    from json import JSONDecodeError
except ImportError:
    JSONDecodeError = ValueError

LOGGER = StyleAdapter(get_logger(__name__))
VALUE_PREDICATES = {
    'not': NotDefinedOrNot,
}

PARAMETER_EFFECTORS = {
    'dist': ParamDistance,
    'angle': ParamAngle,
    'dihedral': ParamDihedral,
    'dihphase': ParamDihedralPhase,
}


class FFDirector(SectionLineParser):
    COMMENT_CHAR = ';'
    interactions_natoms = {
        'bonds': 2,
        'angles': 3,
        'dihedrals': 4,
        'impropers': 4,
        'constraints': 2,
        'pairs': 2,
        'pairs_nb': 2,
        'SETTLE': 1,
        'virtual_sites2': 3,
        'virtual_sites3': 4,
        'virtual_sites4': 5,
        'position_restraints': 1,
        'distance_restraints': 2,
        'dihedral_restraints': 4,
        'orientation_restraints': 2,
        'angle_restraints': 4,
        'angle_restraints_z': 2
    }

    def __init__(self, force_field):
        super().__init__()
        self.force_field = force_field
        self.current_block = None
        self.current_link = None
        self.current_modification = None
        self.blocks = collections.OrderedDict()
        self.links = []
        self.modifications = []
        self.citations = set()
        self.header_actions = {
            ('moleculetype', ): self._new_block,
            ('link', ): self._new_link,
            ('modification', ): self._new_modification,
        }

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets
        :attr:`vermouth.parser_utils.SectionLineParser.section` when applicable.
        Does not check whether `line` is a valid section header.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`finalize_section`, which is called
            if a section ends.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
        prev_section = None

        ended = []
        section = self.section + [line.strip('[ ]').casefold()]

        if tuple(section[-1:]) in self.METH_DICT:
            prev_section = self.section
            self.section = section[-1:]

        else:
            while (tuple(section) not in self.METH_DICT
                   and len(section) > 1):
                ended.append(section.pop(-2))  # [a, b, c, d] -> [a, b, d]
            self.section = section

        result = None

        if prev_section:
            result = self.finalize_section(prev_section, ended)

        action = self.header_actions.get(tuple(self.section))
        if action:
            action()

        return result


    def finalize_section(self, previous_section, ended_section):
        """
        Called once a section is finished. It appends the current_links list
        to the links and update the block dictionary with current_block. Thereby it
        finishes the reading a given section.

        Parameters
        ---------
        previous_section: list[str]
            The last parsed section.
        ended_section: list[str]
            The sections that have been ended.
        """

        if self.current_block is not None:
            # add FF wide citations
            self.current_block.citations.update(self.citations)
            self.current_block.make_edges_from_interactions()
            self.force_field.blocks[self.current_block.name] = self.current_block

        if self.current_link is not None:
            # add FF wide citations
            self.current_link.citations.update(self.citations)
            self.current_link.make_edges_from_interactions()
            self.force_field.links.append(self.current_link)

        if self.current_modification is not None:
            # add FF wide citations
            if self.current_modification and not nx.is_connected(self.current_modification):
                LOGGER.error('Modification {} in force field {} is not a single connected component',
                             self.current_modification.name, self.force_field.name)
            self.current_modification.citations.update(self.citations)
            self.force_field.modifications[self.current_modification.name] = self.current_modification

    def get_context(self, context_type):
        possible_contexts = {
            'block': self.current_block,
            'link': self.current_link,
            'molmeta': self.current_link,
            'modification': self.current_modification,
        }
        return possible_contexts[context_type]

    def has_context(self):
        open_contexts = [
            self.current_block, self.current_link, self.current_modification]
        return open_contexts != ([None] * len(open_contexts))

    def _new_block(self):
        self.current_block = Block(force_field=self.force_field)

    def _new_link(self):
        self.current_link = Link(force_field=self.force_field)

    def _new_modification(self):
        self.current_modification = Modification(force_field=self.force_field)

    @SectionLineParser.section_parser('variables')
    def _variables(self, line, lineno=0):
        if self.has_context():
            raise IOError('The [variables] section must be defined '
                          'before the blocks, links, and modifications.')
        tokens = _tokenize(line)
        _parse_variables(tokens, self.force_field, 'variables')

    @SectionLineParser.section_parser('moleculetype')
    def _block(self, line, lineno=0):
        name, nrexcl = line.split()
        self.current_block.name = name
        self.current_block.nrexcl = int(nrexcl)

    @SectionLineParser.section_parser('moleculetype', 'atoms')
    def _block_atoms(self, line, lineno=0):
        tokens = collections.deque(_tokenize(line))
        _parse_block_atom(tokens, self.current_block)

    @SectionLineParser.section_parser('moleculetype', 'edges',
                                      negate=False, context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'non-edges',
                                      negate=True, context_type='block')
    @SectionLineParser.section_parser('link', 'edges',
                                      negate=False, context_type='link')
    @SectionLineParser.section_parser('link', 'non-edges',
                                      negate=True, context_type='link')
    @SectionLineParser.section_parser('modification', 'edges',
                                      negate=False, context_type='modification')
    def _edges(self, line, lineno=0, negate=False, context_type=''):
        context = self.get_context(context_type)
        tokens = collections.deque(_tokenize(line))
        _parse_edges(tokens, context, context_type, negate=negate)

    @SectionLineParser.section_parser('moleculetype', 'bonds', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'angles', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'impropers', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'constraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'pairs', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'exclusions', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'pairs_nb', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'SETTLE', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites2', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites3', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sites4', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'virtual_sitesn', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'position_restraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'distance_restraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'dihedral_restraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'orientation_restraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints', context_type='block')
    @SectionLineParser.section_parser('moleculetype', 'angle_restraints_z', context_type='block')
    @SectionLineParser.section_parser('link', 'bonds', context_type='link')
    @SectionLineParser.section_parser('link', 'angles', context_type='link')
    @SectionLineParser.section_parser('link', 'impropers', context_type='link')

    @SectionLineParser.section_parser('link', 'constraints', context_type='link')
    @SectionLineParser.section_parser('link', 'pairs', context_type='link')
    @SectionLineParser.section_parser('link', 'exclusions', context_type='link')
    @SectionLineParser.section_parser('link', 'pairs_nb', context_type='block')
    @SectionLineParser.section_parser('link', 'SETTLE', context_type='link')
    @SectionLineParser.section_parser('link', 'virtual_sites2', context_type='link')
    @SectionLineParser.section_parser('link', 'virtual_sites3', context_type='link')
    @SectionLineParser.section_parser('link', 'virtual_sites4', context_type='link')
    @SectionLineParser.section_parser('link', 'virtual_sitesn', context_type='link')
    @SectionLineParser.section_parser('link', 'position_restraints', context_type='link')
    @SectionLineParser.section_parser('link', 'distance_restraints', context_type='link')
    @SectionLineParser.section_parser('link', 'dihedral_restraints', context_type='link')
    @SectionLineParser.section_parser('link', 'orientation_restraints', context_type='link')
    @SectionLineParser.section_parser('link', 'angle_restraints', context_type='link')
    @SectionLineParser.section_parser('link', 'angle_restraints_z', context_type='link')
    @SectionLineParser.section_parser('link', '!bonds', context_type='link')
    @SectionLineParser.section_parser('link', '!angles', context_type='link')
    @SectionLineParser.section_parser('link', '!impropers', context_type='link')
    @SectionLineParser.section_parser('link', '!constraints', context_type='link')
    @SectionLineParser.section_parser('link', '!pairs', context_type='link')
    @SectionLineParser.section_parser('link', '!exclusions', context_type='link')
    @SectionLineParser.section_parser('link', '!pairs_nb', context_type='link')
    @SectionLineParser.section_parser('link', '!SETTLE', context_type='link')
    @SectionLineParser.section_parser('link', '!virtual_sites2', context_type='link')
    @SectionLineParser.section_parser('link', '!virtual_sites3', context_type='link')
    @SectionLineParser.section_parser('link', '!virtual_sites4', context_type='link')
    @SectionLineParser.section_parser('link', '!virtual_sitesn', context_type='link')
    @SectionLineParser.section_parser('link', '!position_restraints', context_type='link')
    @SectionLineParser.section_parser('link', '!distance_restraints', context_type='link')
    @SectionLineParser.section_parser('link', '!dihedral_restraints', context_type='link')
    @SectionLineParser.section_parser('link', '!orientation_restraints', context_type='link')
    @SectionLineParser.section_parser('link', '!angle_restraints', context_type='link')
    @SectionLineParser.section_parser('link', '!angle_restraints_z', context_type='link')
    @SectionLineParser.section_parser('modification', 'bonds', context_type='modification')
    @SectionLineParser.section_parser('modification', 'angles', context_type='modification')
    @SectionLineParser.section_parser('modification', 'impropers', context_type='modification')
    @SectionLineParser.section_parser('modification', 'constraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'pairs', context_type='modification')
    @SectionLineParser.section_parser('modification', 'exclusions', context_type='modification')
    @SectionLineParser.section_parser('modification', 'pairs_nb', context_type='modification')
    @SectionLineParser.section_parser('modification', 'SETTLE', context_type='modification')
    @SectionLineParser.section_parser('modification', 'virtual_sites2', context_type='modification')
    @SectionLineParser.section_parser('modification', 'virtual_sites3', context_type='modification')
    @SectionLineParser.section_parser('modification', 'virtual_sites4', context_type='modification')
    @SectionLineParser.section_parser('modification', 'virtual_sitesn', context_type='modification')
    @SectionLineParser.section_parser('modification', 'position_restraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'distance_restraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'dihedral_restraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'orientation_restraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'angle_restraints', context_type='modification')
    @SectionLineParser.section_parser('modification', 'angle_restraints_z', context_type='modification')
    def _interactions(self, line, lineno=0, context_type=''):
        context = self.get_context(context_type)
        interaction_name = self.section[-1]
        delete = False
        if interaction_name.startswith('!'):
            interaction_name = interaction_name[1:]
            delete = True
        tokens = collections.deque(_tokenize(line))
        if tokens[0] == '#meta':
            _parse_meta(
                tokens,
                context,
                context_type=context_type,
                section=interaction_name,
            )
        else:
            n_atoms = self.interactions_natoms.get(interaction_name)
            _base_parser(
                tokens,
                context,
                context_type=context_type,
                section=interaction_name,
                natoms=n_atoms,
                delete=delete,
            )

    @SectionLineParser.section_parser('link', 'dihedrals', context_type='link')
    @SectionLineParser.section_parser('link', '!dihedrals', context_type='link')
    @SectionLineParser.section_parser('moleculetype', 'dihedrals', context_type='block')
    @SectionLineParser.section_parser('modification', 'dihedrals', context_type='modification')
    def _dih_interactions(self, line, lineno=0, context_type=''):
        context = self.get_context(context_type)
        interaction_name = self.section[-1]
        delete = False
        tokens = collections.deque(_tokenize(line))
        if tokens[0] == '#meta':
            _parse_meta(
                tokens,
                context,
                context_type=context_type,
                section=interaction_name,
            )
        else:
            n_atoms = self.interactions_natoms.get(interaction_name)
            _base_parser(
                tokens,
                context,
                context_type=context_type,
                section=interaction_name,
                natoms=n_atoms,
                delete=delete,
            )

          # Because of how they are described in gromacs, proper and improper
          # dihedral angles are all under the [ dihedrals ] section. However
          # the way they are treated differently in the library, at least on how
          # they generate edges. Here we move the all the impropers into their own
          # [ impropers ] section.

        propers = []
        impropers = context.interactions.get('impropers', [])
        for dihedral in context.interactions.get('dihedrals', []):
            if dihedral.parameters and dihedral.parameters[0] == '2':
                impropers.append(dihedral)
            else:
                propers.append(dihedral)

        context.interactions['dihedrals'] = propers
        context.interactions['impropers'] = impropers


    @SectionLineParser.section_parser('moleculetype', 'patterns')
    @SectionLineParser.section_parser('moleculetype', 'features')
    @SectionLineParser.section_parser('moleculetype', 'non-edge')
    @SectionLineParser.section_parser('modification', 'non-edge')
    def _invalid_out_of_link(self, line, lineno=0):
        raise IOError('The "{}" section is only valid in links.'
                      .format(self.section[-1]))

    @SectionLineParser.section_parser('link', context_type='link')
    @SectionLineParser.section_parser('link', 'molmeta', context_type='molmeta')
    def _link(self, line, lineno=0, context_type=''):
        tokens = collections.deque(_tokenize(line))
        _parse_link_attribute(tokens, self.current_link, context_type)

    @SectionLineParser.section_parser('modification')
    def _modification(self, line, lineno=0):
        self.current_modification.name = line

    @SectionLineParser.section_parser('link', 'atoms')
    def _link_atoms(self, line, lineno=0):
        tokens = collections.deque(_tokenize(line))
        _parse_link_atom(tokens, self.current_link)

    @SectionLineParser.section_parser('modification', 'atoms')
    def _modification_atoms(self, line, lineno=0):
        tokens = collections.deque(_tokenize(line))
        _parse_link_atom(tokens, self.current_modification,
                         defaults={'PTM_atom': False},
                         treat_prefix=True)

    @SectionLineParser.section_parser('link', 'patterns', context_type='link')
    @SectionLineParser.section_parser('modification', 'patterns', context_type='modification')
    def _link_patterns(self, line, lineno=0, context_type=''):
        context = self.get_context(context_type)
        tokens = collections.deque(_tokenize(line))
        _parse_patterns(tokens, context, context_type)

    @SectionLineParser.section_parser('link', 'features', context_type='link')
    @SectionLineParser.section_parser('modification', 'features', context_type='modification')
    def _link_features(self, line, lineno=0, context_type=''):
        context = self.get_context(context_type)
        tokens = collections.deque(_tokenize(line))
        _parse_features(tokens, context, context_type)

    @SectionLineParser.section_parser('moleculetype', 'citation', context_type='block')
    @SectionLineParser.section_parser('link', 'citation', context_type='link')
    @SectionLineParser.section_parser('modification', 'citation', context_type='modification')
    def _parse_citation(self, line, lineno=0, context_type=""):
        cite_keys = line.split()
        self.get_context(context_type).citations.update(cite_keys)

    @SectionLineParser.section_parser('citations')
    def _pase_ff_citations(self, line, lineno=0):
        # parses force-field wide citations
        cite_keys = line.split()
        self.citations.update(cite_keys)

def _some_atoms_left(tokens, atoms, natoms):
    """
    Return True if the token list expected to contain atoms.

    If the number of atoms is known before hand, then the function compares the
    number of already found atoms to the expected number. If the '--' token if
    found, it is removed from the token list and there is no atom left.

    Parameters
    ----------
    tokens: collections.deque[str]
        Deque of token to inspect. The deque **can be modified** in place.
    atoms: list
        List of already found atoms.
    natoms: int or None
        The number of expected atoms if known, else None.

    Returns
    -------
    bool
    """
    if not tokens:
        return False
    if tokens and tokens[0] == '--':
        tokens.popleft()
        return False
    if natoms is not None and len(atoms) >= natoms:
        return False
    return True


def _parse_atom_attributes(token):
    """
    Parse bracketed tokens.

    Parameters
    ----------
    token: str
        Token in the form of a json dictionary.

    Returns
    -------
    dict
    """
    if not token.strip().startswith('{'):
        raise ValueError('The token should start with a curly bracket.')
    try:
        attributes = json.loads(token)
    except JSONDecodeError as error:
        raise ValueError('The following value is not a valid atom attribute token: "{}".'
                         .format(token)) from error
    modifications = {}
    for key, value in attributes.items():
        try:
            if '|' in value:
                modifications[key] = Choice(value.split('|'))
        except TypeError:
            pass
    attributes.update(modifications)
    return attributes


def _get_atoms(tokens, natoms):
    atoms = []
    while tokens and _some_atoms_left(tokens, atoms, natoms):
        token = tokens.popleft()
        if token.startswith('{'):
            msg = 'Found atom attributes without an atom reference.'
            raise IOError(msg)
        if tokens:
            next_token = tokens[0]
        else:
            next_token = ''
        if next_token.startswith('{'):
            atoms.append([token, _parse_atom_attributes(next_token)])
            tokens.popleft()
        else:
            atoms.append([token, {}])
    return atoms


def _treat_block_interaction_atoms(atoms, context, section):
    atom_names = list(context.nodes)
    all_references = []
    for atom in atoms:
        reference = atom[0]
        if reference.isdigit():
            # The indices in the file are 1-based
            reference = int(reference) - 1
            try:
                reference = atom_names[reference]
            except IndexError:
                msg = ('There are {} atoms defined in the block "{}". '
                       'Interaction in section "{}" cannot refer to '
                       'atom index {}.')
                raise IOError(msg.format(len(context), context.name,
                                         section, reference + 1))
            atom[0] = reference
        else:
            if reference not in context:
                msg = ('There is no atom "{}" defined in the block "{}". '
                       'Section "{}" cannot refer to it.')
                raise IOError(msg.format(reference, context.name, section))
            if reference[0] in '+-<>':
                msg = ('Atom names in blocks cannot be prefixed with + or -. '
                       'The name "{}", used in section "{}" of the block "{}" '
                       'is not valid in a block.')
                raise IOError(msg.format(reference, section, context.name))
        all_references.append(reference)
    return all_references


def _split_node_key(key):
    """
    Split a node key into a prefix and a base and validate the key validity.
    """
    if not key:
        raise IOError('A node key cannot be empty.')

    # If the atom name is prefixed, we can get the order.
    prefix_end = 0  # Make sure prefix_end is defined even if key is empty
    for prefix_end, char in enumerate(key):
        if char not in '+-><*':
            break
    else:  # no break
        # This branch could also be taken if 'key' was empty. However, we
        # tested already that it was not the case.
        msg = ('The atom key "{}" is not valid. There must be a name '
               'following prefix.')
        raise IOError(msg.format(key))

    prefix = key[:prefix_end]
    if len(set(prefix)) > 1:
        msg = ('Atom name prefix cannot mix characters. Atom name "{}" '
               'is not a valid key.')
        raise IOError(msg.format(key))

    base = key[prefix_end:]

    return prefix, base


def _get_order_and_prefix_from_attributes(attributes):
    prefix_from_attributes = ''
    order_from_attributes = None
    Sequence = collections.abc.Sequence  # pylint: disable=invalid-name
    if attributes.get('order') is not None:
        order = attributes['order']
        order_from_attributes = order
        if isinstance(order, numbers.Integral) and not isinstance(order, bool):
            # Boolean as abstract subclasses of number.Integral as they can be
            # considered as 0 and 1. Yet, they yield unexpected results and
            # should not be accepted as valid values for 'order'.
            if order > 0:
                prefix_char = '+'
            else:
                prefix_char = '-'
            prefix_from_attributes = prefix_char * int(abs(order))
        elif (isinstance(order, Sequence)  # We can do the following operations
              and len(set(order)) == 1     # It is homogeneous
              and order[0] in '><*'):      # The character is an expected one
            prefix_from_attributes = order
        else:
            raise IOError('The order given in attribute ("{}") is not valid. '
                          'It must be am integer or a homogeneous series '
                          'of ">", "<", or "*".'
                          .format(order))
    return prefix_from_attributes, order_from_attributes


def _get_order_and_prefix_from_prefix(prefix):
    """
    Convert a prefix into a numerical value.
    """
    prefix_from_prefix = None
    order_from_prefix = 0
    if not prefix:
        return prefix_from_prefix, order_from_prefix

    # It is already validated.
    prefix_from_prefix = prefix
    if prefix[0] == '+':
        order_from_prefix = len(prefix)
    elif prefix[0] == '-':
        order_from_prefix = -len(prefix)
    else:
        order_from_prefix = prefix
    return prefix_from_prefix, order_from_prefix


def _treat_atom_prefix(reference, attributes):
    """
    Connect graph keys, order, and atom names.

    In a link, the graph keys, the order attribute of the atoms, and the atom
    names are interconnected. In most cases, the graph key is the atom name
    prefixed in a way that represent the order attribute. It is possible to
    define the order and the atom name from the graph key, or to set the graph
    key to represent the order, depending on what is explicitly specified in
    the file.

    In a link node, the order can be an integer, a series of '>' (*e.g.* '>',
    '>>', '>>>', ...), a series of '<', or a series of '*'. The series of '>',
    '<', and '*' translate directly from the key prefix to the order attribute,
    and *vice versa*. Numerical values of the order attribute, however, are
    converted into series of '+' or '-' for the key prefix; there, the number
    of '+' or '-' in the prefix corresponds to the value of the order
    attribute.

    The order can be specified either by the key prefix, or by the attribute.
    If it is specified in the two places, then they have to match each other or
    a :exc:`IOError` is raised.

    If the atom name is explicitly specified, then it is not modified. If it is
    not specified, then it is set from the node key. The base of the node key
    (*i.e.* what follows the prefix) is not modified, but a prefix can be added
    if there is none. The base of the node key and the atom name *can* differ.
    The atom name is what will be use for the graph isomorphism. The base of
    the key cannot be empty (*i.e.* '+++' or '*' are not valid keys); if it is,
    then an :exc:`IOError` is raised.

    Parameters
    ----------
    reference: str
        A node key for a link, as written in the file.
    attributes: dict
        The node attributes read fro the file.

    Returns
    -------
    prefixed_reference: str
        The node key with the appropriate prefix.
    attributes: dict
        A shalow copy of the node attribute dictionary with the 'order' and the
        'atomname' attributes set as appropriate.

    Raises
    ------
    IOError
        The node key, or an attribute value is invalid.

    Examples
    --------

    >>> _treat_atom_prefix('+BB', {})
    ('+BB', {'order': 1, 'atomname': 'BB'})
    >>> _treat_atom_prefix('BB', {'order': 1})
    ('+BB', {'order': 1, 'atomname': 'BB'})
    >>> _treat_atom_prefix('--XX', {'atomname': 'BB'})
    ('+BB', {'order': -2, 'atomname': 'BB'})
    >>> _treat_atom_prefix('>>BB', {})
    ('>>BB', {'order': '>>', 'atomname': 'BB'})

    """
    prefix, base = _split_node_key(reference)

    # Is the order specified in the attributes?
    (prefix_from_attributes,
     order_from_attributes) = _get_order_and_prefix_from_attributes(attributes)
    # Is there a specified prefix?
    (prefix_from_prefix,
     order_from_prefix) = _get_order_and_prefix_from_prefix(prefix)

    # If the order is defined twice, is it consistent?
    if (order_from_attributes is not None
            and prefix_from_prefix is not None
            and order_from_attributes != order_from_prefix):
        msg = ('The sequence order for atom "{}" of a '
               'link is not consistent between the name prefix '
               '(order={}) and the atom attributes (order={}).')
        raise IOError(msg.format(reference, order_from_prefix, order_from_attributes))

    return_attributes = copy.copy(attributes)
    if order_from_attributes is None:
        return_attributes['order'] = order_from_prefix
    if 'atomname' not in return_attributes:
        return_attributes['atomname'] = base
    if prefix_from_prefix is None:
        prefixed = prefix_from_attributes + base
    else:
        prefixed = reference

    return prefixed, return_attributes


def _treat_link_interaction_atoms(atoms, context, section):
    all_references = []
    for reference, attributes in atoms:
        intermediate = context._apply_to_all_nodes.copy()
        intermediate.update(attributes)
        attributes = intermediate

        prefixed_reference, attributes = _treat_atom_prefix(reference, attributes)
        all_references.append(prefixed_reference)

        if prefixed_reference in context:
            context_atom = context.nodes[prefixed_reference]
            for key, value in attributes.items():
                if key in context_atom and value != context_atom[key]:
                    msg = ('Attribute {} of atom {} conflicts in a link '
                           'between its definition in section "{}" '
                           '(value is "{}") and its previous definition '
                           '(value was "{}").')
                    raise IOError(msg.format(key, reference, section,
                                             value, context_atom[key]))
            context_atom.update(attributes)
        else:
            context.add_node(prefixed_reference, **attributes)
    return all_references


def _parse_interaction_parameters(tokens):
    parameters = []
    for token in tokens:
        if _is_param_effector(token):
            effector_name, effector_param_str = token.split('(', 1)
            effector_param_str = effector_param_str[:-1]  # Remove the closing parenthesis
            try:
                effector_class = PARAMETER_EFFECTORS[effector_name]
            except KeyError:
                raise IOError('{} is not a known parameter effector.'
                              .format(effector_name))
            if '|' in effector_param_str:
                effector_param_str, effector_format = effector_param_str.split('|')
            else:
                effector_format = None
            effector_param = [elem.strip() for elem in effector_param_str.split(',')]
            parameter = effector_class(effector_param, format_spec=effector_format)
        else:
            parameter = token
        parameters.append(parameter)
    return parameters


def _is_param_effector(token):
    return (
        '(' in token
        and not token.startswith('(')
        and token.endswith(')')
    )


def _base_parser(tokens, context, context_type, section, natoms=None, delete=False):
    if context_type != 'link' and delete:
        raise IOError('Interactions can only be removed in links.')
    delimiter_count = tokens.count('--')
    if delimiter_count > 1:
        msg = 'There can be 0 or 1 "--" delimiter; {} found.'
        raise IOError(msg.format(delimiter_count))

    # Group the atoms and their attributes
    atoms = _get_atoms(tokens, natoms)
    if natoms is not None and len(atoms) != natoms:
        raise IOError('Found {} atoms while {} were expected.'
                      .format(len(atoms), natoms))

    # Normalize the atom references.
    # Blocks and links treat these references differently.
    # For blocks:
    # * references can be written as indices or atom names
    # * a reference cannot be prefixed by + or -
    # * an interaction cannot create a new atom
    # For links:
    # * references must be atom names, but they can be prefixed with one or
    #   more + or - to signify the order in the sequence
    # * interactions create nodes
    if context_type == 'block':
        treated_atoms = _treat_block_interaction_atoms(atoms, context, section)
    elif context_type in ('link', 'modification'):
        treated_atoms = _treat_link_interaction_atoms(atoms, context, section)

    # Getting the atoms consumed the "--" delimiter if any. So what is left
    # are the interaction parameters or the meta attributes.
    if tokens and tokens[-1].startswith('{'):
        token = tokens.pop()
        meta = json.loads(token)
    else:
        meta = {}
    parameters = _parse_interaction_parameters(tokens)

    apply_to_all_interactions = context._apply_to_all_interactions[section]
    meta = dict(collections.ChainMap(meta, apply_to_all_interactions))

    if delete:
        interaction = DeleteInteraction(
            atoms=treated_atoms,
            atom_attrs=[atom[1] for atom in atoms],
            parameters=parameters,
            meta=meta,
        )
        interaction_list = context.removed_interactions.get(section, [])
        interaction_list.append(interaction)
        context.removed_interactions[section] = interaction_list
    else:
        interaction = Interaction(
            atoms=treated_atoms,
            parameters=parameters,
            meta=meta,
        )
        interaction_list = context.interactions.get(section, [])
        interaction_list.append(interaction)
        context.interactions[section] = interaction_list


def _parse_block_atom(tokens, context):
    if tokens[-1].startswith('{'):
        attributes = _parse_atom_attributes(tokens.pop())
    else:
        attributes = {}

    # deque does not support slicing
    first_six = (tokens.popleft() for _ in range(6))
    _, atype, resid, resname, name, charge_group = first_six
    if name in context:
        msg = ('There is already an atom named "{}" in the block "{}". '
               'Atom names must be unique within a block.')
        raise IOError(msg.format(name, context.name))
    atom = {
        'atomname': name,
        'atype': atype,
        'resname': resname,
        'resid': int(resid),
        'charge_group': int(charge_group),
    }
    # charge and mass are optional, but charge has to be defined for mass to be
    if tokens:
        atom['charge'] = float(tokens.popleft())
    if tokens:
        atom['mass'] = float(tokens.popleft())
    context.add_atom(dict(collections.ChainMap(attributes, atom)))


def _parse_link_atom(tokens, context, defaults=None, treat_prefix=True):
    if len(tokens) > 2:
        raise IOError('Unexpected column in link atom definition.')
    if len(tokens) < 2:
        raise IOError('Missing column in link atom definition.')
    reference = tokens[0]
    if defaults is None:
        defaults = {}
    attributes = _parse_atom_attributes(tokens[1])
    if treat_prefix:
        prefixed_reference, attributes = _treat_atom_prefix(reference, attributes)
    else:
        prefixed_reference = reference
        attributes['atomname'] = reference
    attributes = dict(collections.ChainMap(attributes, context._apply_to_all_nodes))
    node_attributes = context.nodes.get(prefixed_reference, {})
    for attr, value in attributes.items():
        if value != node_attributes.get(attr, value):
            msg = ('Conflict in an atom attributes in the definition of a '
                   'link node. Cannot set the attribute "{}" of atom "{}" '
                   'to "{}" because it is already defined as "{}".')
            raise IOError(msg.format(attr, reference, value,
                                     node_attributes[node_attributes[attr]]))
    full_attributes = dict(collections.ChainMap(attributes, node_attributes, defaults))

    if prefixed_reference in context.nodes:
        context.nodes[prefixed_reference] = full_attributes
    else:
        context.add_node(prefixed_reference, **full_attributes)


def _parse_link_attribute(tokens, context, section):
    if len(tokens) > 2:
        raise IOError('Unexpected column in section "{}".'.format(section))
    if len(tokens) < 2:
        raise IOError('Missing column in section "{}".'.format(section))
    key, value = tokens
    if '|' in value:
        value = Choice(json.loads(value).split('|'))
    elif '(' in value and value.endswith(')') and not value.startswith('('):
        open_pos = value.find('(')
        function = value[:open_pos]
        argument = json.loads(value[open_pos + 1:-1])
        value = VALUE_PREDICATES[function](argument)
    else:
        value = json.loads(value)

    if section == 'link':
        context._apply_to_all_nodes[key] = value
    elif section == 'molmeta':
        context.molecule_meta[key] = value
    else:
        raise ValueError('Parser only defined for sections "link" and "molmeta".')


def _parse_meta(tokens, context, context_type, section):
    """
    Parse lines starting with '#meta'.

    The function expects 2 tokens. The first token is assumed to be '#meta' and
    is ignored. The second must be a bracketed token. This second token is
    parsed as a dictionary and updated the dictionary of attributes to add to
    all the nodes involved in a given type of interaction for the context.

    The type of interaction is set by the `section` argument.

    The context is a :class:`vermouth.molecule.Block` or a subclass such as a
    :class:`vermouth.molecule.Link`.

    The `context_type` is a string version of the first level section (i.e
    "link", "block", or "modification").
    """
    if len(tokens) > 2:
        msg = ('Unexpected column when defining meta attributes for section '
               '"{}" of a {}. {} tokens read instead of 2.')
        raise IOError(msg.format(section, context_type, len(tokens)))
    if len(tokens) < 2:
        msg = 'Missing column when defining meta attributes for section "{}" of a {}.'
        raise IOError(msg.format(section, context_type))
    attributes = json.loads(tokens[-1])
    context._apply_to_all_interactions[section].update(attributes)


def _parse_edges(tokens, context, context_type, negate):
    if negate and context_type != 'link':
        raise IOError('The "non-edges" section is only valid in links.')
    atoms = _get_atoms(tokens, natoms=2)
    prefixed_atoms = []
    for atom in atoms:
        prefixed_reference, attributes = _treat_atom_prefix(*atom)
        try:
            apply_to_all_nodes = context._apply_to_all_nodes
        except AttributeError:
            apply_to_all_nodes = {}
        full_attributes = dict(collections.ChainMap(attributes, apply_to_all_nodes))
        prefixed_atoms.append([prefixed_reference, full_attributes])
    if negate:
        context.non_edges.append([prefixed_atoms[0][0], prefixed_atoms[1][1]])
    else:
        error_message = 'Atom with name {} not found for {} {}'
        for prefixed_atom in prefixed_atoms:
            atomname = prefixed_atom[0]
            if atomname not in context and context_type == 'modification':
                raise KeyError(error_message.format(atomname, context_type,
                                                    context.name))
        context.add_edge(prefixed_atoms[0][0], prefixed_atoms[1][0])


def _parse_patterns(tokens, context, context_type):
    if context_type != 'link':
        raise IOError('The "partterns" section is only valid in links.')
    atoms = _get_atoms(tokens, natoms=None)
    context.patterns.append(atoms)


def _parse_variables(tokens, force_field, section):
    if len(tokens) > 2:
        raise IOError('Unexpected column in section "{}".'.format(section))
    if len(tokens) < 2:
        raise IOError('Missing column in section "{}".'.format(section))
    key, value = tokens
    try:
        value = json.loads(value)
    except JSONDecodeError:
        value = str(value)
    force_field.variables[key] = value


def _parse_features(tokens, context, context_type):
    if context_type != 'link':
        raise IOError('The "features" section is only valid in links.')
    context.features.update(set(tokens))


def read_ff(lines, force_field):
    director = FFDirector(force_field)
    return list(director.parse(iter(lines)))
