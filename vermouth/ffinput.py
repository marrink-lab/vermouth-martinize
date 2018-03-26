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
import math
import json
from .molecule import Block, Link, Interaction, DeleteInteraction, Choice


def _tokenize(line):
    """
    Split an interaction line into its elementary components.

    An interaction line is any uncommented and non empty line that follows a
    section header about an interaction type. Such a line is composed of the
    following parts:
    
    * a list of atoms involved in the interaction,
    * an optional delimiter that indicates the end of the atom list,
    * a list of parameters for the interaction.

    The list of atoms is *a minima* a list of atom references. In blocks, these
    references can be atom 1-based indices referring to the order of the atoms
    in the "[ atoms ]" section. It is however more readable, and more robust,
    to refer to atoms by their name. Only the reference by name is allowed in
    links, as links may not have a full "[ atoms ]" section. In links, each
    atom reference can be complemented by atom attributes to specify the  scope
    of the link. These attribute follow the atom reference and are formatted
    like a python dictionary.

    The end-of-atoms delimiter is useful for interaction types that are not
    explicitly encoded in the parser. It allows to indicate when the list of
    atoms ends, and where the list of parameters starts. Two dashes ("--") are
    used as the delimiter. The delimiter is optional for the interaction types
    that are explicitly encoded in the parser and that refer to a fixed number
    of atoms.

    The list of parameters will be copied as-is in an ITP file.

    In its simplest form, an interaction line is what is used in an ITP file.
    Here is an example for a bond:

        2  3  1  0.2  1000

    The two first numbers refer to the second and third atoms of the block,
    respectively. The next three values are the parameters for a bond (*i.e.*
    the function type, the equilibrium distance, and the force constant).

    The two first numbers could be replaced by the corresponding atom names:

        PO4  GL1  1 0.2  1000

    where "PO4" and "GL1" are the names of the second and third atoms of the
    block.

    Optionally, the "--" delimiter can be used after the list of atoms:

        PO4  GL1  --  1 0.2  1000

    If the line is part of a link, then the atom selection may be limited in
    scope. Atom attributes is how to implement such scope limitation:

        BB {'resname': 'ALA', 'secstruc': 'H'} BB {'resname': 'LYS', 'secstruc': 'H', 'order': +1} 1 0.2 1000

    Here, we add a bond to the current link. At one end of the bond is the atom
    named "BB" and annotated as part of an alpha helix ('secstruc': 'H') of a
    residue called "ALA". On the other end of the link is an other
    atom named "BB" that is part of an alpha helix, but that is part of the
    next residue ('order': +1) if this next residue is named "LYS".

    The order parameter has a shortcut in the form of a + or - prefix to the
    atom reference name. Then, "+ATOM" refers to "ATOM" in the next residue,
    and is equivalent to "ATOM {'order': +1}"; "-ATOM" refers to the previous
    residue. There can be multiple + or -, "++ATOM" is equivalent to "ATOM
    {'order': +2}".

    When using attributes, the optional delimiter can increase the readability:

        BB {'resname': 'ALA', 'secstruc': 'H'} +BB {'resname': 'LYS', 'secstruc': 'H'} -- 1 0.2 1000

    Tokens on an interaction line are its different elements. These elements
    are considered as one token each: am atom reference, a set of atom
    attributes, the optional delimiter, each space-separated element of the
    parameter list. The line above splits into the following tokens:

    * ``BB``
    * ``{'resname': 'ALA', 'secstruc': 'H'}``
    * ``+BB``
    * ``{'resname': 'LYS', 'secstruc': 'H'}``
    * ``--``
    * ``1``
    * ``0.2``
    * ``1000``

    Atom attributes can be written next to the previous or the next token
    without an explicit separator. The two following lines yield the same three
    tokens:

        ATOM1{attributes}ATOM2
        ATOM1 {attributes} ATOM2

    Parameters
    ----------
    line: str

    Returns
    -------
    list of str
    """
    separators = ' \t\n'
    tokens = []
    start = 0
    end = -1
    # Find the first non-separator character
    for start, char in enumerate(line):
        if char not in separators:
            break

    # Find the tokens. This has to be a while-loop because we cannot predict
    # what will be the next value of start.
    while start < len(line):
        end = start
        # We count the brackets because if a token starts with an opening
        # bracket, we want to end it with the *matching* closing bracket.
        # Note also that we do not yet implement a way to escape a bracket, nor
        # do we check if the bracket is not part of a string.
        brackets = 0
        for end, end_char in enumerate(line[start:], start=start):
            if end_char == '{':
                # We reached an opening bracket. If it is the first character
                # of the token or if we are already engaged in a bracketized
                # token, then we go on. But if the current token was not
                # a bracketized token, it means we are at the beginning of
                # a new token, so we treat the opening bracket as a separator.
                if not brackets and end != start:
                    end -= 1
                    break
                brackets += 1
            elif end_char == '}':
                brackets -= 1
                if not brackets:
                    break
            elif end_char in separators:
                if not brackets:
                    # We reached a separator. We do not want the separator to
                    # be included in the token, so we push the end by one
                    # character to the left.
                    end -= 1
                    break
        if brackets > 0:
            msg = 'Unexpected end of line. A closing bracket is missing.'
            raise IOError(msg)

        tokens.append(line[start:end + 1])

        # Find the beginning of the next token.
        start = end + 1
        while start < len(line) and line[start] in separators:
            start += 1
    return tokens


def _substitute_macros(line, macros):
    start = None
    while start is None or 0 <= start < len(line):
        start = line.find('$', start)
        if start < 0:
            break
        for end, char in enumerate(line[start + 1:], start=start + 1):
            if char in ' \t\n{}':
                break
        else: # no break
            end += 1
        macro_name = line[start + 1:end]
        macro_value = macros[macro_name]
        line = line[:start] + macro_value + line[end:]
        end = start + len(macro_value)
    return line

        
def _some_atoms_left(tokens, atoms, natoms):
    if tokens and tokens[0] == '--':
        tokens.popleft()
        return False
    if natoms is not None and len(atoms) >= natoms:
        return False
    return True


def _parse_atom_attributes(token):
    attributes = json.loads(token)
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
            if reference[0] in '+-':
                msg = ('Atom names in blocks cannot be prefixed with + or -. '
                       'The name "{}", used in section "{}" of the block "{}" '
                       'is not valid in a block.')
                raise IOError(msg.format(reference, section, block.name))


def _treat_atom_prefix(reference, attributes):
    # If the atom name is prefixed, we can get the order.
    for prefix_end, char in enumerate(reference):
        if char not in '+-':
            break
    prefix = reference[:prefix_end]
    if len(set(prefix)) > 1:
        msg = ('Atom name prefix cannot mix + and -. Atom name "{}" '
               'is not a valid name in section "{}" of a link.')
        raise IOError(msg.format(reference, section))
    
    factors = {'+': +1, '-': -1}
    # If there is no prefix, then `prefix[0]` does not exist. There
    # should, however, always be a `reference[0]` that will be the
    # same as `prefix[0]` is there is a prefix, and not an existing
    # key in `factors` if there is none. So order is 0 * 0 if there
    # is no prefix, which is what we expect.
    order_prefix = factors.get(reference[0], 0) * len(prefix)

    try:
        order_attribute = attributes.get('order', 0)
    except AttributeError as e:
        raise e
    # If the order read from the prefix is 0, then it may just be that
    # the order was not specified by prefix, but only by atom attribute.
    # If the order is not defined in the attributes, it is assumed to
    # be as set by the prefix.
    # If the order is defined in both places, it has to match.
    if (order_prefix and 'order' in attributes) and order_prefix != order_attribute:
        msg = ('The sequence order for atom "{}" in section "{}" of a '
               'link is not consistent between the name prefix '
               '(order={}) and the atom attributes (order={}).')
        raise IOError(msg.format(reference, section,
                                 order_prefix, order_attribute))
    if 'order' not in attributes:
        order_attribute = order_prefix
        attributes['order'] = order_attribute

    # When possible, we favor the prefixed name for references to nodes
    prefix_symbol = '+'
    if order_attribute < 0:
        prefix_symbol = '-'
    atom_name = reference[prefix_end:]
    attributes['atomname'] = attributes.get('atomname', atom_name)
    prefixed_reference = prefix_symbol * int(math.fabs(order_attribute)) + atom_name

    return prefixed_reference, atom_name, attributes


def _treat_link_interaction_atoms(atoms, context, section):
    for reference, attributes in atoms:
        if hasattr(context, '_apply_to_all_nodes'):
            intermediate = context._apply_to_all_nodes.copy()
            intermediate.update(attributes)
            attributes = intermediate

        prefixed_reference, atom_name, attributes = _treat_atom_prefix(reference, attributes)

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
        _treat_block_interaction_atoms(atoms, context, section)
    elif context_type == 'link':
        _treat_link_interaction_atoms(atoms, context, section)


    # Getting the atoms consumed the "--" delimiter if any. So what is left
    # are the interaction parameters or the meta attributes.
    if tokens and tokens[-1].startswith('{'):
        token = tokens.pop()
        meta = json.loads(token)
    else:
        meta = {}
    parameters = list(tokens)

    apply_to_all_interactions = context._apply_to_all_interactions[section]
    meta = dict(collections.ChainMap(meta, apply_to_all_interactions))

    if delete:
        interaction = DeleteInteraction(
            atoms=[atom[0] for atom in atoms],
            atom_attrs=[atom[1] for atom in atoms],
            parameters=parameters,
            meta=meta,
        )
        interaction_list = context.removed_interactions.get(section, [])
        interaction_list.append(interaction)
        context.removed_interactions[section] = interaction_list
    else:
        interaction = Interaction(
            atoms=[atom[0] for atom in atoms],
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
    _, atype, _, resname, name, charge_group = first_six
    if name in context:
        msg = ('There is already an atom named "{}" in the block "{}". '
               'Atom names must be unique within a block.')
        raise IOError(msg.format(name, context.name))
    atom = {
        'atomname': name,
        'atype': atype,
        'resname': resname,
        'charge_group': int(charge_group),
    }
    # charge and mass are optional, but charge has to be defined for mass to be
    if tokens:
        atom['charge'] = float(tokens.popleft())
    if tokens:
        atom['mass'] = float(tokens.popleft())
    context.add_atom(dict(collections.ChainMap(attributes, atom)))


def _parse_link_atom(tokens, context):
    if len(tokens) > 2:
        raise IOError('Unexpected column in link atom definition.')
    elif len(tokens) < 2:
        raise IOError('Missing column in link atom definition.')
    reference = tokens[0]
    attributes = _parse_atom_attributes(tokens[1])
    prefixed_reference, _, attributes = _treat_atom_prefix(reference, attributes)
    attributes = dict(collections.ChainMap(attributes, context._apply_to_all_nodes))
    node_attributes = context.nodes.get(prefixed_reference, {})
    for attr, value in attributes.items():
        if value != node_attributes.get(attr, value):
            msg = ('Conflict in an atom attributes in the definition of a '
                   'link node. Cannot set the attribute "{}" of atom "{}" '
                   'to "{}" because it is already defined as "{}".')
            raise IOError(msg.format(attr, reference, value,
                                     node_attributes[node_attributes[attr]]))
    full_attributes = dict(collections.ChainMap(attributes, node_attributes))
    if prefixed_reference in context.nodes:
        context.nodes[prefixed_reference] = full_attributes
    else:
        context.add_node(prefixed_reference, **full_attributes)


def _parse_macro(tokens, macros):
    macro_name = tokens.popleft()
    macro_value = tokens.popleft()
    if tokens:
        raise IOError('Unexpected column in macro definition.')
    macros[macro_name] = macro_value


def _parse_link_attribute(tokens, context, section):
    if len(tokens) > 2:
        raise IOError('Unexpected column in section "{}".'.format(section))
    elif len(tokens) < 2:
        raise IOError('Missing column in section "{}".'.format(section))
    key, value = tokens
    value = json.loads(value)
    try:
        if '|' in value:
            value = Choice(value.split('|'))
    except TypeError:
        # "value" can be something else than an iterable (bool, number...),
        # then, looking for "|" in it will fail; which is perfectly normal and
        # expected.
        pass
    if section == 'link':
        context._apply_to_all_nodes[key] = value
    elif section == 'molmeta':
        context.molecule_meta[key] = value
    else:
        raise ValueError('Parser only defined for sections "link" and "molmeta".')


def _parse_meta(tokens, context, context_type, section):
    if len(tokens) > 2:
        msg = 'Unexpected column when defining meta attributes for section "{}" of a {}.'
        raise IOError(msg.format(section, context_type))
    elif len(tokens) < 2:
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
        prefixed_reference, _, attributes = _treat_atom_prefix(*atom)
        try:
            apply_to_all_nodes = context._apply_to_all_nodes
        except AttributeError:
            apply_to_all_nodes = {}
        full_attributes = dict(collections.ChainMap(attributes, apply_to_all_nodes))
        prefixed_atoms.append([prefixed_reference, full_attributes])
    if negate:
        context.non_edges.append([prefixed_atoms[0][0], prefixed_atoms[1][1]])
    else:
        context.add_edge(prefixed_atoms[0][0], prefixed_atoms[1][0])


def _parse_patterns(tokens, context, context_type):
    if context_type != 'link':
        raise IOError('The "partterns" section is only valid in links.')
    atoms = _get_atoms(tokens, natoms=None)
    context.patterns.append(atoms)


def read_ff(lines):
    interactions_natoms = {
        'bonds': 2,
        'angles': 3,
        'dihedrals': 4,
        'impropers': 4,
        'constraints': 2,
        'virtual_sites2': 3,
        'pairs': 2,
    }

    macros = {}
    blocks = {}
    links = []
    context_type = None
    context = None
    section = None
    delete = False
    for line_num, line in enumerate(lines, start=1):
        cleaned = _substitute_macros(line.split(';', 1)[0].strip(), macros)
        if not cleaned:
            continue

        tokens = collections.deque(_tokenize(cleaned))

        if cleaned.startswith('['):
            if not cleaned.endswith(']'):
                raise IOError('Misformated section header at line {}.'
                              .format(line_num))
            section = cleaned[1:-1].strip().lower()
            if section.startswith('!'):
                section = section[1:]
                delete = True
            else:
                delete = False
            if section == 'link':
                context_type = 'link'
                context = Link()
                links.append(context)
        elif section == 'moleculetype':
            context_type = 'block'
            context = Block()
            name, nrexcl = cleaned.split()
            context.name = name
            context.nrexcl = int(nrexcl)
            blocks[name] = context
        elif section == 'macros':
            context = None
            context_type = None
            _parse_macro(tokens, macros)
        elif section == 'link':
            _parse_link_attribute(tokens, context, section)
        elif section == 'molmeta':
            _parse_link_attribute(tokens, context, section)
        elif section == 'atoms':
            if context_type == 'block':
                _parse_block_atom(tokens, context)
            elif context_type == 'link':
                _parse_link_atom(tokens, context)
        elif section == 'non-edges':
            _parse_edges(tokens, context, context_type, negate=True)
        elif section == 'edges':
            _parse_edges(tokens, context, context_type, negate=False)
        elif section == 'patterns':
            _parse_patterns(tokens, context, context_type)
        elif tokens[0] == '#meta':
            _parse_meta(tokens, context, context_type, section)
        else:
            natoms = interactions_natoms.get(section)
            try:
                _base_parser(tokens, context, context_type, section,
                             natoms=natoms, delete=delete)
            except Exception:
                raise IOError('Error while reading line {} in section {}.'
                              .format(line_num, section))

    # Finish the blocks and the links.
    # Because of hos they are described in gromacs, proper and improper
    # dihedral angles are all under the [ dihedrals ] section. However
    # the way they are treated differently in the library, at least on how
    # they generate edges. Here we move the all the impropers into their own
    # [ impropers ] section.
    for block in blocks.values():
        propers = []
        impropers = []
        for dihedral in block.interactions.get('dihedrals', []):
            if dihedral.parameters and dihedral.parameters[0] == '2':
                impropers.append(dihedral)
            else:
                propers.append(dihedral)
        block.interactions['dihedrals'] = propers
        block.interactions['impropers'] = impropers
    # We have the nodes and The # interactions, but the edges are missing.
    for block in blocks.values():
        block.make_edges_from_interactions()
    for link in links:
        link.make_edges_from_interactions()

    # For debug purpose, we add a comment to the link interactions so
    # they can be easily identified in the output topology.
    for link in links:
        for interactions in link.interactions.values():
            for interaction in interactions:
                interaction.meta['comment'] = 'Link'

    return blocks, links
