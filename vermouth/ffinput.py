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
from .molecule import Block, Link, Interaction


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

        BB {'resname': ('ALA', 'GLY'), 'secstruc': 'H'} BB {'resname': ('LYS', 'ARG'), 'secstruc': 'H', 'order': +1} 1 0.2 1000

    Here, we add a bond to the current link. At one end of the bond is the atom
    named "BB" and annotated as part of an alpha helix ('secstruc': 'H') of a
    residue called "ALA" or "GLY". On the other end of the link is an other
    atom named "BB" that is part of an alpha helix, but that is part of the
    next residue ('order': +1) if this next residue is named "LYS" or "ARG".

    The order parameter has a shortcut in the form of a + or - prefix to the
    atom reference name. Then, "+ATOM" refers to "ATOM" in the next residue,
    and is equivalent to "ATOM {'order': +1}"; "-ATOM" refers to the previous
    residue. There can be multiple + or -, "++ATOM" is equivalent to "ATOM
    {'order': +2}".

    When using attributes, the optional delimiter can increase the readability:

        BB {'resname': ('ALA', 'GLY'), 'secstruc': 'H'} +BB {'resname': ('LYS', 'ARG'), 'secstruc': 'H'} -- 1 0.2 1000

    Tokens on an interaction line are its different elements. These elements
    are considered as one token each: am atom reference, a set of atom
    attributes, the optional delimiter, each space-separated element of the
    parameter list. The line above splits into the following tokens:

    * ``BB``
    * ``{'resname': ('ALA', 'GLY'), 'secstruc': 'H'}``
    * ``+BB``
    * ``{'resname': ('LYS', 'ARG'), 'secstruc': 'H'}``
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

        
def _some_atoms_left(tokens, atoms, natoms):
    if tokens and tokens[0] == '--':
        tokens.popleft()
        return False
    if natoms is not None and len(atoms) >= natoms:
        return False
    return True


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
            atoms.append([token, next_token])
            tokens.popleft()
        else:
            atoms.append([token, ])
    return atoms


def _base_parser(line, context, section, natoms=None):
    tokens = collections.deque(_tokenize(line))

    delimiter_count = tokens.count('--')
    if delimiter_count > 1:
        msg = 'There can be 0 or 1 "--" delimiter; {} found.'
        raise IOError(msg.format(delimiter_count))

    # Group the atoms and their attributes
    atoms = _get_atoms(tokens, natoms)
    if natoms is not None and len(atoms) != natoms:
        raise IOError('Found {} atoms while {} were expected.'
                      .format(len(atoms), natoms))

    # Normalize the atom references. We use names as references, so we need to
    # convert indices to names. This normalization is only relevant for blocks.
    # Links need to reference atoms by name.
    atom_names = list(context.nodes)
    for atom in atoms:
        reference = atom[0]
        if reference.isdigit():
            # The indices in the file are 1-based
            reference = int(reference) - 1
            reference = atom_names[reference]
            atom[0] = reference

    # Getting the atoms consumed the "--" delimiter if any. So what is left
    # are the interaction parameters.
    parameters = list(tokens)

    interaction = Interaction(
        atoms=[atom[0] for atom in atoms],
        parameters=parameters,
        meta={},
    )
    interaction_list = context.interactions.get(section, [])
    interaction_list.append(interaction)
    context.interactions[section] = interaction_list


def _parse_atom(line, context):
    _, atype, _, resname, name, charge_group, charge = line.split()
    atom = {
        'atomname': name,
        'atype': atype,
        'resname': resname,
        'charge': float(charge),
        'charge_group': int(charge_group),
    }
    context.add_atom(atom)


def read_ff(lines):
    interactions_natoms = {
        'bonds': 2,
        'angles': 3,
        'dihedrals': 4,
        'impropers': 4,
        'constraints': 2,
    }

    blocks = {}
    links = []
    context = None
    section = None
    for line_num, line in enumerate(lines, start=1):
        cleaned = line.split(';', 1)[0].strip()
        if not cleaned:
            continue

        if cleaned.startswith('['):
            if not cleaned.endswith(']'):
                raise IOError('Misformated section header at line {}.'
                              .format(line_num))
            section = cleaned[1:-1].strip().lower()
            if section == 'link':
                context = Link()
                links.append(context)
        elif section == 'moleculetype':
            context = Block()
            name, nrexcl = cleaned.split()
            context.name = name
            context.nrexcl = int(nrexcl)
            blocks[name] = context
        elif section == 'atoms':
            _parse_atom(cleaned, context)
        else:
            natoms = interactions_natoms.get(section)
            try:
                _base_parser(cleaned, context, section, natoms)
            except Exception:
                raise IOError('Error while reading line {} in section {}.'
                              .format(line_num, section))

        # Finish the blocks and the links. We have the nodes and The
        # interactions, but the edges are missing.
        for name, block in blocks.items():
            block.make_edges_from_interactions()
        for link in links:
            link.make_edges_from_interactions()

    return blocks, links

