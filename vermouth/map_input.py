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
Read force field to force field mappings.
"""

from pathlib import Path
import collections
import itertools


def read_mapping_file(lines):
    """
    Partial reader for modified Backward mapping files.

    Read mappings from a Backward mapping file. Not all fields are supported,
    only the "molecule" and the "atoms" fields are read. If not explicitly
    specified, the origin force field for a molecule is assumed to be
    "universal", and the destination force field is assumed to be "martini22".

    The mapping collection is a 3 level dictionary where the first key is the
    name of the initial force field, the second key is the name of the
    destination force field, and the third key is the name of the molecule.

    Parameters
    ----------
    lines: collections.abc.Iterable[str]
        Collection of lines to read.

    Return
    ------
    dict
    """
    lines = iter(lines)
    mappings = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Throw away everything before [ molecule ]
    # If `lines` is empty, then `line_number` is never set by the loop and
    # pylint really does not like it. It should not matter that `line_number`
    # is not set, because it means an exception will be raised anyway. It does
    # not hust to define the variable before the loop, though.
    line_number = None
    for line_number, line in enumerate(lines, start=1):
        cleaned = line.split(';', 1)[0].strip()
        if (cleaned.startswith('[')
                and cleaned.endswith(']')
                and cleaned[1:-1].strip() == 'molecule'):
            break
    else:  # no break
        msg = ('No mapping defined. '
               'A mapping must start with a [ molecule ] section.')
        raise IOError(msg)

    # At this point the last line read was [ molecule ] or else we would have
    # raised an exception.
    while True:
        full_mapping = _read_mapping_partial(lines, line_number + 1)
        name, from_ff_list, to_ff_list, mapping, weights, extra, line_number = full_mapping
        if name is not None:
            for from_ff, to_ff in itertools.product(from_ff_list, to_ff_list):
                mappings[from_ff][to_ff][name] = (mapping, weights, extra)
        else:
            break

    # We do not want to return defaultdicts.
    mappings = _default_to_dict(mappings)

    return mappings


def _default_to_dict(mappings):
    """
    Convert a mapping collection from a defaultdict to a dict.
    """
    if isinstance(mappings, dict):
        return {
            key: _default_to_dict(value)
            for key, value in mappings.items()
        }
    else:
        return mappings


def _compute_weights(mapping, name):
    """
    Calculate the mapping weights from a preliminary mapping.

    The preliminary mapping refers to atoms by their names instead of the tuple
    ``(resid, name)`` used in a final mapping. Target atoms with null weights
    are prefixed with a '!'. The result dictionary also refers to atoms
    directly by name.

    Parameters
    ----------
    mapping: dict[str, list[str]]
        Preliminary mapping.
    name: str
        Molecule name. Used to generate error messages.

    Returns
    -------
    dict
    """
    rev_mapping = collections.defaultdict(list)
    for from_atom, to_atoms in mapping.items():
        for to_atom in to_atoms:
            rev_mapping[to_atom].append(from_atom)

    # Atoms can be mapped with a null weight by prefixing the target particle
    # with a "!". We first set the non-null weights.
    pre_weights = {
        from_atom: dict(collections.Counter([
            to_atom for to_atom in to_atoms if not to_atom.startswith('!')
        ]))
        for from_atom, to_atoms in mapping.items()
    }
    for atom_weights in pre_weights.values():
        total = sum(atom_weights.values())
        for to_atom in atom_weights:
            atom_weights[to_atom] /= total
    weights = collections.defaultdict(dict)
    for from_atom, to_atoms in pre_weights.items():
        for to_atom, weight in to_atoms.items():
            weights[to_atom][from_atom] = weight

    # Then we add the null weights.
    null_weights = {
        to_atom[1:]: {from_atom: 0 for from_atom in from_atoms}
        for to_atom, from_atoms in rev_mapping.items()
        if to_atom.startswith('!')
    }
    for to_atom, from_weights in null_weights.items():
        null_keys = set(from_weights.keys())
        non_null_keys = set(weights.get(to_atom, {}).keys())
        redefined_keys = null_keys & non_null_keys
        if redefined_keys:
            msg = ('Atom(s) {} is mapped to "{}" with and without a weight '
                   'in the molecule "{}". '
                   'There cannot be the same target atom name with and '
                   'without a "!" prefix on a same line.')
            raise IOError(msg.format(redefined_keys, to_atom, name))
        weights[to_atom] = weights.get(to_atom, {})
        weights[to_atom].update(from_weights)

    return weights


def _read_mapping_partial(lines, start_line):
    """
    Partial reader for modified Backward mapping files.

    Read mapping from a Backward mapping file. Not all fields are supported,
    only the "molecule" and the "atoms" fields are read. The origin force field
    is assumed to be "universal", and the destination force field is assumed to
    be "martini22".

    The content is assumed to start just *after* the '[ molecule  ]' line. The
    function consumes the iterator until, and including, the next
    '[ molecule ]' line if any.

    Parameters
    ----------
    lines: collections.abc.Iterator[str]
        Collection of lines to read.
    start_line: int
        The first line number. Starts at 1.

    Returns
    -------
    name: str
        The name of the fragment as read in the "molecule" field.
    from_ff: list[str]
        A list of force field origins. Each force field is referred by name.
    to_ff: list[str]
        A list of force field destinations. Each force field is referred by name.
    mapping: dict[tuple[int, str], list[tuple[int, str]]]
        The mapping. The keys of the dictionary are pairs of residue indices
        and the atom names in the origin force field as tupples (resid,
        atomname); the values are lists of resid, and atom names pairs
        in the destination force field.
    weights: dict
    extra: list
        Unmapped atoms to be added.
    line_number: int
        The number of the last line consumed.
    """
    from_ff = []
    to_ff = []
    mapping = {}
    extra = []
    context = 'molecule'
    name = None

    has_content = False
    line_number = None
    for line_number, line in enumerate(lines, start=start_line):
        cleaned = line.split(';', 1)[0].strip()
        if not cleaned:
            continue
        elif cleaned.startswith('['):
            if not cleaned.endswith(']'):
                raise IOError('Format error at line {}.'.format(line_number))
            context = cleaned[1:-1].strip()
            if context == 'molecule':
                if not has_content:
                    # This detects cases where the file looks like:
                    # [ molecule ]
                    # [ molecule ]
                    #
                    # Then the partial function gets:
                    # [ molecule ]
                    #
                    # as the first section header is consimed by the parent
                    # function.
                    raise IOError('Mapping starting at line {} is empty.'
                                  .format(start_line))
                break
        elif context == 'molecule':
            if name is None:
                name = cleaned
            else:
                msg = ('Line {} tries to redefine the name of the molecule '
                       'starting at line {}.')
                raise IOError(msg.format(line_number, start_line))
        elif context == 'atoms':
            _, from_atom, *to_atoms = cleaned.split()
            if from_atom in mapping:
                msg = ('At line {}, the atom "{}", that is already defined, '
                       'get defined again.')
                raise IOError(msg.format(line_number, from_atom))
            mapping[from_atom] = to_atoms
        elif context in ['from', 'mapping']:
            from_ff.extend(cleaned.split())
        elif context == 'to':
            to_ff.extend(cleaned.split())
        elif context == 'extra':
            extra.extend(cleaned.split())
        # else: we are dealing with a section that we should ignore.
        has_content = True

    if name is None and context != 'molecule':
        # At this point, there are two cases where the name can be None:
        # either it was not defined, or there was no content to read. In the
        # later case, the context was not changed from its initial value.
        # Here, we are in the former case, and the name was not defined. This
        #case is an erroroneous one.
        msg = ('The mapping starting at line {} is defined without a name. '
               'The block name must follow the [ molecule ] section.')
        raise IOError(msg.format(start_line))

    weights = _compute_weights(mapping, name)

    # While it is not supported by the file format, mappings can contain
    # residue information. Atom identifiers in all the outputs must be formated
    # as `(residue, name)`; yet, we only use the name so far. Here we set the
    # residue to 0 for all the atoms assuming we only deal with single residue
    # mappings.
    # We use the opportunity to clean out the null-weight mappings.
    mapping = {
        (0, from_atom): [(0, to_atom if not to_atom.startswith('!') else to_atom[1:])
                         for to_atom in to_atoms]
        for from_atom, to_atoms in mapping.items()
    }
    weights = {
        (0, to_atom): {(0, from_atom): weight
                       for from_atom, weight in from_weights.items()}
        for to_atom, from_weights in weights.items()
    }

    # If not specified in the file, we assume the mapping is for
    # universal -> martini22
    if not from_ff:
        from_ff = ['universal', ]
    if not to_ff:
        to_ff = ['martini22', ]

    return name, from_ff, to_ff, mapping, weights, extra, line_number


def read_mapping_directory(directory):
    """
    Read all the mapping files in a directory.

    The resulting mapping collection is a 3-level dict where the keys are:
    * the name of the origin force field
    * the name of the destination force field
    * the name of the residue

    The values after these 3 levels is a mapping dict where the keys are the
    atom names in the origin force field and the values are lists of names in
    the destination force field.

    Parameters
    ----------
    directory: str
        The path to the directory to search. Files with a '.map' extension will
        be read. There is no recursive search.

    Returns
    -------
    dict
        A collection of mappings.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError('"{}" is not a directory.'.format(directory))
    mappings = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in directory.glob('**/*.map'):
        with open(str(path)) as infile:
            try:
                new_mappings = read_mapping_file(infile)
            except IOError:
                raise IOError('An error occured while reading "{}".'.format(path))
            else:
                combine_mappings(mappings, new_mappings)
    return dict(mappings)


def generate_self_mappings(blocks):
    """
    Generate self mappings from a collection of blocks.

    A self mapping is a mapping that maps a force field to itself. Applying
    such mapping is applying a neutral transformation.

    Parameters
    ----------
    blocks: dict[str, networkx.Graph]
        A dictionary of blocks with block names as keys and the blocks
        themselves as values. The blocks must be instances of :class:`networkx.Graph`
        with each node having an 'atomname' attribute.

    Returns
    -------
    mappings: dict[str, tuple]
        A dictionary of mappings where the keys are the names of the blocks,
        and the values are tuples like (mapping, weights, extra).

    Raises
    ------
    KeyError
        Raised if a node does not have am 'atomname' attribute.

    See Also
    --------
    read_mapping_file
        Read a mapping from a file.
    generate_all_self_mappings
        Generate self mappings for a list of force fields.
    """
    mappings = {}
    for name, block in blocks.items():
        mapping = {
            (0, atom['atomname']): [(0, atom['atomname'])]
            for atom in block.nodes.values()
        }
        weights = {
            (0, atom['atomname']): {(0, atom['atomname']): 1}
            for atom in block.nodes.values()
        }
        extra = []
        mappings[name] = (mapping, weights, extra)
    return mappings


def generate_all_self_mappings(force_fields):
    """
    Generate self mappings for a list of force fields.

    Parameters
    ----------
    force_fields: collections.abc.Iterable
        List of instances of :class:`~vermouth.forcefield.ForceField`.

    Returns
    -------
    dict
        A collection of mappings formatted as the output of the
        :func:`read_mapping_directory` function.
    """
    mappings = collections.defaultdict(dict)
    for force_field in force_fields:
        name = force_field.name
        mappings[name][name] = generate_self_mappings(force_field.blocks)
    return dict(mappings)


def combine_mappings(known_mappings, partial_mapping):
    """
    Update a collection of mappings.

    Add the mappings from the 'partial_mapping' argument into the
    'known_mappings' collection. Both arguments are collections of mappings
    similar to the output of the :func:`read_mapping_directory` function. They
    are dictionary with 3 levels of keys: the name of the initial force field,
    the name of the target force field, and the name of the block. The values
    in the third level dictionary are tuples of (mapping, weights, extra).

    If a force field appears in 'partial_mapping' that is not in
    'known_mappings', then it is added. For existing pairs of initial and
    target force fields, the blocks are updated and the version in
    'partial_mapping' is kept in priority.

    Parameters
    ----------
    known_mappings: dict
        Collection of mapping to update **in-place**.
    partial_mapping: dict
        Collection of mappings to update from.
    """
    for origin, destinations in partial_mapping.items():
        known_mappings[origin] = known_mappings.get(origin, {})
        for destination, residues in destinations.items():
            known_mappings[origin][destination] = known_mappings[origin].get(destination, {})
            for residue, mapping in residues.items():
                known_mappings[origin][destination][residue] = mapping
