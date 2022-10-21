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

from itertools import combinations
import numbers

import networkx as nx
from numpy import sign

from ..molecule import attributes_match
from .processor import Processor


def _atoms_match(node1, node2):
    return attributes_match(node1, node2, ignore_keys=('order', 'replace'))


def _is_valid_non_edges(molecule, link, rev_raw_match):
    for from_node, to_node_attrs in link.non_edges:
        if from_node not in link:
            continue
        from_mol_node_name = rev_raw_match[from_node]
        from_mol = molecule.nodes[from_mol_node_name]
        # from_link = link.nodes[from_node]
        from_resid = from_mol['resid']
        # from_order = from_link.get('order', 0)
        for neighbor in molecule.neighbors(from_mol_node_name):
            to_mol = molecule.nodes[neighbor]
            to_link = to_node_attrs
            to_resid = to_mol['resid']
            to_order = to_link.get('order', 0)
            if to_resid == from_resid + to_order and _atoms_match(to_mol, to_link):
                return False
    return True


def _pattern_match(molecule, atoms, raw_match):
    for link_key, template_attr in atoms:
        molecule_key = raw_match[link_key]
        molecule_attr = molecule.nodes[molecule_key]
        if not _atoms_match(molecule_attr, template_attr):
            return False
    return True


def _any_pattern_match(molecule, patterns, rev_raw_match):
    return any(_pattern_match(molecule, atoms, rev_raw_match) for atoms in patterns)


def _interpret_order(order):
    error_msg = ('"{}" is not a valid value for the "order" node attribute. '
                 'The value must be an integer, a series of + '
                 '(i.e. >, >>, >>>, ...), a series of <, or a series of *.')
    if order is True or order is False:
        # Booleans match the Number abstract base class, so we have to test
        # for them separately.
        raise ValueError(error_msg.format(order))
    elif isinstance(order, numbers.Number):
        if int(order) != float(order):
            # order is a number but not an int (or int-like)
            raise ValueError(error_msg.format(order))
        order_type = 'number'
        order_value = order
    else:
        try:
            first_character = order[0]
        except (TypeError, IndexError, KeyError):
            # order is not an int, nor a sequence (str, list, tuple,...)
            # or it is an empty sequence. Anyway, we cannot work with it.
            raise ValueError(error_msg.format(order))
        if len(set(order)) != 1 or first_character not in '><*':
            # order is a str (or any sequence, we do not really care),
            # but it contains a mixture of characters (e.g. '+-'), or
            # the characters are not among the ones we expect.
            raise ValueError(error_msg.format(order))
        signs = {'>': +1, '<': -1}
        if first_character in signs:
            order_type = '><'
            order_value = signs[first_character] * len(order)
        elif first_character == '*':
            # This could be an 'else', but it would hide bugs if the code
            # above changes.
            order_type = '*'
            order_value = len(order)
    return order_type, order_value


def match_order(order1, resid1, order2, resid2):
    r"""
    Check if two residues match the order constraints.

    The order can be:

    an integer
        It is then the expected distance in resid with a reference residue.
    a series of >
        This indicates that the residue must have a larger resid than a
        reference residue. Multiple atoms with the same number of > are
        expected to be part of the same residue. The more > are in the serie,
        the further away the residue is expected to be from the reference, so a
        residue with >> is expected to have a greater resid than a residue with
        >.
    a series of <
        Same as a series of >, but for smaller resid.
    a series of *
        This indicates a different residue than the reference, but without a
        specified order. As for the > or the <, atoms with the same number of *
        are expected to be part of the same residue.

    The comparison matrix can be sumerized as follow, with 0 being the
    reference residue, n being an integer. In the matrix, a ? means that the
    result depends on the comparison of the actual numbers, a ! means that the
    comparison should not be considered, and / means that the resids must be
    different. The rows correspond to the order at the left of the comparison
    (order1 argument), while the columns correspond to the order at the right
    of it (order2 argument).

    +-----+---+----+---+----+---+---+----+-----+
    |     | > | >> | < | << | n | 0 | \* | \** |
    +-----+---+----+---+----+---+---+----+-----+
    | >   | = | <  | > | >  | ! | > | !  | !   |
    +-----+---+----+---+----+---+---+----+-----+
    | >>  | > | =  | > | >  | ! | > | !  | !   |
    +-----+---+----+---+----+---+---+----+-----+
    | <   | < | <  | = | >  | ! | < | !  | !   |
    +-----+---+----+---+----+---+---+----+-----+
    | <<  | < | <  | < | =  | ! | < | !  | !   |
    +-----+---+----+---+----+---+---+----+-----+
    | n   | ! | !  | ! | !  | ? | ? | !  | !   |
    +-----+---+----+---+----+---+---+----+-----+
    | 0   | < | <  | > | >  | ? | = | /  | /   |
    +-----+---+----+---+----+---+---+----+-----+
    | \*  | ! | !  | ! | !  | ! | / | =  | /   |
    +-----+---+----+---+----+---+---+----+-----+
    | \** | ! | !  | ! | !  | ! | / | /  | =   |
    +-----+---+----+---+----+---+---+----+-----+

    Parameters
    ----------
    order1: int or str
        The order attribute of the residue on the left of the comparison.
    resid1: int
        The residue id of the residue on the left of the comparison.
    order2: int or str
        The order attribute of the residue on the right of the comparison.
    resid2: int
        The residue id of the residue on the right of the comparison.

    Returns
    -------
    bool
        `True` if the conditions match.

    Raises
    ------
    ValueError
        Raised if the order arguments do not follow the expected format.
    """
    # Validate the order arguments, and format it for what comes next.
    orders = []
    order_types = []
    for order in (order1, order2):
        order_type, order_value = _interpret_order(order)
        order_types.append(order_type)
        orders.append(order_value)

    if order_types[0] == 'number':  # Rows n and 0 in the comparison matrix
        if order_types[1] == 'number':
            # Columns n, and 0
            if (orders[1] - orders[0]) != (resid2 - resid1):
                return False
        elif orders[0] == 0:  # Row 0 in the comparison matrix
            if order_types[1] == '><' and sign(resid2 - resid1) != sign(orders[1]):
                # Columns >, >>, <, and <<
                return False
            elif order_types[1] == '*' and resid1 == resid2:
                # Columns *, and **
                return False
    elif order_types[0] == '><':  # Rows >, >>, <, and <<
        if (order_types[1] == 'number' and orders[1] == 0
                and sign(resid1 - resid2) != sign(orders[0])):
            # Column 0
            return False
        elif (order_types[1] == '><'
              and sign(resid2 - resid1) != sign(orders[1] - orders[0])):
            # Column >, >>, <, and <<
            return False
    elif order_types[0] == '*':  # Rows *, and **
        if order_types[1] == 'number' and orders[1] == 0 and resid1 == resid2:
            # Column 0
            return False
        elif order_types[1] == '*' and ((orders[0] == orders[1]) != (resid1 == resid2)):
            # Columns *, and **
            return False

    return True


def match_link(molecule, link):
    if not attributes_match(molecule.meta, link.molecule_meta):
        return

    GM = nx.isomorphism.GraphMatcher(molecule, link, node_match=_atoms_match)

    raw_matches = GM.subgraph_isomorphisms_iter()
    for raw_match in raw_matches:
        # raw_match: mol -> link
        # rev_raw_match: link -> mol
        rev_raw_match = {value: key for key, value in raw_match.items()}
        if not _is_valid_non_edges(molecule, link, rev_raw_match):
            continue
        any_pattern_match = _any_pattern_match(molecule, link.patterns, rev_raw_match)
        if link.patterns and (not any_pattern_match):
            continue
        order_match = {}
        for mol_idx, link_idx in raw_match.items():
            mol_node = molecule.nodes[mol_idx]
            link_node = link.nodes[link_idx]
            if 'order' in link_node:
                order = link_node['order']
                resid = mol_node['resid']
                if order not in order_match:
                    order_match[order] = resid
                # Assert all orders correspond to the same resid
                elif order in order_match and order_match[order] != resid:
                    break
        else:  # No break
            for ((order1, resid1), (order2, resid2)) in combinations(order_match.items(), 2):
                # Assert the differences between resids correspond to what
                # the orders require.
                if not match_order(order1, resid1, order2, resid2):
                    break
            else:  # No break
                # raw_match is molecule -> link. The other way around is more
                # useful
                yield {v: k for k, v in raw_match.items()}


def _build_link_interaction_from(molecule, interaction, match):
    atoms = tuple(match[idx] for idx in interaction.atoms)
    parameters = [
        param(molecule, match) if callable(param) else param
        for param in interaction.parameters
    ]
    new_interaction = interaction._replace(
        atoms=atoms,
        parameters=parameters
    )
    return new_interaction


class DoLinks(Processor):
    """
    Apply Links, taken from a molecule's force field, to the molecule.
    """
    def run_molecule(self, molecule):
        # TODO: Separate this into a function
        links = molecule.force_field.links
        _nodes_to_remove = []
        for link in links:
            matches = match_link(molecule, link)
            for match in matches:
                for node, node_attrs in link.nodes.items():
                    if 'replace' in node_attrs:
                        if node_attrs['replace'].get('atomname', False) is None:
                            _nodes_to_remove.append(match[node])
                        else:
                            node_mol = molecule.nodes[match[node]]
                            node_mol.update(node_attrs['replace'])
                for inter_type, interactions in link.removed_interactions.items():
                    for interaction in interactions:
                        interaction = _build_link_interaction_from(molecule, interaction, match)
                        try:
                            molecule.remove_matching_interaction(inter_type, interaction)
                        except ValueError:
                            pass
                for inter_type, interactions in link.interactions.items():
                    for interaction in interactions:
                        interaction = _build_link_interaction_from(molecule, interaction, match)
                        molecule.add_or_replace_interaction(inter_type, *interaction, link.citations)

                for loglevel, entries in link.log_entries.items():
                    for entry, fmt_args in entries.items():
                        fmt_args = fmt_args + [match]
                        molecule.log_entries[loglevel][entry] += fmt_args

            molecule.remove_nodes_from(_nodes_to_remove)
        return molecule
