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
Provides a processor that can perform a resolution transformation on a
molecule.
"""
from collections import defaultdict, Counter
from functools import partial
from itertools import product, combinations, groupby

import networkx as nx

from ..map_parser import parse_mapping_file
from ..molecule import Molecule, Block, attributes_match
from .processor import Processor
from ..utils import are_all_equal, format_atom_string
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


def build_graph_mapping_collection(from_ff, to_ff, mappings):
    """
    Function that produces a collection of :class:`vermouth.map_parser.Mapping`
    objects.
    Hereby deprecated.

    Parameters
    ----------
    from_ff: vermouth.forcefield.ForceField
        Origin force field.
    to_ff: vermouth.forcefield.ForceField
        Destination force field.
    mappings: dict[str, dict[str, vermouth.map_parser.Mapping]]
        All known mappings

    Returns
    -------
    collections.abc.Iterable
        A collection of mappings that map from `from_ff` to `to_ff`.
    """
    return mappings[from_ff.name][to_ff.name].values()


def edge_matcher(graph1, graph2, node11, node12, node21, node22):
    """
    Checks whether the resids for node11 and node12 in graph1 are the same, and
    whether that's also true for node21 and node22 in graph2.
    """
    node11 = graph1.nodes[node11]
    node12 = graph1.nodes[node12]
    node21 = graph2.nodes[node21]
    node22 = graph2.nodes[node22]
    return (node11.get('resid') == node12.get('resid')) ==\
           (node21.get('resid') == node22.get('resid'))


def node_matcher(node1, node2):
    return attributes_match(node1, node2,
                            ignore_keys=('atype', 'charge', 'charge_group',
                                         'resid', 'replace', '_old_atomname'))


def _old_atomname_match(node1, node2):
    name1 = node1.get('_old_atomname', node1['atomname'])
    name2 = node2.get('_old_atomname', node2['atomname'])
    node1 = node1.copy()
    node2 = node2.copy()
    node1['_name'] = name1
    node2['_name'] = name2
    del node1['atomname']
    del node2['atomname']
    return node_matcher(node1, node2)


def ptm_resname_match(node1, node2):
    if 'resname' in node2 and not node2['resname']:
        node2 = node2.copy()
        del node2['resname']
    if 'PTM_atom' in node2 and not node2['PTM_atom']:
        del node2['PTM_atom']
    is_equal = node_matcher(node1, node2)
    return is_equal


def cover(to_cover, options):
    if not to_cover:
        return []
    for idx, option in enumerate(options):
        if all(item in to_cover for item in option):
            left_to_cover = to_cover.copy()
            for item in option:
                # Only remove the leftmost item. PS. we know for sure all items
                # in option are in left_to_cover at least once.
                left_to_cover.remove(item)
            found = cover(left_to_cover, options[idx:])
            if found is not None:
                return [option] + found
    return None


def get_mod_mappings(mappings):
    """
    Returns a dict of all known modification mappings.

    Parameters
    ----------
    mappings: collections.abc.Iterable[vermouth.map_parser.Mapping]
        All known mappings.

    Returns
    -------
    dict[tuple[str], vermouth.map_parser.Mapping]
        All mappings that describe a modification mapping.
    """
    out = {}
    for mapping in mappings:
        if mapping.type == 'modification':
            out[mapping.names] = mapping
    return out


def modification_matches(molecule, mappings):
    """
    Returns a minimal combination of modification mappings and where they
    should be applied that describes all modifications in `molecule`.

    Parameters
    ----------
    molecule: networkx.Graph
        The molecule whose modifications should be treated. Modifications are
        described by the 'modifications' node attribute.
    mappings: collections.abc.Iterable[vermouth.map_parser.Mapping]
        All known mappings.

    Returns
    -------
    list[tuple[dict, vermouth.molecule.Link, dict]]
        A list with the following items:
            Dict describing the correspondence of node keys in `molecule` to
                node keys in the modification.
            The modification.
                Dict with all reference atoms, mapping modification nodes to
                    nodes in `molecule`.
    """
    modified_nodes = set()  # This will contain whole residues.
    for idx, node in molecule.nodes.items():
        if node.get('modifications', []):
            modified_nodes.add(idx)
    ptm_subgraph = molecule.subgraph(modified_nodes)
    grouped = nx.connected_components(ptm_subgraph)
    found_ptm_groups = set()
    for group in grouped:
        modifications = {tuple(molecule.nodes[mol_idx].get('modifications', []))
                         for mol_idx in group}
        # Every group of PTMs should have the same modifications
        assert len(modifications) == 1
        found_ptm_groups.update(modifications)

    needed_mod_mappings = set()
    known_mod_mappings = get_mod_mappings(mappings)
    for group in found_ptm_groups:
        # known_mod_mappings is a dict[tuple[str], Mapping]. We want to know
        # the minimal combination of those needed to cover all the PTMs found
        # in group. The cheapest solution is covering the names of the PTMs in
        # group with keys from known_mod_mappings. An improvement would be to
        # do the graph covering again.
        # TODO?
        covered_by = cover([ptm.name for ptm in group],
                           sorted(known_mod_mappings, key=len, reverse=True))
        if covered_by is None:
            LOGGER.warning("Can't find modification mappings for the "
                           "modifications {}. The following modification "
                           "mappings are known: {}",
                           [ptm.name for ptm in group], known_mod_mappings)
            return []
        needed_mod_mappings.update(covered_by)

    # type: atoms: modifications
    applied_interactions = defaultdict(lambda: defaultdict(list))
    to_remove = set()
    touched_atoms = defaultdict(list)
    all_matches = []
    # Sort on the tuple[str] type names of the mappings so that mappings that
    # define most modifications at the same time get processed first
    for mod_name in sorted(needed_mod_mappings, key=len, reverse=True):
        mod_mapping = known_mod_mappings[mod_name]
        # TODO: include modifications in matching criterion (just add it to the
        #       modification from-block).
        # For now, just make sure the intersection with the modified_nodes is
        # not empty.
        for mol_to_mod, modification, references in mod_mapping.map(molecule, node_match=ptm_resname_match):
            if not modified_nodes & set(mol_to_mod):
                continue
            all_matches.append((mol_to_mod, modification))
            LOGGER.info('Applying modification mapping {}', modification.name, type='general')
            mod_to_mol = defaultdict(dict)
            for mol_idx, mod_idxs in mol_to_mod.items():
                for mod_idx in mod_idxs:
                    mod_to_mol[mod_idx][mol_idx] = mol_to_mod[mol_idx][mod_idx]
            mod_to_mol = dict(mod_to_mol)
            if not set(mol_to_mod) <= modified_nodes:
                # TODO: better message
                LOGGER.warning('Overlapping modification mappings')
            modified_nodes -= set(mol_to_mod)

    return matches


def apply_block_mapping(match, molecule, graph_out, mol_to_out, out_to_mol):
    """
    Performs a mapping operation for a "block". `match` is a tuple of 3
    elements that describes what nodes in `molecule` should correspond to
    a :class:`vermouth.molecule.Block` that should be added to `graph_out`, and
    any atoms that should be used a references.
    Add the required :class:`vermouth.molecule.Block` to `graph_out`, and
    updates `mol_to_out` and `out_to_mol` *in-place*.

    Parameters
    ----------
    match
    molecule: networkx.Graph
        The original molecule
    graph_out: vermouth.molecule.Molecule
        The newly created graph that describes `molecule` at a different
        resolution.
    mol_to_out: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `molecule` to nodes in `graph_out` with the
        associated weights.
    out_to_mol: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `graph_out` to nodes in `molecule` with the
        associated weights.

    Returns
    -------
    set
        A set of all overlapping nodes that were already mapped before.
    set
        A set of none-to-one mappings. I.e. nodes that were created without
        nodes mapping to them.
    dict
        A dict of reference atoms, mapping `graph_out` nodes to nodes in
        `molecule`.
    """
    mol_to_block, blocks_to, references = match
    if graph_out.nrexcl is None:
        graph_out.nrexcl = blocks_to.nrexcl
    try:
        # merge_molecule will return a dict mapping the node keys of the
        # added block to the ones in graph_out
        # FIXME: Issue #154 lives here.
        block_to_out = graph_out.merge_molecule(blocks_to)
    except ValueError:
        # This probably means the nrexcl of the block is different from the
        # others. This means the user messed up their data. Or there are
        # different forcefields in the same forcefield folder...
        LOGGER.exception('Residue(s) {} is not compatible with the others',
                         set(nx.get_node_attributes(blocks_to, 'resname').values()),
                         type='inconsistent-data')
        raise
    # overlap does not have to be a dict, since the values in block_to_out are
    # guaranteed to be unique in graph_out. So we can look them up in
    # mol_to_out
    overlap = set(mol_to_out.keys()) & set(mol_to_block.keys())
    for mol_idx in mol_to_block:
        for block_idx, weight in mol_to_block[mol_idx].items():
            out_idx = block_to_out[block_idx]
            mol_to_out[mol_idx][out_idx] = weight
            out_to_mol[out_idx][mol_idx] = weight

    none_to_one_mappings = set()
    mapped_block_idxs = {block_idx
                         for mol_idx in mol_to_block
                         for block_idx in mol_to_block[mol_idx]}
    for spawned in set(blocks_to.nodes) - mapped_block_idxs:
        # These nodes come from "nowhere", so, let's pretend they come from
        # all nodes in the block. This helps with setting attributes such
        # as 'chain'
        # "None to one" mapping - this is fine. This happens with e.g.
        # charge dummies.
        spawned = block_to_out[spawned]
        none_to_one_mappings.add(spawned)
        for mol_idx in mol_to_block:
            mol_to_out[mol_idx][spawned] = 0
            out_to_mol[spawned][mol_idx] = 0

    new_references = {block_to_out[mod_idx]: mol_idx for mod_idx, mol_idx in references.items()}
    return overlap, none_to_one_mappings, new_references


def apply_mod_mapping(match, molecule, graph_out, mol_to_out, out_to_mol):
    """
    Performs the mapping operation for a modification.

    Parameters
    ----------
    match
    molecule: networkx.Graph
        The original molecule
    graph_out: vermouth.molecule.Molecule
        The newly created graph that describes `molecule` at a different
        resolution.
    mol_to_out: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `molecule` to nodes in `graph_out` with the
        associated weights.
    out_to_mol: dict[collections.abc.Hashable, dict[collections.abc.Hashable, float]]
        A dict mapping nodes in `graph_out` to nodes in `molecule` with the
        associated weights.

    Returns
    -------
    dict[str, dict[tuple, vermouth.molecule.Link]]
        A dict of all modifications that have been applied by this modification
        mapping operations. Maps interaction type to involved atoms to the
        modification responsible.
    dict
        A dict of reference atoms, mapping `graph_out` nodes to nodes in
        `molecule`.
    """
    mol_to_mod, modification, references = match
    LOGGER.info('Applying modification mapping {}', modification.name, type='general')
    mod_to_mol = defaultdict(dict)
    for mol_idx, mod_idxs in mol_to_mod.items():
        for mod_idx in mod_idxs:
            mod_to_mol[mod_idx][mol_idx] = mol_to_mod[mol_idx][mod_idx]
    mod_to_mol = dict(mod_to_mol)
    mod_to_out = {}
    # Some nodes of modification will already exist. The question is
    # which, and which index they have in graph_out.
    for mod_idx in modification:
        # FIXME: Bad way of detecting whether the node should already
        #        exist!
        if modification.nodes[mod_idx].get('PTM_atom', False):
            # Node does not exist yet.
            if not graph_out.nodes:
                out_idx = 0
            else:
                out_idx = max(graph_out) + 1
            mod_to_out[mod_idx] = out_idx
            graph_out.add_node(out_idx, **modification.nodes[mod_idx])
        else:
            # Node should already exist
            # We need to find the out_index of this node. Since the
            # node already exists, there is at least one mol_idx in
            # mol_to_out that refers to the correct out_idx. What we do
            # is try to find those mol indices by looking at
            # mod_to_mol.
            # Find the other mol nodes that map to this bead according
            # to the mod mapping...
            mol_idxs = mod_to_mol[mod_idx]
            # ...and make the node with the correct attributes.
            out_idxs = set()
            for mol_idx in mol_idxs:
                out_idxs.update(mol_to_out.get(mol_idx, {}))
            for out_idx in out_idxs:
                out_node = graph_out.nodes[out_idx]
                if modification.nodes[mod_idx]['atomname'] == out_node['atomname']:
                    break
            else:  # No break, so no matching node found
                raise ValueError("No node found in molecule with "
                                 "atomname {}".format(modification.nodes[mod_idx]['atomname']))
            mod_to_out[mod_idx] = out_idx
            # FIXME: modify out_node as required. We will lose access to
            #        "modification" after this.
    for mol_idx in mol_to_mod:
        for mod_idx, weight in mol_to_mod[mol_idx].items():
            out_idx = mod_to_out[mod_idx]
            if mol_idx not in mol_to_out:
                mol_to_out[mol_idx] = {}
            mol_to_out[mol_idx][out_idx] = weight
            if out_idx not in out_to_mol:
                out_to_mol[out_idx] = {}
            out_to_mol[out_idx][mol_idx] = weight

    new_references = {mod_to_out[mod_idx]: mol_idx for mod_idx, mol_idx in references.items()}

    # Apply interactions
    applied_interactions = defaultdict(dict)
    for interaction_type, interactions in modification.interactions.items():
        for interaction in interactions:
            atoms = [mod_to_mol[mod_idx] for mod_idx in interaction.atoms]
            interaction = interaction._replace(atoms=atoms)
            applied_interactions[interaction_type][tuple(atoms)].append(modification)
            graph_out.add_interaction(interaction_type, **interaction._asdict())
    return dict(applied_interactions), new_references


def attrs_from_node(node, attrs):
    """
    Helper function that applies a "replace" operations on the node if
    required, and then returns a dict of the attributes listed in `attrs`.

    Parameters
    ----------
    node: dict
    attrs: collections.abc.Container
        Attributes that should be in the output.
    Returns
    -------
    dict
    """
    if 'replace' in node:
        node = node.copy().update(node['replace'])
    return {attr: val for attr, val in node.items() if attr in attrs}


def do_mapping(molecule, mappings, to_ff, attribute_keep=()):
    """
    Creates a new :class:`~vermouth.molecule.Molecule` in force field `to_ff`
    from `molecule`, based on `mappings`. It does this by doing a subgraph
    isomorphism of all blocks in `mappings` and `molecule`. Will issue warnings
    if there's atoms not contibuting to the new molecule, or if there's
    overlapping blocks.
    Node attributes in the new molecule will come from the blocks constructing
    it, except for those in `attribute_keep`, which lists the attributes that
    will be kept from `molecule`.

    Parameters
    ----------
    molecule: :class:`~vermouth.molecule.Molecule`
        The molecule to transform.
    mappings: dict[str, dict[str, dict[str, tuple]]]
        ``{ff_name: {ff_name: {block_name: (mapping, weights, extra)}}}``
        A collection of mappings, as returned by e.g.
        :func:`~vermouth.map_input.read_mapping_directory`.
    to_ff: :class:`~vermouth.forcefield.ForceField`
        The force field to transform to.
    attribute_keep: :class:`~collections.abc.Iterable`
        The attributes to keep from `molecule`

    Returns
    -------
    :class:`~vermouth.molecule.Molecule`
        A new molecule, created by transforming `molecule` to `to_ff` according
        to `mappings`.
    """
    attribute_keep = tuple(attribute_keep)
    attribute_must = tuple(attribute_must)
    # Transfering the meta maybe should be a copy, or a deep copy...
    # If it breaks we look at this line.
    graph_out = Molecule(force_field=to_ff, meta=molecule.meta)
    # We want to keep the 'chain' property from the original molecule.
    attribute_keep = ['chain'] + list(attribute_keep)
    mappings = build_graph_mapping_collection(molecule.force_field, to_ff, mappings)
    all_matches = []
    for mapping in mappings:
        all_matches.extend(mapping.map(molecule, node_match=_old_atomname_match,
                                       edge_match=edge_matcher))

    # There are a few separate mapping cases to be considered:
    # One to one mapping - e.g. AA to AA, the simplest case
    # Many to one mapping - e.g. AA to CG without sharing atoms between beads
    # Many to many mapping - e.g. AA to CG *with* sharing atoms between bonds
    # These three cases are covered by the normal operation, the following are
    # caught with some additional logic
    # None to one - whole block taken as origin, with weights 0
    # One to none - unmapped atoms (produces a warning)
    # One to many - e.g. CG to AA. This mostly works, but we don't know how to
    #               make sure the "many" should be connected togeher. Gives a
    #               warning if it's disconnected.

    mol_to_out = defaultdict(dict)
    out_to_mol = defaultdict(dict)
    overlapping_mappings = set()
    none_to_one_mappings = set()
    all_references = {}
    # Sort by lowest node key per residue. We need to do this, since
    # merge_molecule creates new resid's in order.
    for mol_to_block, blocks_to, references in sorted(all_matches, key=lambda x: min(x[0].keys())):
        if graph_out.nrexcl is None:
            graph_out.nrexcl = blocks_to.nrexcl
        try:
            # merge_molecule will return a dict mapping the node keys of the
            # added block to the ones in graph_out
            # FIXME: Issue #154 lives here.
            block_to_out = graph_out.merge_molecule(blocks_to)
        except ValueError:
            # This probably means the nrexcl of the block is different from the
            # others. This means the user messed up their data. Or there are
            # different forcefields in the same forcefield folder...
            LOGGER.exception('Residue(s) {} is not compatible with the others',
                             set(nx.get_node_attributes(blocks_to, 'resname').values()),
                             type='inconsistent-data')
            raise
        # overlapping_mappings does not have to be a dict, since the values in
        # block_to_out are guaranteed to be unique in graph_out. So we can look
        # them up in mol_to_out
        overlapping_mappings.update(set(mol_to_block.keys()) & set(mol_to_out.keys()))
        for mol_idx in mol_to_block:
            for block_idx, weight in mol_to_block[mol_idx].items():
                out_idx = block_to_out[block_idx]
                mol_to_out[mol_idx][out_idx] = weight
                out_to_mol[out_idx][mol_idx] = weight

        mapped_block_idxs = {idx for mol_idx in mol_to_block for idx in mol_to_block[mol_idx]}
        for spawned in set(blocks_to.nodes) - mapped_block_idxs:
            # These nodes come from "nowhere", so, let's pretend they come from
            # all nodes in the block. This helps with setting attributes such
            # as 'chain'
            # "None to one" mapping - this is fine. This happens with e.g.
            # charge dummies.
            spawned = block_to_out[spawned]
            none_to_one_mappings.add(spawned)
            for mol_idx in mol_to_block:
                mol_to_out[mol_idx][spawned] = 0
                out_to_mol[spawned][mol_idx] = 0
        all_references.update(references)
    out_to_mol = dict(out_to_mol)
    mol_to_out = dict(mol_to_out)

    # Set node attributes based on what the original atoms are.
    # TODO: reference atoms.
    for out_idx in out_to_mol:
        mol_idxs = out_to_mol[out_idx].keys()
        # Keep track of what bead comes from where
        subgraph = molecule.subgraph(mol_idxs)
        graph_out.nodes[out_idx]['graph'] = subgraph
        weights = out_to_mol[out_idx]
        graph_out.nodes[out_idx]['mapping_weights'] = weights
        if out_idx in all_references:
            ref_idx = all_references[out_idx]
            for attr, vals in molecule.nodes[ref_idx]:
                if attr in attribute_keep:
                    graph_out[out_idx][attr] = molecule.nodes[ref_idx][attr]
        else:
            # We drop the node keys, since those are not super relevant. We are
            # just interested in values of the node attributes, and whether
            # they're all equal.
            attrs = {name: list(nx.get_node_attributes(subgraph, name).values())
                     for name in attribute_keep}
            for attr, vals in attrs.items():
                if not are_all_equal(vals):
                    LOGGER.warning('The attribute {} for atom {} is going to'
                                   ' be garbage.', attr, format_atom_string(graph_out.nodes[out_idx]),
                                   type='inconsistent-data')
                if vals:
                    graph_out.nodes[out_idx][attr] = vals[0]
                else:
                    # No nodes hat the attribute `name`. And
                    # nx.get_node_attributes doesn't take a default.
                    graph_out.nodes[out_idx][attr] = None

    mod_matches = map_modifications(molecule, graph_out, mol_to_out, out_to_mol, mappings)
    all_matches.extend(mod_matches)

    # We need to add edges between residues. Within residues comes from the
    # blocks.
    for match1, match2 in combinations(all_matches, 2):
        match1 = match1[0]
        match2 = match2[0]
        edges = molecule.edges_between(match1.keys(), match2.keys())
        # TODO: Backmapping needs love here
        for mol_idx, mol_jdx in edges:
            # Substract none_to_one_mappings, since those should not be made to
            # connect to things automatically.
            out_idxs = mol_to_out[mol_idx].keys() - none_to_one_mappings
            out_jdxs = mol_to_out[mol_jdx].keys() - none_to_one_mappings
            for out_idx, out_jdx in product(out_idxs, out_jdxs):
                if out_idx != out_jdx:
                    graph_out.add_edge(out_idx, out_jdx)


    # Sanity check the results
    # "Many to one" mapping - overlapping blocks means dubious node properties
    if overlapping_mappings:
        LOGGER.warning('These atoms are covered by multiple blocks. This is a '
                       'bad idea: {}. This probably means the following output'
                       ' particles are wrong: {}.',
                       {format_atom_string(molecule.nodes[mol_idx])
                        for mol_idx in overlapping_mappings},
                       {format_atom_string(graph_out.nodes[out_idx], atomid='')
                        for mol_idx in overlapping_mappings
                        for out_idx in mol_to_out[mol_idx]},
                       type='inconsistent-data')

    # "One to many" mapping - not necessarrily a problem, unless it leads to
    # missing edges
    for mol_idx in mol_to_out:
        # Substract the none to one mapped nodes, since those don't contribute
        # and make false positives.
        out_idxs = mol_to_out[mol_idx].keys() - none_to_one_mappings
        if len(out_idxs) > 1 and not nx.is_connected(graph_out.subgraph(out_idxs)):
            # In this case there's a single input particle mapping to multiple
            # output particles. This probably means there's bonds missing
            LOGGER.warning('The input particle {} maps to multiple output '
                           'particles: {}, which are disconnected. There are '
                           'probably edges missing.',
                           format_atom_string(molecule.nodes[mol_idx]),
                           {format_atom_string(graph_out.nodes[out_idx], atomid='')
                            for out_idx in out_idxs},
                           type='inconsistent-data')

    # "One to none" mapping - this means your mapping files are incomplete
    uncovered_atoms = set(molecule.nodes.keys()) - set(mol_to_out.keys())
    if uncovered_atoms:
        uncovered_hydrogens = {idx for idx in uncovered_atoms
                               if molecule.nodes[idx].get('element', '') == 'H'}
        if uncovered_hydrogens:
            # Maybe this should be info?
            LOGGER.debug('These hydrogen atoms are not covered by a mapping.'
                         ' This is not the best idea. {}',
                         [format_atom_string(molecule.nodes[idx])
                          for idx in uncovered_hydrogens],
                         type='unmapped-atom'
                         )
        other_uncovered = uncovered_atoms - uncovered_hydrogens
        if other_uncovered:
            LOGGER.warning("These atoms are not covered by a mapping. Either"
                           " your mappings don't describe all atoms (bad idea),"
                           " or, there's no mapping available for all residues."
                           " {}",
                           [format_atom_string(molecule.nodes[idx])
                            for idx in other_uncovered],
                           type='unmapped-atom')
    return graph_out


class DoMapping(Processor):
    def __init__(self, mappings, to_ff, delete_unknown=False, attribute_keep=()):
        self.mappings = mappings
        self.to_ff = to_ff
        self.delete_unknown = delete_unknown
        self.attribute_keep = attribute_keep
        super().__init__()

    def run_molecule(self, molecule):
        return do_mapping(
            molecule,
            mappings=self.mappings,
            to_ff=self.to_ff,
            attribute_keep=self.attribute_keep
        )

    def run_system(self, system):
        mols = []
        for molecule in system.molecules:
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    raise
                    # TODO: raise a loud warning here
            else:
                if new_molecule:
                    mols.append(new_molecule)
        system.molecules = mols
        system.force_field = self.to_ff
