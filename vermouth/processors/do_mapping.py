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
    # FIXME!
    return parse_mapping_file('jon.map')


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


def get_mod_mappings(from_ff):
    """
    Returns a dict of all known modification mappings.
    """
    # FIXME!
    mappings = parse_mapping_file('modifications.map')
    out = {}
    for mapping in mappings:
        if mapping.type == 'modification':
            out[mapping.names] = mapping
    return out


def map_modifications(molecule, graph_out, mol_to_out, out_to_mol):
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
    known_mod_mappings = get_mod_mappings(molecule.force_field)
    for group in found_ptm_groups:
        # known_mod_mappings is a dict[tuple[str], Mapping]. We want to know
        # the minimal combination of those needed to cover all the PTMs found
        # in group. The cheapest solution is covering the names of the PTMs in
        # group with keys from known_mod_mappings. An improvement would be to
        # do the graph covering again.
        # TODO?
        covered_by = cover([ptm.name for ptm in group],
                           sorted(known_mod_mappings, key=len, reverse=True))
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

            mod_to_out = {}
            # Some nodes of modification will already exist. The question is
            # which, and which index they have in graph_out.
            for mod_idx in modification:
                # FIXME: Bad way of detecting whether the node should already
                #        exist!
                if modification.nodes[mod_idx].get('PTM_atom', False):
                    # New node
                    out_idx = max(graph_out) + 1
                    mod_to_out[mod_idx] = out_idx
                    graph_out.add_node(out_idx, **modification.nodes[mod_idx])
                    if mod_idx in references:
                        ref = references[mod_idx]
                        for attr in molecule.nodes[ref]:
                            if attr not in graph_out.nodes[out_idx]:
                                graph_out.nodes[out_idx][attr] = molecule.nodes[ref][attr]
                    else:
                        print("Don't know where to get attributes for {}"
                              "".format(graph_out.nodes[out_idx]))
                else:
                    # Node should already exists.
                    # Find the other mol nodes that map to this bead according
                    # to the mod mapping...
                    mol_idxs = mod_to_mol[mod_idx]
                    # ...and make sure those all map to the same output node.
                    out_idxs = set()
                    for mol_idx in mol_idxs:
                        out_idxs.update(mol_to_out.get(mol_idx, {}))
                    assert len(out_idxs) == 1
                    out_idx = next(iter(out_idxs))
                    mod_to_out[mod_idx] = out_idx
                    if 'replace' in modification.nodes[mod_idx]:
                        if out_idx in touched_atoms:
                            # TODO: better message
                            LOGGER.warning('Multiple modifications changed atom', type='inconsistent-data')
                        remove_atom = modification.nodes[mod_idx]['replace'].get('atomname', False) is None
                        if remove_atom:
                            to_remove.add(out_idx)
                        else:
                            graph_out.nodes[out_idx].update(modification.nodes[mod_idx]['replace'])
                        touched_atoms[out_idx].append(modification)
                out_idx = mod_to_out[mod_idx]
                graph_out.nodes[out_idx]['mapping_weights'] = mod_to_mol[mod_idx]
                graph_out.nodes[out_idx]['graph'] = molecule.subgraph(mod_to_mol[mod_idx].keys())
            for mol_idx in mol_to_mod:
                for mod_idx, weight in mol_to_mod[mol_idx].items():
                    out_idx = mod_to_out[mod_idx]
                    if mol_idx not in mol_to_out:
                        mol_to_out[mol_idx] = {}
                    mol_to_out[mol_idx][out_idx] = weight
                    if out_idx not in out_to_mol:
                        out_to_mol[out_idx] = {}
                    out_to_mol[out_idx][mol_idx] = weight

            # Apply interactions
            for interaction_type, interactions in modification.interactions.items():
                for interaction in interactions:
                    atoms = [mod_to_mol[mod_idx] for mod_idx in interaction.atoms]
                    interaction = interaction._replace(atoms=atoms)
                    applied_interactions[interaction_type][atoms].append(modification)
                    graph_out.add_interaction(interaction_type, **interaction._asdict())
            # Done applying this modification to all possible positions
        # Done applying all modifications of this type
    graph_out.remove_nodes_from(to_remove)
    for interaction_type in applied_interactions:
        for atoms, modifications in applied_interactions[interaction_type].items():
            if len(modifications) != 1:
                # TODO: better message
                LOGGER.warning('Interaction set by multiple modification '
                               'mappings', type='inconsistent-data')

    return all_matches


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

    mod_matches = map_modifications(molecule, graph_out, mol_to_out, out_to_mol)
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
