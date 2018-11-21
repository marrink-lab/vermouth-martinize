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
from itertools import product, combinations

import networkx as nx

from ..molecule import Molecule
from .processor import Processor
from ..utils import are_all_equal, format_atom_string
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))


class GraphMapping:
    # Attributes to be removed from the blocks.
    forbidden = ['charge_group']

    # TODO: Add __getitem__, __iter__, __len__, __contains__, keys, values and
    #       items methods to emulate a Mapping?

    # TODO: Different methods of initializing the mapping? It might be nice to
    #       also provide the option to provide two molecules (instead) of lists
    #       of blocks and a mapping of {node_idx: [node_idx, ...], ...}
    def __init__(self, blocks_from, blocks_to, mapping, weights=None, extra=()):
        """
        blocks_from and blocks_to are sequences of Blocks.
        Mapping is a dictionary of {(residx, atomname): [(residx, atomname), ...], ...}.
        residx in these cases is the index of the residue in blocks_from and
        blocks_to respectively.
        """
        if weights is None:
            weights = {}
        self.block_from = self._merge(blocks_from)
        self.block_to = self._merge(blocks_to)
        self.mapping = defaultdict(set)
        self.weights = defaultdict(dict)
        # Translate atomnames in mapping to node keys.
        for from_, to in mapping.items():
            res_from, name_from = from_
            # +1 here (and below), because the blockidxs in mapping are start
            # at 0, while resids start at 1.
            from_idxs = list(self.block_from.find_atoms(atomname=name_from, resid=res_from+1))
            for res_to, name_to in to:
                to_idxs = self.block_to.find_atoms(atomname=name_to, resid=res_to+1)
                for to_idx in to_idxs:
                    self.mapping[to_idx].update(from_idxs)
                    if from_idxs:
                        self.weights[to_idx][from_idxs[0]] = (
                            weights[(res_to, name_to)][(res_from, name_from)]
                        )

        self.mapping = dict(self.mapping)
        self.extra = extra

        # We can't do this in _merge, since we need the resids to translate the
        # mapping from (ambiguous) atomnames to (unique) graph keys. We do have
        # to get rid of them, otherwise they overwrite the resids of the graph
        # we're mapping (in do_mapping).
        self._purge_forbidden(self.block_from)
        self._purge_forbidden(self.block_to)

        # Since we merged blocks, there may be edges missing in both (between
        # the provided blocks). This is bad. We should add that info to the
        # mapping, somehow.

        # One option is to add from eachother, but this tends to introduce
        # issues.
        # Example:
        #   from_blocks: a0-b0-c0 a1-b1-c1
        #   to_blocks:   A0-B0 C1-A1 B2-C2
        #   Mapping: {(0, A): [(0, a)], (0, B): [(0, b)], (1, C): [(0, c)],
        #             (1, A): [(1, a)], (2, B): [(1, b)], (2, C): [(1, c)]}
        # There are edges missing in both from_blocks and to_blocks (c0-a1, and
        # B0-C1 and A1-B2; respectively). There's enough information in the
        # "other" block to create those, so that's what we do.
#        for to_idx, to_jdx in self.block_to.edges():
#            try:
#                self.block_from.add_edges_from(product(self.mapping[to_idx],
#                                                       self.mapping[to_jdx]))
#            except KeyError:
#                # Either to_idx or to_jdx don't actually contribute to the
#                # mapping
#                pass
#            # e.g. for edge B0-C1 this is:
#            # add_edges_from(product([(0, b)], [(0, c)])); which is the same as
#            # add_edges_from(((0, b), (0, c)))
        # Cache the reverse map for a while. Maybe that means it shouldn't be
        # a property...
#        reverse_map = self.reverse_mapping
#        # This loop does the same as the one above, but in the other direction.
#        for from_idx, from_jdx in self.block_from.edges():
#            try:
#                self.block_to.add_edges_from(product(reverse_map[from_idx],
#                                                     reverse_map[from_jdx]))
#            except KeyError:
#                # Either from_idx or from_jdx don't actually contribute to the
#                # mapping
#                pass

    @classmethod
    def _merge(cls, blocks):
        out = blocks[0].to_molecule()
        for block in blocks[1:]:
            out.merge_molecule(block)
        return out

    @classmethod
    def _purge_forbidden(cls, block):
        for n_idx in block:
            node = block.nodes[n_idx]
            for attr in cls.forbidden:
                if attr in node:
                    del node[attr]

    @property
    def reverse_mapping(self):
        reverse_mapping = defaultdict(set)
        for to_idx, from_idxs in self.mapping.items():
            for from_idx in from_idxs:
                reverse_mapping[from_idx].add(to_idx)
        reverse_mapping = dict(reverse_mapping)
        return reverse_mapping


def build_graph_mapping_collection(from_ff, to_ff, mappings):
    graph_mapping_collection = {}
    pair_mapping = mappings[from_ff.name][to_ff.name]
    for name in from_ff.blocks.keys():
        if name in to_ff.blocks and name in pair_mapping:
            mapping, weights, extra = pair_mapping[name]
            graph_mapping_collection[name] = GraphMapping(
                [from_ff.blocks[name], ],
                [to_ff.blocks[name], ],
                mapping,
                weights,
                extra,
            )
    return graph_mapping_collection


# We can't inherit from nx.isomorphism.GraphMatcher to override
# `semantic_feasibility`. That implementation will clobber this one's method.
class MappingGraphMatcher(nx.isomorphism.isomorphvf2.GraphMatcher):
    def __init__(self, *args, edge_match=None, node_match=None, **kwargs):
        self.edge_match = edge_match
        self.node_match = node_match
        super().__init__(*args, **kwargs)
        self.G1_adj = self.G1.adj

    def semantic_feasibility(self, G1_node, G2_node):
        """
        Returns True if mapping G1_node to G2_node is semantically feasible.
        Adapted from networkx.algorithms.isomorphism.vf2userfunc._semantic_feasibility.
        """
        # Make sure the nodes match
        if self.node_match is not None:
            nm = self.node_match(self.G1.nodes[G1_node], self.G2.nodes[G2_node])
            if not nm:
                return False

        # Make sure the edges match
        if self.edge_match is not None:

            # Cached lookups
            core_1 = self.core_1
            edge_match = self.edge_match

            for neighbor in self.G1_adj[G1_node]:
                # G1_node is not in core_1, so we must handle R_self separately
                if neighbor == G1_node:
                    if not edge_match(G1_node, G1_node, G2_node, G2_node):
                        return False
                elif neighbor in core_1:
                    if not edge_match(G1_node, neighbor, G2_node, core_1[neighbor]):
                        return False
            # syntactic check has already verified that neighbors are symmetric
        return True


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
    # Transfering the meta meybe should be a copy, or a deep copy...
    # If it breaks we look at this line.
    graph_out = Molecule(force_field=to_ff, meta=molecule.meta)
    # We want to keep the 'chain' property from the original molecule.
    attribute_keep = ['chain'] + list(attribute_keep)
    pair_mapping = build_graph_mapping_collection(molecule.force_field, to_ff, mappings)
    all_matches = []
    for resname, mapping in pair_mapping.items():
        # TODO: add PTMs as a matching criterion here.
        # Make sure the atomname and resname match
        node_match = nx.isomorphism.categorical_node_match(['atomname', 'resname'], ['', ''])
        # And make sure that we don't accidentally cross a residue boundary,
        # unless that's allowed by the mapping.
        edge_match = partial(edge_matcher, molecule, mapping.block_from)
        # We're going to find *every* way block fits on molecule.
        graphmatcher = MappingGraphMatcher(molecule, mapping.block_from,
                                           node_match=node_match, edge_match=edge_match)
        matches = graphmatcher.subgraph_isomorphisms_iter()
        for match in matches:
            all_matches.append((match, resname, mapping))
    mol_to_out = defaultdict(list)
    blocks_per_atom = Counter()
    # Sort by lowest node key per residue. We need to do this, since
    # merge_molecule creates new resid's in order.
    for match, name, mapping in sorted(all_matches, key=lambda x: min(x[0].keys())):
        blocks_per_atom.update(match.keys())
        if graph_out.nrexcl is None:
            graph_out.nrexcl = mapping.block_to.nrexcl
        try:
            # merge_molecule will return a dict mapping the node keys of the
            # added block to the ones in graph_out
            block_to_out = graph_out.merge_molecule(mapping.block_to)
        except ValueError:
            # This probably means the nrexcl of the block is different from the
            # others. This means the user messed up their data. Or there are
            # different forcefields in the same forcefield folder...
            LOGGER.exception('Residue {} is not compatible with the others',
                             name, type='inconsistent-data')
            raise
        block_to_mol = {v: k for k, v in match.items()}
        for to_idx, from_idxs in mapping.mapping.items():
            # Some bookkeeping with indices.
            out_idx = block_to_out[to_idx]
            mol_idxs = [block_to_mol[from_idx] for from_idx in from_idxs]
            for mol_idx in mol_idxs:
                mol_to_out[mol_idx].append(out_idx)

            # Keep track of what bead comes from where
            subgraph = molecule.subgraph(mol_idxs)
            graph_out.nodes[out_idx]['graph'] = subgraph
            weights = {block_to_mol[from_idx]: mapping.weights[to_idx][from_idx]
                       for from_idx in from_idxs}
            graph_out.nodes[out_idx]['mapping_weights'] = weights
            # We drop the node keys, since those are not super relevant. We are
            # just interested in values of the node attributes, and whether
            # they're all equal.
            attrs = {name: list(nx.get_node_attributes(subgraph, name).values())
                     for name in attribute_keep}
            for attr, vals in attrs.items():
                if not are_all_equal(vals):
                    LOGGER.warning('The attribute {} for atom {} is going to'
                                   ' be garbage.', name, format_atom_string(graph_out.nodes[out_idx]),
                                   type='inconsistent-data')
                if vals:
                    graph_out.nodes[out_idx][attr] = vals[0]
                else:
                    # No nodes hat the attribute `name`. And
                    # nx.get_ndoe_attributes doesn't take a default.
                    graph_out.nodes[out_idx][attr] = None
    mol_to_out = dict(mol_to_out)
    # We need to add edges between residues. Within residues comes from the
    # blocks.
    # TODO: backmapping needs some magic here.
    for match1, match2 in combinations(all_matches, 2):
        match1 = match1[0]
        match2 = match2[0]
        edges = molecule.edges_between(match1.keys(), match2.keys())
        for mol_idx, mol_jdx in edges:
            out_idxs = mol_to_out[mol_idx]
            out_jdxs = mol_to_out[mol_jdx]
            for out_idx, out_jdx in product(out_idxs, out_jdxs):
                if out_idx != out_jdx:
                    graph_out.add_edge(out_idx, out_jdx)
        shared_atoms = set(match1.keys()) & set(match2.keys())
        shared_out_atoms = [mol_to_out[mol_idx] for mol_idx in shared_atoms]
        for in_atom, out_atoms in zip(shared_atoms, shared_out_atoms):
            if len(out_atoms) < 2:
                LOGGER.critical('The atom {} is shared between blocks, but'
                                ' only mapped once to {}?',
                                format_atom_string(molecule.nodes[in_atom]),
                                [format_atom_string(graph_out.nodes[idx])
                                 for idx in out_atoms],
                                type='inconsistent-data')
                raise ValueError("This atom is shared between blocks, but only"
                                 " mapped once?")
            for out_idx, out_jdx in combinations(out_atoms, 2):
                if out_idx != out_jdx:
                    graph_out.add_edge(out_idx, out_jdx)
        if shared_atoms:
            LOGGER.warning("You have the following atoms that are shared"
                           " between blocks. This may mean you have too many"
                           " particles in your output and/or erroneous bonds."
                           " {}. They've end up in the following output atoms:"
                           " {}.",
                           [format_atom_string(molecule.nodes[idx], atomid=idx) for idx in shared_atoms],
                           [format_atom_string(graph_out.nodes[idx], atomid=idx) for idxs in shared_out_atoms for idx in idxs],
                           type='inconsistent-data')

    # Sanity check the results
    if any(v > 1 for v in blocks_per_atom.values()):
        LOGGER.warning('These atoms are covered by multiple blocks. This is a '
                       'bad idea: {}', {format_atom_string(molecule.nodes[k], atomid=k): v
                                        for k, v in blocks_per_atom.items() if v > 1},
                       type='inconsistent-data')
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
