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
Provides a processor that repairs a graph based on a reference.
"""
import networkx as nx

from ..molecule import Block
from .processor import Processor
from ..graph_utils import *  # FIXME  # pylint: disable=unused-wildcard-import
from ..ismags import ISMAGS
from ..log_helpers import StyleAdapter, get_logger
from ..utils import format_atom_string, are_all_equal

LOGGER = StyleAdapter(get_logger(__name__))


def get_default(dictionary, attr, default):
    """
    Functions like :meth:`dict.get`, except that when `attr` is in `dictionary`
    and `dictionary[attr]` is `None`, it will return `default`.

    Parameters
    ----------
    dictionary: dict
    attr: collections.abc.Hashable
    default

    Returns
    -------
    object
        The value of `dictionary[attr]` if `attr` is in `dictionary` and
        `dictionary[attr]` is not None. `default otherwise.`
    """
    item = dictionary.get(attr, None)
    if item is None:
        item = default
    return item


def _node_equal(node1, node2):
    return node1.get('atomname') == node2.get('atomname')


def _patch_modification(block, modification):
    """
    Applies, in order, modifications to block and returns a new Block.
    Modifications are applied by overlaying the anchor of a modification with
    the block and adding the remaining nodes and edges.
    """
    anchor_idxs = set()
    for mod_idx in modification:
        if not modification.nodes[mod_idx]['PTM_atom']:
            anchor_idxs.add(mod_idx)

    anchor = nx.subgraph(modification, anchor_idxs)
    non_anchor_idxs = set(modification) - anchor_idxs
    non_anchor = nx.subgraph(modification, non_anchor_idxs)

    block = nx.convert_node_labels_to_integers(block)

    ismags = ISMAGS(block, anchor, node_match=_node_equal)
    anchor_block_to_mod = list(ismags.subgraph_isomorphisms_iter())
    if not anchor_block_to_mod:
        LOGGER.error("Modification {} doesn't fit on Block {}", modification.name,
                     block.name)
        raise ValueError("Cannot apply modification to block")
    # This probably can't happen, since atoms in Modifications /should/ have
    # atomnames. Which should be unique. So, nocover.
    elif len(anchor_block_to_mod) > 1:  # pragma: nocover
        LOGGER.error("Modification {} fits on Block {} in {} ways",
                     modification.name, block.name, len(anchor_block_to_mod))  # pragma: nocover
        raise ValueError("Cannot apply modification to block")  # pragma: nocover

    block_to_mod = anchor_block_to_mod[0]
    mod_to_block = {val: key for key, val in block_to_mod.items()}
    # Add the extra modification atoms to the mapping/match. It's important to
    # note that this is under the assumption that when nx.disjoint_union
    # relabels nodes in non_anchor to ints it keeps the same order.
    # The alternative to this assumption is to reimplement disjoint_union where
    # the mapping is also returned
    mod_to_block.update(dict(zip(non_anchor_idxs, range(len(block), len(block)+len(non_anchor_idxs)))))
    result = Block(nx.disjoint_union(block, non_anchor))

    # Mark the modified atoms with the specified modification, so canonicalize
    # modifications has an easier job
    for mod_idx in modification:
        idx = mod_to_block[mod_idx]
        node_mods = result.nodes[idx].get('modifications', [])
        if modification not in node_mods:
            result.nodes[idx]['modifications'] = node_mods + [modification]

    for mod_idx, mod_jdx in modification.edges_between(anchor_idxs, non_anchor_idxs):
        idx = mod_to_block[mod_idx]
        jdx = mod_to_block[mod_jdx]
        result.add_edge(idx, jdx)
    return result


def _get_reference_residue(residue, force_field):
    """
    Uses the 'mutation', 'modify' and 'resname' attributes of `residue` to find
    or generate the correct reference residue, based on `force_field`. Mutation
    takes priority over resname, and the corresponding block will be taken from
    the FF. Afterwards, all modifications in modify will be taken from the FF
    and applied.

    Returns the generated reference block.
    """
    if 'mutation' in residue:
        mutation = residue['mutation']
        if not are_all_equal(mutation):
            message = 'Can only mutate residue {}-{}{} once, {} mutations were requested.'
            LOGGER.error(message, residue['chain'], residue['resname'],
                         residue['resid'], len(mutation))
            raise ValueError(message.format(residue['chain'], residue['resname'],
                                            residue['resid'], len(mutation)))
        mutation = mutation[0]
        LOGGER.info('Mutating residue {}-{}{} to {}',
                    residue['chain'], residue['resname'], residue['resid'],
                    mutation)
        resname = mutation
    else:
        resname = residue['resname']
    reference_block = force_field.reference_graphs[resname]

    if 'modification' in residue:
        modifications = residue['modification']
        for mod_name in modifications:
            LOGGER.info('Applying modification {} to residue {}-{}{}',
                        mod_name, residue['chain'], resname, residue['resid'])
            if mod_name != 'none':
                mod = force_field.modifications[mod_name]
                reference_block = _patch_modification(reference_block, mod)
        for node_idx in reference_block:
            reference_block.nodes[node_idx]['modification'] = modifications
    if 'mutation' in residue:
        for node_idx in reference_block:
            reference_block.nodes[node_idx]['mutation'] = mutation

    return reference_block


def make_reference(mol):
    """
    Takes an molecule graph (e.g. as read from a PDB file), and finds and
    returns the graph how it should look like, including all matching nodes
    between the input graph and the references.
    Requires residue names to be correct.

    Notes
    -----
        The match between hydrogren atoms need not be perfect. See the
        documentation of ``isomorphism``.

    Parameters
    ----------
    mol : networkx.Graph
        The graph read from e.g. a PDB file. Required node attributes:

        :resname: The residue name.
        :resid: The residue id.
        :chain: The chain identifier.
        :element: The element.
        :atomname: The atomname.

    Returns
    -------
    networkx.Graph
        The constructed reference graph with the following node attributes:

        :resid: The residue id.
        :resname: The residue name.
        :chain: The chain identifier.
        :found: The residue subgraph from the PDB file.
        :reference: The residue subgraph used as reference.
        :match: A dictionary describing how the reference corresponds
            with the provided graph. Keys are node indices of the
            reference, values are node indices of the provided graph.
    """
    reference_graph = nx.Graph()
    residues = make_residue_graph(mol)
    symmetry_cache = {}
    LOGGER.debug('Making reference graph', type='step')
    for residx in residues:
        # TODO: make separate function for just one residue.
        # TODO: Merge degree 1 nodes (hydrogens!) with the parent node. And
        # check whether the node degrees match?

        resname = residues.nodes[residx]['resname']
        resid = residues.nodes[residx]['resid']
        chain = residues.nodes[residx]['chain']
        residue = residues.nodes[residx]['graph']
        reference = _get_reference_residue(residues.nodes[residx], mol.force_field)
        if 'mutation' in residues.nodes[residx]:
            # We need to do this, since we need to propagate all common
            # attributes in the input residue to any reconstructed atoms.
            # make_residue_graph already picks up on all the common attributes
            # and sets those as residue/node attributes, but any mutation needs
            # to be propagated as well, otherwise the newly constructed atoms
            # will end up with the old residue name. This way, we can just add
            # *all* attributes in the residue graph to the new atoms.
            resname = residues.nodes[residx]['mutation'][0]
            residues.nodes[residx]['resname'] = resname
        add_element_attr(reference)
        add_element_attr(residue)
        # We are going to sort the nodes of reference and residue by atomname.
        # We do this, because the ISMAGS algorithm prefers to match nodes with
        # lower IDs.
        # Get a \uFFFF for every node that doesn't have an atomname attribute
        # or when it's None, since that sorts higher than letters, giving them
        # the lowest priority in ISMAGS.

        res_names = {idx: get_default(residue.nodes[idx], 'atomname', '\uFFFF') for idx in residue}
        ref_names = {idx: get_default(reference.nodes[idx], 'atomname', '\uFFFF') for idx in reference}

        # Sort the nodes such that any atomnames that are common to both
        # reference and residue are first, and then the rest.
        # Also, sort it all by atomname. This is combined in one by sorting by
        # the tuple (not common, atomname). False < True.

        # If we want to relabel the nodes in-place we need to find new
        # non-overlapping labels. The easiest way of doing this is by turning
        # them into tuples. But this makes everything slow; probably because
        # ISMAGS does quite a lot of inequality comparisons, and those are way
        # faster for str/int. So, sacrifice the memory, and relabel by making a
        # new copy.

        # TODO: include a geometric alignment in the sorting. Humans are really
        #       good at solving isomorphism problems iff graphs look alike. We
        #       can do a similar trick here by rot+trans aligning the given
        #       residue with a reference conformation. And then sort by
        #       distance as third criterion
        new_residue_names = {old: new for new, old in enumerate(sorted(
            residue,
            key=lambda jdx: (res_names[jdx] not in ref_names.values(), res_names[jdx])  # pylint: disable=cell-var-from-loop
        ))}
        new_reference_names = {old: new for new, old in enumerate(sorted(
            reference,
            key=lambda jdx: (ref_names[jdx] not in res_names.values(), ref_names[jdx])  # pylint: disable=cell-var-from-loop
        ))}

        old_res_names = {v: k for k, v in new_residue_names.items()}
        old_ref_names = {v: k for k, v in new_reference_names.items()}

        # It would be nice if we were able to relabel them in-place, but it
        # seems to make everything slower. See above.
        res_copy = nx.relabel_nodes(residue, new_residue_names, copy=True)
        ref_copy = nx.relabel_nodes(reference, new_reference_names, copy=True)

        LOGGER.debug('Matching residue {}-{}{} to its reference', chain, resname, resid, type='step')

        # If we assume residue > reference the tests run *way* faster, but the
        # actual program becomes *much* *much* slower.
        # TODO: swap ref_copy and res_copy so that the smaller graph is the
        #       "subgraph"?
        ismags = ISMAGS(ref_copy, res_copy,
                        node_match=nx.isomorphism.categorical_node_match('element', None),
                        cache=symmetry_cache)
        # Finding the largest common subgraph is expensive, but the first step
        # is to try and find a subgraph isomorphism between
        # residue <= reference, so best case it makes no difference, and worst
        # case we avoid trying to find that isomorphism twice.
        match_iter = ismags.largest_common_subgraph()
        try:
            # We take only the first found match, since because the nodes are
            # sorted by atomname, and ISMAGS prefers to take nodes with low ID,
            # that match should have most matching atomnames.
            match = next(match_iter)
        except StopIteration:
            LOGGER.error("Can't find isomorphism between {}-{}{} and its "
                         "reference.", chain, resname, resid, type='inconsistent-data')
            continue
        # TODO: Since we only have one isomorphism we don't know whether the
        # assigment we're making is ambiguous. So iff the residue is small
        # enough (or a flag is set, whatever), also find the second isomorphism
        # and check whether it has the same number of correct atomnames. If so,
        # issue a warning and carry on. We can't do this for all residues,
        # since that takes a cup of coffee.

        # "unsort" the matches
        match = {old_ref_names[ref]: old_res_names[res] for ref, res in match.items()}

        # The residue graph has attributes which are common to all atoms in the
        # input residue, so propagate those to new atoms. These attributes are
        # things like residue name (see above in case of mutations), resid,
        # insertion code, etc.
        reference_graph.add_node(residx, reference=reference, found=residue,
                                 match=match, **residues.nodes[residx])
    reference_graph.add_edges_from(residues.edges())
    return reference_graph


def repair_residue(molecule, ref_residue, include_graph):
    """
    Rebuild missing atoms and canonicalize atomnames
    """
    # Rebuild missing atoms and canonicalize atomnames
    missing = []
    # Step 1: find all missing atoms. Canonicalize names while we're at it.
    reference = ref_residue['reference']
    found = ref_residue['found']
    match = ref_residue['match']

    resid = ref_residue['resid']
    resname = ref_residue['resname']
    LOGGER.debug('Repairing residue {}{}', resname, resid, type='step')
    for ref_idx in reference:
        # We're only really interested in correcting the atomname. Obscure
        # usecases may utilize e.g. charge, mass, atype. Either way, we need
        # to remove the resid. Resname shouldn't matter since that should
        # already be correct.
        ref_node = reference.nodes[ref_idx].copy()
        if 'resid' in ref_node:
            del ref_node['resid']

        if ref_idx in match:
            res_idx = match[ref_idx]
            node = molecule.nodes[res_idx]
            if include_graph:
                node['graph'] = molecule.subgraph([res_idx])
            node.update(ref_node)
            # Update found as well to keep found and molecule in line. It would
            # be better to try and figure why found is not a reference, but meh
            found.nodes[res_idx].update(ref_node)
        else:
            message = 'Missing atom {}{}:{}'
            args = (resname, resid, reference.nodes[ref_idx]['atomname'])
            if reference.nodes[ref_idx]['element'] != 'H':
                LOGGER.info(message, *args, type='missing-atom')
            else:
                # These are logged *below* debug level. Otherwise your screen
                # fills up pretty fast.
                LOGGER.log(5, message, *args, type='missing-atom')
            missing.append(ref_idx)
    # Step 2: try to add all missing atoms one by one. As long as we added
    # *something* the situation changed, and we might be able to place another.
    # We can only place atoms for which we know a neighbour.
    added = True
    while missing and added:
        added = False
        for ref_idx in missing:
            # See if the atom we want to add has a known neighbour. Otherwise,
            # continue to the next.
            if all(ref_neighbour in missing for ref_neighbour in reference[ref_idx]):
                continue
            added = True
            missing.pop(missing.index(ref_idx))
            # We don't find the lowest available number since that's just
            # asking for problems where you find an atom you don't expect
            # because the old one you were looking for was removed, and it's
            # number was reassigned.
            res_idx = max(molecule) + 1

            # Create the new node
            node = {}
            for key, val in ref_residue.items():
                # Some attributes are only relevant on a residue level, not on
                # an atom level.
                if key not in ('match', 'found', 'reference', 'nnodes',
                               'nedges', 'density'):
                    node[key] = val
            ref_node = reference.nodes[ref_idx].copy()
            if 'resid' in ref_node:
                del ref_node['resid']
            node.update(ref_node)
            node['atomid'] = res_idx + 1

            match[ref_idx] = res_idx
            molecule.add_node(res_idx, **node)
            found.add_node(res_idx, **node)

            message = "Adding {}"
            args = format_atom_string(node)
            if node['element'] != 'H':
                LOGGER.debug(message, args, type='missing-atom')
            else:
                # These are logged *below* debug level. Otherwise your screen
                # fills up pretty fast.
                LOGGER.log(5, message, args, type='missing-atom')

            neighbours = 0
            for neighbour_ref_idx in reference[ref_idx]:
                try:
                    neighbour_res_idx = match[neighbour_ref_idx]
                except KeyError:
                    continue
                if not molecule.has_edge(neighbour_res_idx, res_idx):
                    molecule.add_edge(neighbour_res_idx, res_idx)
                    neighbours += 1
            assert neighbours != 0

    for ref_idx in missing:
        # TODO: use utils.format_atom_string
        LOGGER.error('Could not reconstruct atom {}{}:{}',
                     reference.nodes[ref_idx]['resname'],
                     resid,
                     reference.nodes[ref_idx]['atomname'],
                     type='missing-atom')


def repair_graph(molecule, reference_graph, include_graph=True):
    """
    Repairs a molecule graph produced based on the information in
    ``reference_graph``. Missing atoms will be added and atom- and residue-
    names will be canonicalized. Atoms not present in ``reference_graph`` will
    have the attribute ``PTM_atom`` set to ``True``.

    ``molecule`` is modified in place. Missing atoms (as per ``reference_graph``)
    are added, atom and residue names are canonicalized, and PTM atoms are
    marked.

    If ``include_graph`` is ``True``, then the subgraph corresponding to each
    node is included in the node under the "graph" attribute.

    Parameters
    ----------
    molecule : molecule.Molecule
        The graph read from e.g. a PDB file. Required node attributes:

        :resname: The residue name.
        :resid: The residue id.
        :element: The element.
        :atomname: The atomname.

    reference_graph : networkx.Graph
        The reference graph as produced by :func:`make_reference`. Required node
        attributes:

        :resid: The residue id.
        :resname: The residue name.
        :found: The residue subgraph from the PDB file.
        :reference: The residue subgraph used as reference.
        :match: A dictionary describing how the reference corresponds
            with the provided graph. Keys are node indices of the
            reference, values are node indices of the provided graph.

    include_graph: bool
        Include the subgraph in the nodes.
    """
    for residx in reference_graph:
        residue = reference_graph.nodes[residx]
        repair_residue(molecule, residue, include_graph=include_graph)
        # Atomnames are canonized, and missing atoms added
        found = reference_graph.nodes[residx]['found']
        match = reference_graph.nodes[residx]['match']

        # Find the PTMs (or termini, or other additions) for *this* residue
        # `extra` is a set of the indices of the nodes from  `found` that have
        # no match in the reference graph.
        # `atachments` is a set of the nodes from `found` that have a match in
        # the reference and are connected to a node from `extra`.
        # We just stick a label on them for now, these are used by the PTM
        # processor.
        extra = set(found.nodes) - set(match.values())
        for idx in extra:
            molecule.nodes[idx]['PTM_atom'] = True
            found.nodes[idx]['PTM_atom'] = True
            if molecule.nodes[idx].get('mutation') or molecule.nodes[idx].get('modification'):
                molecule.remove_node(idx)

    return molecule


class RepairGraph(Processor):
    """
    Repairs a molecule such that it contains all atoms with appropriate atom
    names, as per the blocks in the system's force field, while taking any
    mutations and modification into account. These should be added as 'mutation'
    and 'modification' attributes to the atoms of the relevant residues.

    Attributes
    ----------
    delete_unknown: bool
        If True, removes any molecules that contain residues that are not known
        to the system's force field.
    include_graph: bool
        If True, every node in the resulting graph will have a 'graph' attribute
        containing a subgraph constructed using the input atoms.

    See Also
    --------
    :func:`repair_graph`
    """
    def __init__(self, delete_unknown=False, include_graph=True):
        super().__init__()
        self.delete_unknown = delete_unknown
        self.include_graph = include_graph

    def run_molecule(self, molecule):
        molecule = molecule.copy()
        reference_graph = make_reference(molecule)
        repair_graph(molecule, reference_graph, include_graph=self.include_graph)
        return molecule

    def run_system(self, system):
        mols = []
        for idx, molecule in enumerate(system.molecules):
            try:
                new_molecule = self.run_molecule(molecule)
            except KeyError as err:
                if not self.delete_unknown:
                    raise err
                else:
                    LOGGER.warning("Cannot recognize residue {} in  molecule {}. "
                                   "Deleting the molecule.",
                                   str(err), idx, type='unknown-residue')
            else:
                mols.append(new_molecule)
        system.molecules = mols
