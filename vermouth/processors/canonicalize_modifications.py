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
Provides a Processor that identifies unexpected atoms such as PTMs and
protonations, and canonicalizes their attributes based on modifications known
in the forcefield.
"""

from collections import defaultdict
import itertools

import networkx as nx

from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger
from ..utils import format_atom_string, are_all_equal

LOGGER = StyleAdapter(get_logger(__name__))


def ptm_node_matcher(node1, node2):
    """
    Returns True iff node1 and node2 should be considered equal. This means
    they are both either marked as PTM_atom, or not. If they both are PTM
    atoms, the elements need to match, and otherwise, the atom names must
    match.
    """
    if node1.get('PTM_atom', False) == node2.get('PTM_atom', False):
        if node2.get('PTM_atom', False):
            # elements must match
            return node1['element'] == node2['element']
        else:
            # atomnames must match
            return node1['atomname'] == node2['atomname']
    else:
        return False


def find_ptm_atoms(molecule):
    """
    Finds all atoms in molecule that have the node attribute ``PTM_atom`` set
    to a value that evaluates to ``True``. ``molecule`` will be traversed
    starting at these atoms until all marked atoms are visited such that they
    are identified per "branch", and for every branch the anchor node is known.
    The anchor node is the node(s) which are not PTM atoms and share an edge
    with the traversed branch.

    Parameters
    ----------
    molecule : networkx.Graph

    Returns
    -------
    list[tuple[set, set]]
        ``[({ptm atom indices}, {anchor indices}), ...]``. Ptm atom indices are
        connected, and are connected to the rest of molecule via anchor
        indices.
    """

    # Atom names have already been fixed, and missing atoms have been added.
    # In addition, unrecognized atoms have been labeled with the PTM attribute.
    extra_atoms = set(n_idx for n_idx in molecule
                      if (molecule.nodes[n_idx].get('PTM_atom', False)
                          or molecule.nodes[n_idx].get('modifications')))
    ptms = []
    while extra_atoms:
        # First PTM atom we'll look at
        orig = next(iter(extra_atoms))
        anchors = set()
        # PTM atoms we've found
        atoms = set()
        # Atoms we still need to see this traversal
        to_see = set()
        # Traverse in molecule.
        while True:
            if orig in extra_atoms and orig not in atoms:
                # If this is a PTM atom, we want to see it's neighbours as
                # well.
                to_see.update(molecule[orig].keys())
                atoms.add(orig)
            elif orig not in extra_atoms:
                # Else, it's an attachment point for the this PTM
                anchors.add(orig)
            if not to_see:
                # We've traversed the interesting bit of the tree
                break
            orig = to_see.pop()
        # Although we know how far our tree spans we may still have work to do
        # for terminal nodes. There has to be a more elegant solution though.
        # for node in to_see:
        #     if node in extra_atoms:
        #         atoms.add(node)
        #     else:
        #         anchors.add(node)
        extra_atoms -= atoms
        ptms.append((atoms, anchors))
    return ptms


def identify_ptms(residue, residue_ptms, known_ptms):
    """
    Identifies all PTMs in ``known_PTMs`` necessary to describe all PTM atoms in
    ``residue_ptms``. Will take PTMs such that all PTM atoms in ``residue``
    will be covered by applying PTMs from ``known_PTMs`` in order.
    Nodes in ``residue`` must have correct ``atomname`` attributes, and may not
    be missing. In addition, every PTM in must be anchored to a non-PTM atom.

    Parameters
    ----------
    residue : networkx.Graph
        The residues involved with these PTMs. Need not be connected.

    residue_ptms : list[tuple[set, set]]
        As returned by ``find_PTM_atoms``, but only those relevant for
        ``residue``.

    known_PTMs : collections.abc.Sequence[tuple[networkx.Graph, networkx.isomorphism.GraphMatcher]]
        The nodes in the graph must have the `PTM_atom` attribute (True or
        False). It should be True for atoms that are not part of the PTM
        itself, but describe where it is attached to the molecule.
        In addition, its nodes must have the `atomname` attribute, which will
        be used to recognize where the PTM  is anchored, or to correct the
        atom names. Lastly, the nodes may have a `replace` attribute, which
        is a dictionary of ``{attribute_name: new_value}`` pairs. The special
        case here is if attribute_name is ``'atomname'`` and new_value is
        ``None``: in this case the node will be removed.
        Lastly, the graph (not its nodes) needs a 'name' attribute.

    Returns
    -------
    list[tuple[networkx.Graph, dict]]
        All PTMs from ``known_PTMs`` needed to describe the PTM atoms in
        ``residue`` along with a ``dict`` of node correspondences. The order of
        ``known_PTMs`` is preserved.

    Raises
    ------
    KeyError
        Not all PTM atoms in ``residue`` can be covered with ``known_PTMs``.
    """
    to_cover = set()
    cover = []
    for res_ptm in residue_ptms:
        ptm_atoms, anchors = res_ptm
        # For every node in this residue, get all modifications already known.
        residue_mods = [residue.nodes[idx].get('modifications', []) for idx in ptm_atoms]
        # Deduplicate the modifications so we only check e.g. C-ter once for
        # this residue
        used_mods = []
        for node_mods in residue_mods:
            for mod in node_mods:
                if mod not in used_mods:
                    used_mods.append(mod)
        if used_mods:
            # Store all atoms matched with already known mods (in used_mods) in
            # known_matched, so we can subtract them from ptm_atoms afterwards.
            # This may cause issues with overlapping modifications. It may be
            # better to subtract match from ptm_atoms in the loop.
            known_matched = set()
            for mod in used_mods:
                gm = nx.isomorphism.GraphMatcher(residue.subgraph(ptm_atoms), mod,
                                                 node_match=nx.isomorphism.categorical_node_match('atomname', ''))
                match = list(gm.subgraph_isomorphisms_iter())
                assert len(match) == 1
                match = match[0]
                cover.append((mod, match))
                # (That would be here)
                known_matched.update(match)
            ptm_atoms -= known_matched
            assert not ptm_atoms
        else:
            to_cover.update(ptm_atoms)
            to_cover.update(anchors)
    cover += _cover_graph(residue, to_cover, known_ptms)
    return cover


def _cover_graph(graph, to_cover, fragments):
    # BASECASE: to_cover is empty
    if not to_cover:
        return []

    # All non-PTM atoms in residue are always available for matching...
    available = set(n_idx for n_idx in graph
                    if not graph.nodes[n_idx].get('PTM_atom', False))
    # ... and add those we still need to cover
    available.update(to_cover)

    # REDUCTION: Apply one of fragments, remove those atoms from to_cover
    # COMBINATION: add the applied option to the output.
    for idx, option in enumerate(fragments):
        graphlet, matcher = option
        matches = list(matcher.subgraph_isomorphisms_iter())
        # Matches: [{graph_idxs: fragment_idxs}, {...}, ...]
        for match in matches:
            matching = set(match.keys())
            # TODO: one of the matching atoms must be an anchor. Should be
            # handled by PTMGraphMatcher already, assuming every PTM graph has
            # at least one non-ptm atom specified
            if matching <= available:
                # Continue with the remaining ptm atoms, and try just this
                # option and all smaller.
                try:
                    rest_cover = _cover_graph(graph, to_cover - matching, fragments[idx:])
                except KeyError:
                    continue
                return [(graphlet, match)] + rest_cover
    raise KeyError('Could not identify PTM')


def allowed_ptms(residue, res_ptms, known_ptms):
    """
    Finds all PTMs in ``known_ptms`` which might be relevant for ``residue``.

    Parameters
    ----------
    residue : networkx.Graph

    res_ptms : list[tuple[set, set]]
        As returned by ``find_PTM_atoms``.
        Currently not used.

    known_ptms : collections.abc.Mapping[str, networkx.Graph]

    Yields
    ------
    tuple[networkx.Graph, networkx.isomorphism.GraphMatcher]
        All graphs in known_ptms which are subgraphs of residue.
    """
    # TODO: filter by element count first
    for ptm in known_ptms.values():
        ptm_graph_matcher = nx.isomorphism.GraphMatcher(residue, ptm, node_match=ptm_node_matcher)
        if ptm_graph_matcher.subgraph_is_isomorphic():
            yield ptm, ptm_graph_matcher


def fix_ptm(molecule):
    '''
    Canonizes all PTM atoms in molecule, and labels the relevant residues with
    which PTMs were recognized. Modifies ``molecule`` such that atom names of
    PTM atoms are corrected, and the relevant residues have been labeled with
    which PTMs were recognized.

    Parameters
    ----------
    molecule : networkx.Graph
        Must not have missing atoms, and atom names must be correct. Atoms which
        could not be recognized must be labeled with the attribute
        PTM_atom=True.
    '''
    ptm_atoms = find_ptm_atoms(molecule)

    def key_func(ptm_atoms):
        node_idxs = ptm_atoms[-1]  # The anchors
        return sorted(molecule.nodes[idx]['resid'] for idx in node_idxs)

    ptm_atoms = sorted(ptm_atoms, key=key_func)

    resid_to_idxs = defaultdict(list)
    for n_idx in molecule:
        residx = molecule.nodes[n_idx]['resid']
        resid_to_idxs[residx].append(n_idx)
    resid_to_idxs = dict(resid_to_idxs)

    # Keep track of all nodes that get removed due to unknown PTMs
    removed = set()

    known_ptms = molecule.force_field.modifications

    for resids, res_ptms in itertools.groupby(ptm_atoms, key_func):
        # How to solve this graph covering problem
        # Filter known_ptms, such that
        #   element_count(known_ptm) <= element_count(found)
        # Filter known_ptms, such that known_ptm <= found (subgraph of).
        #   Note that this does mean that the PTMs in the PDB *must* be
        #   complete. So no missing atoms.
        # Find all the exactly covering combinations.
        # Pick the best solution, such that the maximum size of the applied
        # PTMs is maximal. (3, 2) > (3, 1, 1) > (2, 2, 1)
            # Numbers are sizes of applied PTMs

        # The last two steps are combined by recursively trying the largest
        # option in identify_ptms

        res_ptms = list(res_ptms)
        n_idxs = set()
        for resid in resids:
            n_idxs.update(resid_to_idxs[resid])
        # TODO: Maybe use graph_utils.make_residue_graph? Or rewrite that
        #       function?
        residue = molecule.subgraph(n_idxs - removed)
        options = allowed_ptms(residue, res_ptms, known_ptms)
        options = sorted(options,
                         key=lambda opt: len([n for n in opt[0]
                                              if opt[0].nodes[n].get('PTM_atom', False)]),
                         reverse=True)
        try:
            identified = identify_ptms(residue, res_ptms, options)
        except KeyError:
            LOGGER.warning('Could not identify the modifications for'
                           ' residues {}, involving atoms {}',
                           ['{resname}{resid}'.format(**molecule.nodes[resid_to_idxs[resid][0]])
                            for resid in sorted(set(resids))],
                           ['{atomid}-{atomname}'.format(**molecule.nodes[idx])
                            for idxs in res_ptms for idx in idxs[0]],
                           type='unknown-input')
            for idxs in res_ptms:
                for idx in idxs[0]:
                    molecule.remove_node(idx)
                    removed.add(idx)
            continue

        # Why this mess? There can be multiple PTMs for a single (set of)
        # residue(s); and a single PTM can span multiple residues.
        LOGGER.info("Identified the modifications {} on residues {}",
                    [out[0].graph['name'] for out in identified],
                    ['{resname}{resid}'.format(**molecule.nodes[resid_to_idxs[resid][0]])
                     for resid in resids])
        for ptm, match in identified:
            ptm.match = match
            for mol_idx, ptm_idx in match.items():
                ptm_node = ptm.nodes[ptm_idx]
                mol_node = molecule.nodes[mol_idx]
                # Names of PTM atoms still need to be corrected, and for some
                # non PTM atoms attributes need to change.
                if ptm_node['PTM_atom']:
                    mol_node['graph'] = molecule.subgraph([mol_idx]).copy()
                    for attr in ptm_node:
                        # FIXME: This probably transfers too many attributes.
                        if attr not in ('PTM_atom', 'replace'):
                            mol_node[attr] = ptm_node[attr]
                if 'replace' in ptm_node:
                    to_replace = ptm_node['replace']
                    for attr_name, val in to_replace.items():
                        if attr_name == 'atomname':
                            mol_node['_old_atomname'] = mol_node['atomname']
                        # We can't remove nodes which get their atomname set to
                        # None here, since mapping would then break due to
                        # missing atoms. Instead, the mapping processor should
                        # make sure superfluous atoms do not get constructed in
                        # the output resolution.
                        if mol_node.get(attr_name) != val:
                            fmt = 'Changing attribute {} from {} to {} for atom {}'
                            LOGGER.debug(fmt, attr_name, mol_node.get(attr_name),
                                         val, format_atom_string(mol_node),
                                         type='change-atom')
                            mol_node[attr_name] = val
            for n_idx in n_idxs:
                node = molecule.nodes[n_idx]
                if not ('modification' in node and ptm in node.get('modifications', [])):
                    # These nodes already had the modification annotated.
                    # Also note that 'modification' != 'modifications'. Yes,
                    # this is an issue. No, I'm not fixing that.
                    node['modifications'] = node.get('modifications', [])
                    node['modifications'].append(ptm)


class CanonicalizeModifications(Processor):
    """
    Identifies all modifications in a molecule and corrects their atom names.

    See Also
    --------
    :func:`fix_ptm`
    """
    def run_molecule(self, molecule):
        fix_ptm(molecule)
        return molecule
