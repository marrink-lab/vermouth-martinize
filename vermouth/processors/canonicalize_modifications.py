#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .processor import Processor

from collections import defaultdict
import itertools

import networkx as nx


class PTMGraphMatcher(nx.isomorphism.GraphMatcher):
    # G1 >= G2; G1 is the found residue; G2 the PTM reference
    def semantic_feasibility(self, node1, node2):
        """
        Returns True iff node1 and node2 should be considered equal. This means
        they are both either marked as PTM_atom, or not. If they both are PTM
        atoms, the elements need to match, and otherwise, the atomnames must
        match.
        """
        node1 = self.G1.nodes[node1]
        node2 = self.G2.nodes[node2]
        if node1.get('PTM_atom', False) == node2['PTM_atom']:
            if node2['PTM_atom']:
                # elements must match
                return node1['element'] == node2['element']
            else:
                # atomnames must match
                return node1['atomname'] == node2['atomname']
        else:
            return False


def find_PTM_atoms(molecule):
    """
    Finds all atoms in molecule that have the node attribute ``PTM_atom`` set
    to a value that evaluates to ``True``. ``molecule`` will be traversed
    starting at these atoms untill all marked atoms are visited such that they
    are identified per "branch", and for every branch the anchor node is known.
    The anchor node is the node(s) which are not PTM atoms and share an edge
    with the traversed branch.

    Parameters
    ----------
    molecule : networkx.Graph

    Returns
    -------
    list of tuples of two sets of node indices
        ``[({ptm atom indices}, {anchor indices}), ...]``. Ptm atom indices are
        connected, and are connected to the rest of molecule via anchor
        indices.
    """

    # Atomnames have already been fixed, and missing atoms have been added.
    # In addition, unrecognized atoms have been labeled with the PTM attribute.
    extra_atoms = set(n_idx for n_idx in molecule
                      if molecule.nodes[n_idx].get('PTM_atom', False))
    PTMs = []
    while extra_atoms:
        # First PTM atom we'll look at
        first = next(iter(extra_atoms))
        anchors = set()
        # PTM atoms we've found
        atoms = set()
        # Atoms we still need to see this traversal
        to_see = set([first])
        # Traverse in molecule.
        for orig, succ in nx.bfs_successors(molecule, first):
            # We've seen orig, so remove it
            to_see.remove(orig)
            if orig in extra_atoms:
                # If this is a PTM atom, we want to see it's neighbours as
                # well.
                to_see.update(succ)
                atoms.add(orig)
            else:
                # Else, it's an attachment point for the this PTM
                anchors.add(orig)
            if not to_see:
                # We've traversed the interesting bit of the tree
                break
        extra_atoms -= atoms
        PTMs.append((atoms, anchors))
    return PTMs


def identify_ptms(residue, residue_ptms, known_PTMs):
    """
    Identifies all PTMs in ``known_PTMs`` nescessary to describe all PTM atoms in
    ``residue_ptms``. Will take PTMs such that all PTM atoms in ``residue``
    will be covered by applying PTMs from ``known_PTMs`` in order.
    Nodes in ``residue`` must have correct ``atomname`` attributes, and may not
    be missing. In addition, every PTM in must be anchored to a non-PTM atom.

    Parameters
    ----------
    residue : networkx.Graph
        The residues involved with these PTMs. Need not be connected.

    residue_ptms : list of tuples of two sets of node indices
        As returned by ``find_PTM_atoms``, but only those relevant for
        ``residue``.

    known_PTMs : sequence of tuples of (networkx.Graph, PTMGraphMatcher)
        The nodes in the graph must have the `PTM_atom` attribute (True or
        False). It should be True for atoms that are not part of the PTM
        itself, but describe where it is attached to the molecule.
        In addition, it's nodes must have the `atomname` attribute, which will
        be used to recognize where the PTM  is anchored, or to correct the
        atomnames. Lastly, the nodes may have a `replace` attribute, which
        is a dictionary of ``{attribute_name: new_value}`` pairs. The special
        case here is if attribute_name is ``'atomname'`` and new_value is
        ``None``: in this case the node will be removed.
        Lastly, the graph (not its nodes) needs a 'name' attribute.

    Returns
    -------
    list of tuples of (networkx.Graph, dict)
        All PTMs from ``known_PTMs`` needed to describe the PTM atoms in
        ``residue`` along with a ``dict`` of node correspondences. The order of
        ``known_PTMs`` is preserved.

    Raises
    ------
    KeyError
        Not all PTM atoms in ``residue`` can be covered with ``known_PTMs``.
    """
    # BASECASE: residue_ptms is empty
    if not any(res_ptm[0] for res_ptm in residue_ptms):
        return []
    # REDUCTION: Apply one of known_PTMs, remove those atoms from residue_ptms
    # COMBINATION: add the applied option to the output.
    for idx, option in enumerate(known_PTMs):
        ptm, matcher = option
        matches = list(matcher.subgraph_isomorphisms_iter())
        # Matches: [{res_idxs: ptm_idxs}, {...}, ...]
        # All non-PTM atoms in residue are always available for matching...
        available = set(n_idx for n_idx in residue
                        if not residue.nodes[n_idx].get('PTM_atom', False))
        # ... and only add non consumed PTM atoms
        for res_ptm in residue_ptms:
            available.update(res_ptm[0])
        for match in matches:
            matching = set(match.keys())
            has_anchor = any(m in r[1] for m in matching for r in residue_ptms)
            if matching.issubset(available) and has_anchor:
                new_res_ptms = []
                for res_ptm in residue_ptms:
                    new_res_ptms.append((res_ptm[0] - matching, res_ptm[1]))
                # Continue with the remaining ptm atoms, and try just this
                # option and all smaller.
                try:
                    return [(ptm, match)] + identify_ptms(residue, new_res_ptms, known_PTMs[idx:])
                except KeyError:
                    continue
    raise KeyError('Could not identify PTM')


def allowed_ptms(residue, res_ptms, known_ptms):
    """
    Finds all PTMs in ``known_ptms`` which might be relevant for ``residue``.

    Parameters
    ----------
    residue : networkx.Graph

    res_ptms : list of tuples of two sets of node indices
        As returned by ``find_PTM_atoms``.
        Currently not used.

    known_ptms : iterable of networkx.Graphs

    Yields
    ------
    tuple of (networkx.Graph, PTMGraphMatcher)
        All graphs in known_ptms which are subgraphs of residue.
    """
    # TODO: filter by element count first
    for ptm in known_ptms:
        ptm_graph_matcher = PTMGraphMatcher(residue, ptm)
        if ptm_graph_matcher.subgraph_is_isomorphic():
            yield ptm, ptm_graph_matcher


def fix_ptm(molecule):
    '''
    Canonizes all PTM atoms in molecule, and labels the relevant residues with
    which PTMs were recognized.

    Parameters
    ----------
    molecule : networkx.Graph
        Must not have missing atoms, and atomnames must be correct. Atoms which
        could not be recognized must be labeled with the attribute
        PTM_atom=True.

    Returns
    -------
    None
        Modifies ``molecule`` such that atomnames of PTM atoms are corrected,
        and the relevant residues have been labeled with which PTMs were
        recognized.
    '''
    PTM_atoms = find_PTM_atoms(molecule)

    def key_func(ptm_atoms):
        node_idxs = ptm_atoms[-1]  # The anchors
        return sorted(molecule.nodes[idx]['resid'] for idx in node_idxs)

    ptm_atoms = sorted(PTM_atoms, key=key_func)

    resid_to_idxs = defaultdict(list)
    for n_idx in molecule:
        residx = molecule.nodes[n_idx]['resid']
        resid_to_idxs[residx].append(n_idx)
    resid_to_idxs = dict(resid_to_idxs)

    known_PTMs = molecule.force_field.modifications

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
        residue = molecule.subgraph(n_idxs)
        options = allowed_ptms(residue, res_ptms, known_PTMs)
        # TODO/FIXME: This includes anchors in sorting by size.
        options = sorted(options, key=lambda opt: len(opt[0]), reverse=True)
        identified = identify_ptms(residue, res_ptms, options)
        # INFO output: Identified modification X.
        print('Today your answer is: {}!'.format([out[0].graph['name'] for out in identified]))
        for ptm, match in identified:
            for mol_idx, ptm_idx in match.items():
                ptm_node = ptm.nodes[ptm_idx]
                mol_node = molecule.nodes[mol_idx]
                # Names of PTM atoms still need to be corrected, and for some
                # non PTM atoms attributes need to change.
                # Nodes with 'replace': {'atomname': None} will be removed.
                if ptm_node['PTM_atom'] or 'replace' in ptm_node:
                    mol_node['graph'] = molecule.subgraph([mol_idx]).copy()
                    to_replace = ptm_node.copy()
                    if 'replace' in to_replace:
                        del to_replace['replace']
                    to_replace.update(ptm_node.get('replace', dict()))
                    for attr_name, val in to_replace.items():
                        if attr_name == 'atomname' and val is None:
                            # DEBUG output
                            print('Removing node {}, {}'.format(mol_idx), mol_node['atomname'])
                            mol_node.remove_node(mol_idx)
                            n_idxs.remove(mol_idx)
                            break
                        # DEBUG output
                        print('Changing attribute {} from {} to {}'.format(attr_name, mol_node[attr_name], val))
                        mol_node[attr_name] = val
            for n_idx in n_idxs:
                molecule.nodes[n_idx]['modifications'] = molecule.nodes[n_idx].get('modifications', [])
                molecule.nodes[n_idx]['modifications'].append(ptm)
    return None


class CanonicalizeModifications(Processor):
    def run_molecule(self, molecule):
        fix_ptm(molecule)
        return molecule
