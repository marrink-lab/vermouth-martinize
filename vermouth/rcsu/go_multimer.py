# Copyright 2026 University of Groningen
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
Obtain the structural bias of the Go model for multimers, such that copies of
a molecule share a single topology.
"""
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from ..graph_utils import make_residue_graph
from ..molecule import Interaction
from ..processors.processor import Processor
from ..selectors import is_protein
from ..gmx.topology import Atomtype, NonbondParam
from ..log_helpers import StyleAdapter, get_logger
from .go_structure_bias import select_go_contacts

LOGGER = StyleAdapter(get_logger(__name__))

# spread in nm of the distances of a contact over the copies above which we warn
MAX_SPREAD = 0.1


def check_complete_chains(system):
    """
    Check that every protein chain forms a single molecule. A chain is split
    over several molecules if residues are missing or if its chain ID is used
    by another chain as well.

    Parameters
    ----------
    system: :class:`vermouth.system.System`

    Raises
    ------
    ValueError
        If a chain occurs in more than one protein molecule.
    """
    molecules = defaultdict(set)
    for mol_idx, molecule in enumerate(system.molecules):
        if is_protein(molecule):
            for chain in set(nx.get_node_attributes(molecule, 'chain').values()):
                molecules[chain].add(mol_idx)
    for chain, mol_indices in molecules.items():
        if len(mol_indices) > 1:
            raise ValueError(f"Chain {chain} is split over {len(mol_indices)} molecules, because "
                             "residues are missing or the chain ID is repeated. The multimer Go "
                             "model requires complete chains with unique chain IDs.")


def read_nonbond_params(path):
    """
    Read the ``[ nonbond_params ]`` of a force field itp file, such as
    martini_v3.0.0.itp. Preprocessor directives are skipped.

    Parameters
    ----------
    path: str or pathlib.Path

    Returns
    -------
    dict[frozenset(str), tuple(float, float)]
        sigma and epsilon per pair of atomtypes
    """
    params = {}
    section = None
    with open(path, encoding='utf-8') as itp:
        for line in itp:
            line = line.split(';')[0].strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('['):
                section = line.strip('[ ]')
            elif section == 'nonbond_params':
                type_a, type_b, _, sigma, epsilon = line.split()[:5]
                params[frozenset((type_a, type_b))] = (float(sigma), float(epsilon))
    return params


class ComputeMultimerGoBias(Processor):
    """
    Generate the Go model structural bias for a system of separate molecules,
    such that the copies of a molecule (molecules with the same moltype) share
    a single topology.

    Contacts are pooled per moltype by resid: a contact found in any copy
    (or with `intersection` in every copy) applies to every copy, with the
    mean distance over the copies; for an inter contact only over its shells
    of pairs of molecules, see :meth:`select_inter_contacts`. Every residue
    already has a Go virtual site (site A, see
    :class:`~vermouth.rcsu.go_vs_includes.VirtualSiteCreator`); the sites
    B, C and D are added where needed:

    - intra contact (i, j): A_i-A_j with +go_eps and B_i-B_j with -go_eps;
      B_i-B_j is excluded within the molecule, so B cancels A only between
      copies. BB_i-BB_j is excluded as in the regular Go model.
    - inter contact (i, j): C_i-C_j with +go_eps_inter (same sequence) or
      +go_eps_hetero, and D_i-D_j with minus the backbone-backbone
      interaction of the force field, which cancels the latter between
      molecules. Within a molecule C_i-C_j and D_i-D_j are excluded.

    Parameters
    ----------
    cutoff_short: float
        distances in nm smaller than this are ignored
    cutoff_long: float
        distances in nm larger than this are ignored
    go_eps: float
        epsilon of intra-molecular contacts in kJ/mol
    go_eps_inter: float
        epsilon of contacts between molecules with the same sequence
    go_eps_hetero: float
        epsilon of contacts between molecules with different sequences
    res_dist: int
        residues closer than res_dist along the residue graph are ignored
    nonbond_params: dict[frozenset(str), tuple(float, float)]
        sigma and epsilon of the force field per pair of atomtypes, see
        :func:`read_nonbond_params`
    go_anchor_bead: str
        name of the backbone bead on which the virtual sites are placed
    shell_width: float
        width in nm of the shells of an inter contact, see
        :meth:`select_inter_contacts`
    intersection: bool
        if True, an intra contact is only used if it is found in every copy,
        and an inter contact if it is found in every pair of molecules of its
        shells
    """
    def __init__(self,
                 cutoff_short,
                 cutoff_long,
                 go_eps,
                 go_eps_inter,
                 go_eps_hetero,
                 res_dist,
                 nonbond_params,
                 go_anchor_bead='BB',
                 shell_width=0.05,
                 intersection=False):
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.go_eps_inter = go_eps_inter
        self.go_eps_hetero = go_eps_hetero
        self.res_dist = res_dist
        self.nonbond_params = nonbond_params
        self.backbone = go_anchor_bead
        self.shell_width = shell_width
        self.intersection = intersection
        self.conversion_factor = 2**(1/6)
        # don't modify
        self.system = None
        self.backbone_atoms = {}
        self.sequences = {}

    def contact_selector(self, system):
        """
        Select all contacts from the contact map that according to their
        distance and graph connectivity are eligible to form a Go bond, see
        :func:`~vermouth.rcsu.go_structure_bias.select_go_contacts`.

        Parameters
        ----------
        system: :class:`vermouth.system.System`

        Returns
        -------
        dict[tuple((int, collections.abc.Hashable), (int, collections.abc.Hashable)), float]
            distance per contact between two backbone beads, each given as
            molecule index and node key
        """
        # residue graph of all molecules, with (molecule index, residue) as nodes
        res_graph = nx.Graph()
        for mol_idx, molecule in enumerate(system.molecules):
            mol_res_graph = make_residue_graph(molecule)
            res_graph.update(nx.relabel_nodes(mol_res_graph, {node: (mol_idx, node) for node in mol_res_graph}))
        # (chain, input resid) -> residue
        residues = {}
        for resnode, residue in res_graph.nodes(data=True):
            key = (residue.get('chain'), residue['stash'].get('resid'))
            if key in residues:
                raise ValueError(f"Residue {key[1]} of chain {key[0]} occurs more than once, for "
                                 "example because of an insertion code. The multimer Go model "
                                 "requires unique chain IDs and residue numbers.")
            residues[key] = resnode

        contacts = {}
        for res_a, res_b, bb_a, bb_b, dist in select_go_contacts(
                system.go_params["go_map"][0], res_graph, lambda chain, resid: residues.get((chain, resid)),
                self.res_dist, self.cutoff_short, self.cutoff_long, self.backbone):
            contacts[tuple(sorted(((res_a[0], bb_a), (res_b[0], bb_b))))] = dist
        return contacts

    def pool_contacts(self, contacts):
        """
        Pool the contacts of all copies of a molecule by moltype and resid.

        Parameters
        ----------
        contacts: dict
            contacts as returned by :meth:`contact_selector`

        Returns
        -------
        intra: dict[str, dict[tuple(int, int), list[float]]]
            distances over the copies per moltype and pair of resids
        inter: dict[tuple((str, int), (str, int)), list[tuple(tuple(int, int), float)]]
            pairs of molecule indices and distances per pair of moltype and resid
        """
        intra = defaultdict(lambda: defaultdict(list))
        inter = defaultdict(list)
        for pair, dist in contacts.items():
            ends = [(self.system.molecules[mol_idx].meta['moltype'],
                     self.system.molecules[mol_idx].nodes[bb_node]['resid'])
                    for mol_idx, bb_node in pair]
            if pair[0][0] == pair[1][0]:
                intra[ends[0][0]][tuple(sorted(resid for _, resid in ends))].append(dist)
            else:
                inter[tuple(sorted(ends))].append((tuple(sorted(mol_idx for mol_idx, _ in pair)), dist))
        return intra, inter

    def select_inter_contacts(self, inter):
        """
        Select the inter contacts and the distances that define their sigma.
        The occurrences of a contact within `shell_width` of its shortest
        distance are its anchors. Its shells are the pairs of molecules of
        the same moltypes that have any inter contact and whose distance,
        measured between the centres of their backbone beads, is within
        `shell_width` of that of the pair of an anchor. The sigma of a
        contact uses all its occurrences in its shells. With `intersection`,
        a contact is only used if it is found in every pair of molecules of
        its shells. Contacts between two charged backbone beads are not used,
        as their Coulomb interaction cannot be excluded between molecules.

        Parameters
        ----------
        inter: dict[tuple((str, int), (str, int)), list[tuple(tuple(int, int), float)]]
            inter contacts as returned by :meth:`pool_contacts`

        Returns
        -------
        dict[tuple((str, int), (str, int)), list[float]]
            distances per pair of moltype and resid
        """
        mol_pairs = {mols for found in inter.values() for mols, _ in found}
        centres = {mol_idx: np.mean([atom['position'] for atom in self.system.molecules[mol_idx].nodes.values()
                                     if atom.get('atomname') == self.backbone], axis=0)
                   for mols in mol_pairs for mol_idx in mols}
        separation = {mols: np.linalg.norm(centres[mols[0]] - centres[mols[1]]) for mols in mol_pairs}
        moltype_pairs = defaultdict(set)
        for ends, found in inter.items():
            moltype_pairs[(ends[0][0], ends[1][0])].update(mols for mols, _ in found)

        selected = {}
        for ends, found in inter.items():
            shortest = min(dist for _, dist in found)
            # distances between the molecules of the anchors
            anchors = {separation[mols] for mols, dist in found if dist <= shortest + self.shell_width}
            shells = {mols for mols in moltype_pairs[(ends[0][0], ends[1][0])]
                      if any(abs(separation[mols] - anchor) <= self.shell_width for anchor in anchors)}
            if self.intersection and not shells <= {mols for mols, _ in found}:
                continue
            atoms = [f"{moltype}_{resid}" for moltype, resid in ends]
            if all(self.backbone_atoms[end].get('charge') for end in ends):
                LOGGER.warning("The Go contact {} - {} joins two charged backbone beads of different "
                               "molecules, whose Coulomb interaction cannot be excluded as in a merged "
                               "Go model; this Go bond is not created.", *atoms, type='go-charged-contact')
                continue
            if max(anchors) - min(anchors) > self.shell_width:
                LOGGER.warning("The Go contact {} - {} is about equally short between pairs of molecules "
                               "{:.3f} to {:.3f} nm apart, so it could not be assigned to a unique "
                               "interface; it is pooled over the pairs of molecules at all of these "
                               "distances.",
                               *atoms, min(anchors), max(anchors), type='go-contact-shells')
            selected[ends] = [dist for mols, dist in found if mols in shells]
        return selected

    def compute_go_interaction(self, intra, inter):
        """
        Compute the interaction parameters of the pooled contacts and store
        them in the systems attribute gmx_topology_params.

        Parameters
        ----------
        intra: dict[str, dict[tuple(int, int), list[float]]]
            intra-molecular contacts as returned by :meth:`pool_contacts`
        inter: dict[tuple((str, int), (str, int)), list[float]]
            inter-molecular contacts as returned by :meth:`select_inter_contacts`
        """
        nonbond_params = self.system.gmx_topology_params["nonbond_params"]
        for moltype, pairs in intra.items():
            for pair, dists in pairs.items():
                atoms = tuple(f"{moltype}_{resid}" for resid in pair)
                dist = self._pooled_distance(dists, atoms)
                sigma = dist / self.conversion_factor
                nonbond_params.append(NonbondParam(
                    atoms=atoms,
                    sigma=sigma, epsilon=self.go_eps,
                    meta={"comment": [f"go bond {dist}"]}))
                nonbond_params.append(NonbondParam(
                    atoms=tuple(f"{moltype}_{resid}_B" for resid in pair),
                    sigma=sigma, epsilon=-self.go_eps,
                    meta={"comment": [f"go bond cancel intra {dist}"]}))

        for ends, dists in inter.items():
            atoms = tuple(f"{moltype}_{resid}_C" for moltype, resid in ends)
            dist = self._pooled_distance(dists, atoms)
            homo = self.sequences[ends[0][0]] == self.sequences[ends[1][0]]
            kind = 'homo' if homo else 'hetero'
            bb_atoms = [self.backbone_atoms[end] for end in ends]
            bb_pair = frozenset(atom['atype'] for atom in bb_atoms)
            if bb_pair not in self.nonbond_params:
                raise KeyError(f"The force field has no nonbond_params for the backbone "
                               f"bead types {' and '.join(sorted(bb_pair))}.")
            sigma_bb, eps_bb = self.nonbond_params[bb_pair]
            nonbond_params.append(NonbondParam(
                atoms=atoms,
                sigma=dist / self.conversion_factor,
                epsilon=self.go_eps_inter if homo else self.go_eps_hetero,
                meta={"comment": [f"go bond inter {kind} {dist}"]}))
            nonbond_params.append(NonbondParam(
                atoms=tuple(f"{moltype}_{resid}_D" for moltype, resid in ends),
                sigma=sigma_bb, epsilon=-eps_bb,
                meta={"comment": [f"BB cancel inter {kind} {dist}"]}))

    def add_go_sites(self, molecule, intra_pairs, inter_pairs, new_atomtypes):
        """
        Add the sites B, C and D to a molecule, together with the exclusions
        within the molecule, where the intra contacts keep their Go bond and
        the inter contacts their regular backbone interaction.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
        intra_pairs: collections.abc.Iterable[tuple(int, int)]
            resids of the intra-molecular contacts of the moltype
        inter_pairs: collections.abc.Iterable[tuple((str, int), (str, int))]
            moltypes and resids of the inter-molecular contacts of the moltype
        new_atomtypes: bool
            if True the atomtypes of the sites are added to the system

        Returns
        -------
        dict[str, dict[int, collections.abc.Hashable]]
            node of the new site per resid, per site
        """
        moltype = molecule.meta['moltype']
        backbone = {atom['resid']: node for node, atom in molecule.nodes(data=True)
                    if atom.get('atomname') == self.backbone}
        inter_resids = {resid for ends in inter_pairs for end_moltype, resid in ends
                        if end_moltype == moltype}
        sites = {site: self.add_sites(molecule, site, resids, backbone, new_atomtypes)
                 for site, resids in (('B', {resid for pair in intra_pairs for resid in pair}),
                                      ('C', inter_resids),
                                      ('D', inter_resids))}
        # the copies of a moltype share their atomtypes, so the sites of an
        # inter contact between two copies have to be excluded within each
        copy_pairs = [(end_a[1], end_b[1]) for end_a, end_b in inter_pairs
                      if end_a[0] == end_b[0] and end_a[1] != end_b[1]]
        exclusions = [(backbone, 'Go model exclusion', pair) for pair in intra_pairs]
        exclusions += [(sites['B'], 'Go model exclusion VWB', pair) for pair in intra_pairs]
        for site in 'CD':
            exclusions += [(sites[site], f'Go model exclusion VW{site}', pair)
                           for pair in copy_pairs]
        for nodes, group, (resid_a, resid_b) in exclusions:
            molecule.interactions['exclusions'].append(Interaction(
                atoms=(nodes[resid_a], nodes[resid_b]), parameters=[], meta={"group": group}))
        return sites

    def add_sites(self, molecule, site, resids, backbone, new_atomtypes):
        """
        Add a Go virtual site on the backbone bead of each residue in
        `resids`, with the atomtype "<moltype>_<resid>_<site>" and the
        atomname "VW<site>".

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
        site: str
            letter of the site
        resids: collections.abc.Iterable[int]
        backbone: dict[int, collections.abc.Hashable]
            backbone node per resid
        new_atomtypes: bool
            if True the atomtypes are added to the system

        Returns
        -------
        dict[int, collections.abc.Hashable]
            node of the new site per resid
        """
        sites = {}
        new_node_id = max(molecule.nodes)
        charge_group = max(nx.get_node_attributes(molecule, 'charge_group').values(), default=0)
        for resid in sorted(resids):
            new_node_id += 1
            charge_group += 1
            atom = molecule.nodes[backbone[resid]]
            molecule.add_node(new_node_id,
                              resid=resid,
                              resname=atom['resname'],
                              atype=f"{molecule.meta['moltype']}_{resid}_{site}",
                              charge_group=charge_group,
                              chain=atom['chain'],
                              position=atom['position'],
                              atomname=f"VW{site}",
                              charge=0,
                              mass=0.0,
                              cgsecstruct=atom.get('cgsecstruct', None),
                              stash=atom.get('stash', None),
                              go_site=site)
            molecule.interactions['virtual_sitesn'].append(Interaction(
                atoms=[new_node_id, backbone[resid]],
                parameters=['1'],
                meta={'go_vs': True, 'group': 'Virtual go site'},
            ))
            if new_atomtypes:
                self.system.gmx_topology_params['atomtypes'].append(
                    Atomtype(node=new_node_id, molecule=molecule, sigma=0.0, epsilon=0.0, meta={}))
            sites[resid] = new_node_id
        return sites

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        self.system = system
        # the backbone beads and the sequence of each moltype; copies share them
        self.backbone_atoms = {(molecule.meta['moltype'], atom['resid']): atom
                               for molecule in system.molecules
                               for atom in molecule.nodes.values()
                               if atom.get('atomname') == self.backbone}
        self.sequences = {molecule.meta['moltype']: [atom['resname'] for atom in molecule.nodes.values()
                                                     if atom.get('atomname') == self.backbone]
                          for molecule in system.molecules}
        copies = Counter(molecule.meta['moltype'] for molecule in system.molecules)
        intra, inter = self.pool_contacts(self.contact_selector(system))
        if self.intersection:
            # an intra contact has one distance per copy in which it is found
            intra = {moltype: {pair: dists for pair, dists in pairs.items() if len(dists) == copies[moltype]}
                     for moltype, pairs in intra.items()}
        inter = self.select_inter_contacts(inter)
        self.compute_go_interaction(intra, inter)

        moltypes_done = set()
        for molecule in system.molecules:
            molecule.citations.add('M3_GO_OLIGOMERS')
            moltype = molecule.meta['moltype']
            intra_pairs = intra.get(moltype, {})
            inter_pairs = [ends for ends in inter if moltype in (ends[0][0], ends[1][0])]
            if not intra_pairs and not inter_pairs:
                continue
            sites = self.add_go_sites(molecule, intra_pairs, inter_pairs,
                                      new_atomtypes=moltype not in moltypes_done)
            if moltype in moltypes_done:
                continue
            moltypes_done.add(moltype)
            kinds = Counter('homo' if self.sequences[end_a[0]] == self.sequences[end_b[0]]
                            else 'hetero' for end_a, end_b in inter_pairs)
            LOGGER.info("{}: {} {}, {} intra-, {} homo-inter- and {} hetero-inter-molecular "
                        "Go contacts; added {} B, {} C and {} D sites per copy.",
                        moltype, copies[moltype], 'copy' if copies[moltype] == 1 else 'copies',
                        len(intra_pairs), kinds['homo'], kinds['hetero'],
                        *(len(sites[site]) for site in 'BCD'))
        return system

    @staticmethod
    def _pooled_distance(dists, atoms):
        """
        Mean of the distances of a contact over the copies; warns if they
        spread more than MAX_SPREAD.
        """
        spread = max(dists) - min(dists)
        if spread > MAX_SPREAD:
            LOGGER.warning("The distance of the Go contact {} - {} differs by {:.3f} nm "
                           "between copies; the mean is used.", *atoms, spread, type='go-contact-spread')
        return np.mean(dists)
