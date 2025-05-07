# Copyright 2023 University of Groningen
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
Obtain the structural bias for the Go model.
"""
import numpy as np
import networkx as nx
from ..graph_utils import make_residue_graph
from ..molecule import Interaction
from ..processors.processor import Processor
from ..selectors import filter_minimal, select_backbone
from ..gmx.topology import NonbondParam
from .go_utils import get_go_type_from_attributes
from ..log_helpers import StyleAdapter, get_logger
import sys
LOGGER = StyleAdapter(get_logger(__name__))

class ComputeStructuralGoBias(Processor):
    """
    Generate the Go model structural bias for a system
    of molecules. This processor class has two main
    functions: .contact_selector and .compute_bias.
    The .run_molecule function simply loops over all
    molecules in the system and calls the other two
    functions. The computed structural bias parameters
    are stored in `system.gmx_topology_params` and can
    be written out using the `vermouth.gmx.write_topology`
    function.

    Subclassing
    -----------
    In order to customize the Go-model structural bias
    it is recommended to subclass this function and
    overwrite the ``contact_selector`` method and/or
    the ``compute_bias`` method. This subclassed Processor
    then has to be added to the into the martinize2
    pipeline in place of the StructuralBiasWriter or as
    replacement in the GoPipeline.
    """
    def __init__(self,
                 go_map_system,
                 all_molecule_system,
                 cutoff_short,
                 cutoff_long,
                 go_eps,
                 cutoff_short_inter,
                 cutoff_long_inter,
                 go_eps_inter,
                 cutoff_short_intra,
                 cutoff_long_intra,
                 go_eps_intra,
                 res_dist,
                 moltype,
                 go_anchor_bead,
                 go_method,
                 go_ff,
                 system=None,
                 res_graph=None):
        """
        Initialize the Processor with arguments required
        to setup the Go model structural bias.

        Parameters
        ----------
        contact_map: list[(str, int, str, int)]
            list of contacts defined as by the chain
            identifier and residue index
        cutoff_short: float
            distances in nm smaller than this are ignored
        cutoff_long: float
            distances in nm larger than this are ignored
        go_eps: float
            epsilon value of the structural bias in
            kJ/mol
        res_dist: int
            if nodes are closer than res_dist along
            the residue graph they are ignored; this
            is similar to sequence distance but takes
            into account disulfide bridges for example
        moltype: str
            name of the molecule to treat
        res_graph: :class:`vermouth.molecule.Molecule`
            residue graph of the molecule; if None it
            gets generated automatically
        system: :class:`vermouth.system.System`
            the system
        magic_number: float
            magic number for Go contacts from the old
            GoVirt script.
        backbone: str
            name of backbone atom where virtual site is placed
        """
        self.go_map_system = go_map_system
        self.all_molecule_system = all_molecule_system
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.cutoff_short_inter=cutoff_short_inter
        self.cutoff_long_inter=cutoff_long_inter
        self.go_eps_inter=go_eps_inter
        self.cutoff_short_intra=cutoff_short_intra
        self.cutoff_long_intra=cutoff_long_intra
        self.go_eps_intra=go_eps_intra
        self.res_dist = res_dist
        self.moltype = moltype
        self.backbone = go_anchor_bead
        self.method = go_method
        self.ff_file = go_ff
        self.molecule_graphs = {}
        # don't modify
        self.res_graph = None
        self.system = system
        self.symmetrical_matrix = []
        self.lennard_jones = []
        self.__chain_id_to_resnode = {}
        self.conversion_factor = 2**(1/6)
        self.before_to_after_merge = {}
        self.after_to_before_merge = {}

    # do not overwrite when subclassing
    def _chain_id_to_resnode(self, chain, resid):
        """
        Return the node corresponding to the chain and
        resid. First time the function is run the dict
        is being created.

        Parameters
        ----------
        chain: str
            chain identifier
        resid: int
            residue index

        Returns
        -------
        dict
            a dict matching the chain,resid to the self.res_graph node
        """
        if self.__chain_id_to_resnode:
            if self.__chain_id_to_resnode.get((chain, resid), None) is not None:
                return self.__chain_id_to_resnode[(chain, resid)]
            else:
                print('do not see in dict')
                LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')

        # for each residue collect the chain and residue in a dict
        # we use this later for identifying the residues from the
        # contact map
        if self.method == 0:
            for resnode in self.res_graph.nodes:
                chain_key = self.res_graph.nodes[resnode].get('chain', None)
                # in vermouth within a molecule all resid are unique
                # when merging multiple chains we store the old resid
                # the go model always references the input resid i.e.
                # the _old_resid
                resid_key = self.res_graph.nodes[resnode].get('_old_resid')
                self.__chain_id_to_resnode[(chain_key, resid_key)] = resnode


        elif self.method == 1:
            chain_to_mol_id = self.go_map_system.go_params['chain_to_mol_id']
            for resnode in self.all_molecule_graph.nodes:
                # print(resnode)
                chain_key = self.all_molecule_graph.nodes[resnode].get('chain', None)
                # in vermouth within a molecule all resid are unique
                # when merging multiple chains we store the old resid
                # the go model always references the input resid i.e.
                # the _old_resid
                before_merge = self.all_molecule_graph.nodes[resnode].get('resid_before_merge')
                resid_key = self.all_molecule_graph.nodes[resnode].get('_old_resid')
                self.before_to_after_merge[before_merge, chain_key] = resid_key
                self.after_to_before_merge[resid_key] = (before_merge, chain_key)
                # print(chain_key, resid_key)
                self.__chain_id_to_resnode[(chain_key, resid_key)] = [resnode]
            for mol in self.go_map_system.molecules:
                current_chain = next(iter(mol.nodes.values()))['chain']
                mol = make_residue_graph(mol)
                self.molecule_graphs[chain_to_mol_id[current_chain]] = mol
                for resnode in mol:
                    chain_key = mol.nodes[resnode].get('chain', None)
                    before_merge = mol.nodes[resnode].get('_old_resid', None)
                    resid_key = self.before_to_after_merge[before_merge, chain_key]
                    # print(chain_key, resid_key)
                    # Check if key exists before appending
                    if (chain_key, resid_key) in self.__chain_id_to_resnode:
                        self.__chain_id_to_resnode[(chain_key, resid_key)].append(resnode)
                    else:
                        # Create a new list if key doesn't exist
                        print(f'Warning, there is no resnode in the all_molecule_graph with {chain_key, resid_key}')

        if self.__chain_id_to_resnode.get((chain, resid), None) is not None:
            return self.__chain_id_to_resnode[(chain, resid)]
        else:
            print('ook niet goed')
            LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')


    def contact_selector(self, molecule):
        """
        Select all contacts from the contact map
        that according to their distance and graph
        connectivity are eligible to form a Go
        bond and create exclusions between the
        backbone beads of those contacts.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`

        Returns
        -------
        list[(collections.abc.Hashable, collections.abc.Hashable, float)]
            list of node keys and distance
        """
        chain_to_mol_id = self.go_map_system.go_params['chain_to_mol_id']
        chain_map = self.go_map_system.go_params['unique_mols_map']
        reference_chain_map = self.go_map_system.go_params['reference_chain_map']
        added_contacts = []
        # distance_matrix of eligible pairs as tuple(node, node, dist)
        contact_matrix = []
        # find all pairs of residues that are within bonded distance of
        # self.res_dist
        if self.method == 0:
            connected_pairs = dict(nx.all_pairs_shortest_path_length(self.res_graph,
                                                                 cutoff=self.res_dist))
        elif self.method == 1:
            connected_pairs = dict(nx.all_pairs_shortest_path_length(self.all_molecule_graph,
                                                                 cutoff=self.res_dist))


        bad_chains_warning = False

        for contact in self.go_map_system.go_params["go_map"][0]:
            resIDA, chainA, resIDB, chainB = contact

            if self.method == 0:
                # identify the contact in the residue graph based on
                # chain ID and resid
                resA = self._chain_id_to_resnode(chainA, resIDA)
                resB = self._chain_id_to_resnode(chainB, resIDB)
                # make sure that both residues are not connected
                # note: contacts should be symmetric so we only
                # check against one

                if (resA is not None) and (resB is not None):
                    if resB not in connected_pairs[resA]:
                        # now we lookup the backbone nodes within the residue contact
                        try:
                            bb_node_A = next(filter_minimal(self.res_graph.nodes[resA]['graph'],
                                                            select_backbone,
                                                            bb_atomname=self.backbone))
                            bb_node_B = next(filter_minimal(self.res_graph.nodes[resB]['graph'],
                                                            select_backbone,
                                                            bb_atomname=self.backbone))
                        except StopIteration:
                            LOGGER.warning(f'No backbone atoms with name "{self.backbone}" found in molecule. '
                                        'Check -go-backbone argument if your forcefield does not use this name for '
                                        'backbone bead atoms. Go model cannot be generated. Will exit now.')
                            sys.exit(1)

                        # compute the distance between bb-beads
                        dist = np.linalg.norm(molecule.nodes[bb_node_A]['position'] -
                                            molecule.nodes[bb_node_B]['position'])
                        # verify that the distance between BB-beads satisfies the
                        # cut-off criteria
                        if self.cutoff_long > dist > self.cutoff_short:
                            atype_a = get_go_type_from_attributes(self.res_graph.nodes[resA]['graph'],
                                                                    _old_resid=resIDA,
                                                                    chain=chainA,
                                                                    prefix=self.moltype,
                                                                    method=self.method)
                            atype_b = get_go_type_from_attributes(self.res_graph.nodes[resB]['graph'],
                                                                    _old_resid=resIDB,
                                                                    chain=chainB,
                                                                    prefix=self.moltype,
                                                                    method=self.method)
                            # Check if symmetric contact has already been processed before
                            # and if so, we append the contact to the final symmetric contact matrix
                            # and add the exclusions. Else, we add to the full valid contact_matrix
                            # and continue searching.
                            if (atype_b, atype_a, dist) in contact_matrix:
                                # generate backbone-backbone exclusions
                                # perhaps one day can be its own function
                                excl = Interaction(atoms=(bb_node_A, bb_node_B),
                                                parameters=[], meta={"group": "Go model exclusion"})
                                molecule.interactions['exclusions'].append(excl)
                                self.symmetrical_matrix.append((atype_a, atype_b, dist))
                            else:
                                contact_matrix.append((atype_a, atype_b, dist))

            if self.method == 1:
                chain_ids = self._chain_id_to_resnode(chainA, resIDA)
                resA, resA_chain = chain_ids[0], chain_ids[1]
                chain_ids = self._chain_id_to_resnode(chainB, resIDB)
                resB, resB_chain = chain_ids[0], chain_ids[1]
                if (resA is not None) and (resB is not None):
                    if resB not in connected_pairs[resA]:
                        # now we lookup the backbone nodes within the residue contact
                        try:
                            bb_node_A = next(filter_minimal(self.all_molecule_graph.nodes[resA]['graph'],
                                                            select_backbone,
                                                            bb_atomname=self.backbone))
                            bb_node_B = next(filter_minimal(self.all_molecule_graph.nodes[resB]['graph'],
                                                            select_backbone,
                                                            bb_atomname=self.backbone))
                            
                            # here get the bead names of the backbones, to later know which LJ
                            # sigma and epsilon are related to this bond
                            graphA = self.all_molecule_graph.nodes[resA]['graph']
                            for atom_id, atom_attrs in graphA.nodes.items():
                                if atom_attrs.get('atomname') == 'BB':
                                    if 'atype' in atom_attrs:
                                        backbone_atype = atom_attrs['atype']
                                        beadA = backbone_atype

                            graphB = self.all_molecule_graph.nodes[resB]['graph']
                            for atom_id, atom_attrs in graphB.nodes.items():
                                if atom_attrs.get('atomname') == 'BB':
                                    if 'atype' in atom_attrs:
                                        backbone_atype = atom_attrs['atype']
                                        beadB = backbone_atype

                        except StopIteration:
                            LOGGER.warning(f'No backbone atoms with name "{self.backbone}" found in molecule. '
                                        'Check -go-backbone argument if your forcefield does not use this name for '
                                        'backbone bead atoms. Go model cannot be generated. Will exit now.')
                            sys.exit(1)

                        # compute the distance between bb-beads
                        dist = np.linalg.norm(self.all_molecule_system.nodes[bb_node_A]['position'] -
                                            self.all_molecule_system.nodes[bb_node_B]['position'])
                        # verify that the distance between BB-beads satisfies the
                        # cut-off criteria
                        if chainA == chainB: # intra
                            low = self.cutoff_short_intra
                            high = self.cutoff_long_intra
                            bond_type = 'intra'
                        elif chainA != chainB:
                            low = self.cutoff_short_inter
                            high = self.cutoff_long_inter
                            if chainA in chain_map[chainB]:
                                bond_type = 'inter'
                            else:
                                bond_type = 'other'

                        if high > dist > low:
                            chainA_id = chain_to_mol_id[chainA]
                            if chainA_id in self.molecule_graphs:
                                old_resid = self.after_to_before_merge[resIDA][0]
                                graph = self.molecule_graphs[chainA_id]
                                atype_a = get_go_type_from_attributes(graph.nodes[resA_chain]['graph'],
                                                                        _old_resid=old_resid,
                                                                        chain=chainA,
                                                                        prefix=self.moltype,
                                                                        method=self.method)
                            else:
                                print(f'Warning there no graph generated for the chain: {chainA}')

                            chainB_id = chain_to_mol_id[chainB]
                            if chainB_id in self.molecule_graphs:
                                graph = self.molecule_graphs[chainB_id]
                                old_resid = self.after_to_before_merge[resIDB][0]
                                atype_b = get_go_type_from_attributes(graph.nodes[resB_chain]['graph'],
                                                                        _old_resid=old_resid,
                                                                        chain=chainB,
                                                                        prefix=self.moltype,
                                                                        method=self.method)
                            else:
                                print(f'Warning there no graph generated for the chain: {chainB}')
                            
                            exclude_a = sorted([x for x in atype_a if isinstance(x, int)])
                            exclude_b = sorted([x for x in atype_b if isinstance(x, int)])
                            sorted_vs_a = sorted([x for x in atype_a if isinstance(x, str)])
                            sorted_vs_b = sorted([x for x in atype_b if isinstance(x, str)])

                            if (atype_b, atype_a, beadB, beadA, dist) in contact_matrix and ((bond_type, atype_b, atype_a, beadB, beadA) not in added_contacts and (bond_type, atype_a, atype_b, beadA, beadB) not in added_contacts):
                                added_contacts.append((bond_type, atype_b, atype_a, beadB, beadA))
                                # generate backbone-backbone exclusions
                                # perhaps one day can be its own function
                                if bond_type == 'intra':
                                    molecule = self.system.molecules[chainA_id]
                                    excl_site_b = Interaction(atoms=(exclude_a[0], exclude_b[0]),
                                                    parameters=[], meta={"group": "Go model exclusion for virtual sites 'b'"})
                                    excl_site_d = Interaction(atoms=(exclude_a[1], exclude_b[1]),
                                                    parameters=[], meta={"group": "Go model exclusion for virtual sites 'd'"})
                                    molecule.interactions['exclusions'].append(excl_site_b)
                                    molecule.interactions['exclusions'].append(excl_site_d)

                                elif bond_type == 'inter':
                                    ref_chainA = reference_chain_map[chainA]
                                    ref_chainA_id = chain_to_mol_id[ref_chainA]
                                    molecule = self.system.molecules[ref_chainA_id]
                                    excl_site_b = Interaction(atoms=(exclude_a[0], exclude_b[0]),
                                                    parameters=[], meta={"group": "Go model exclusion for virtual sites 'b'"})
                                    excl_site_d = Interaction(atoms=(exclude_a[1], exclude_b[1]),
                                                    parameters=[], meta={"group": "Go model exclusion for virtual sites 'd'"})
                                    molecule.interactions['exclusions'].append(excl_site_b)
                                    molecule.interactions['exclusions'].append(excl_site_d)

                                elif bond_type == 'other': # other type of molecule
                                    pass

                                key_without_dist = (sorted_vs_a, sorted_vs_b, beadA, beadB, bond_type)
                                if all(entry[:5] != key_without_dist for entry in self.symmetrical_matrix): # might give problems, due to the appending of all go interactions of each homologous molecule/chain
                                    self.symmetrical_matrix.append((sorted_vs_a, sorted_vs_b, beadA, beadB, bond_type, dist))
                                if (beadA, beadB) not in self.lennard_jones and (beadB, beadA) not in self.lennard_jones:
                                    self.lennard_jones.append((beadA, beadB))

                            else:
                                contact_matrix.append((atype_a, atype_b, beadA, beadB, dist))
            else:
                pass # can add ensureance that every connection is done maybe

        return

    def compute_go_interaction(self):
        """
        Compute the epsilon value given a distance between
        two nodes, figure out the atomtype name and store
        it in the systems attribute gmx_topology_params.

        Parameters
        ----------
        contacts: list[(str, str, float)]
            list of node-keys and their distance

        Returns
        ----------
        dict[frozenset(str, str): float]
            dict of interaction parameters indexed by atomtype
        """
        contacts = self.symmetrical_matrix
        LJ_combies = self.lennard_jones
        LJ_sigma_epsilon_dict = {}
        contact_bias_list = []
        if self.method == 0:
            for atype_a, atype_b, dist in contacts:
                if self.method == 1:
                    sigma = dist / self.conversion_factor
                    # find the go virtual-sites for this residue
                    # probably can be done smarter but mehhhh
                    contact_bias = NonbondParam(atoms=(atype_a, atype_b),
                                                sigma=sigma,
                                                epsilon=self.go_eps,
                                                meta={"comment": [f"go bond {dist}"]})
                    self.system.gmx_topology_params["nonbond_params"].append(contact_bias)

        elif self.method == 1:
            with open(self.ff_file, 'r') as file:
                current_section = None
                for line in file:
                    line = line.strip()
                    # could give a note to say that the given martini ff does not contain the GO_VIRT ifdef
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line.strip('[]').strip()
                    elif current_section == 'nonbond_params' and line and not line.startswith(';') and not line.startswith('#'):
                        tokens = line.split()
                        bead_1 = tokens[0]
                        bead_2 = tokens[1]
                        LJ_combi = (bead_1, bead_2)
                        if LJ_combi in LJ_combies or (bead_2, bead_1) in LJ_combies:
                            sigma = tokens[3]
                            epsilon = tokens[4]
                            LJ_sigma_epsilon_dict[LJ_combi] = (sigma, epsilon)

            for atype_a, atype_b, beadA, beadB, bond_type, dist in contacts:
                
                if (beadA, beadB) in LJ_sigma_epsilon_dict:
                    LJ_sigma, LJ_epsilon = LJ_sigma_epsilon_dict[(beadA, beadB)]
                    LJ_sigma, LJ_epsilon = float(LJ_sigma), float(LJ_epsilon)
                elif (beadB, beadA) in LJ_sigma_epsilon_dict:
                    LJ_sigma, LJ_epsilon = LJ_sigma_epsilon_dict[(beadB, beadA)]
                    LJ_sigma, LJ_epsilon = float(LJ_sigma), float(LJ_epsilon)
                else:
                    print('Warning there is a bead combi that is not in the LJ_sigma_epsilon_dict')

                if bond_type == 'intra':
                    go_epsilon = self.go_eps_intra
                elif bond_type == 'inter' or bond_type == 'other':
                    go_epsilon = self.go_eps_inter
                go_sigma = dist / self.conversion_factor

                vs_already_added = False
                for counter, (virtual_site_a, virtual_site_b) in enumerate(zip(atype_a, atype_b)):
                    if counter == 0:
                        for contact in contact_bias_list:
                            if ((virtual_site_a, virtual_site_b) == contact[0] or (virtual_site_b, virtual_site_a) == contact[0]):
                                vs_already_added = True

                    if vs_already_added:
                        for contact in contact_bias_list:
                            if contact[0] == (virtual_site_a, virtual_site_b) or contact[0] == (virtual_site_b, virtual_site_a):
                                if counter == 1:
                                    if bond_type == 'inter' or bond_type == 'other':
                                        contact[1] = go_sigma
                                        contact[2] = go_epsilon
                                        contact[3] = [f"inter molecular go bond {dist}"]
                                elif counter == 2:
                                    if bond_type == 'intra':
                                        contact[1] = go_sigma
                                        contact[2] = go_epsilon
                                        contact[3] = [f"intra molecular go bond {dist}"]
                                elif counter == 3:
                                    if bond_type == 'intra':
                                        contact[1] = go_sigma
                                        contact[2] = -go_epsilon
                                        contact[3] = [f"intra molecular go bond {dist}"]

                    elif not vs_already_added:
                        if counter == 0:
                            sigma = LJ_sigma
                            epsilon = -LJ_epsilon
                            meta = ['counter LJ']
                        elif counter == 1:
                            if bond_type == 'inter' or bond_type == 'other':
                                sigma = go_sigma
                                epsilon = go_epsilon
                                meta = [f"inter molecular go bond {dist}"]
                            else:
                                sigma = LJ_sigma
                                epsilon = LJ_epsilon
                                meta = ['']
                        elif counter == 2:
                            if bond_type == 'intra':
                                sigma = go_sigma
                                epsilon = go_epsilon
                                meta = [f"intra molecular go bond {dist}"]
                            else:
                                sigma = LJ_sigma
                                epsilon = LJ_epsilon
                                meta = ['']
                        elif counter == 3:
                            if bond_type == 'intra':
                                sigma = go_sigma
                                epsilon = -go_epsilon
                                meta = [f"intra molecular go bond {dist}"]
                            else:
                                sigma = LJ_sigma
                                epsilon = -LJ_epsilon
                                meta = ['']

                        contact_bias_list.append([(virtual_site_a, virtual_site_b), sigma, epsilon, meta])

            for atoms, sigma, epsilon, meta in contact_bias_list:
                contact_bias = NonbondParam(atoms=atoms, sigma=sigma, epsilon=epsilon, meta={"comment": meta})
                self.system.gmx_topology_params["nonbond_params"].append(contact_bias)


    def run_molecule(self, molecule):

        self.res_graph = make_residue_graph(molecule)
        # compute the contacts; this also creates
        # the exclusions
        self.contact_selector(molecule)
        # compute the interaction parameters
        self.compute_go_interaction()
        return molecule

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        self.system = system
        if self.method == 0:
            super().run_system(system)
        elif self.method == 1:
            self.all_molecule_system = self.all_molecule_system.molecules[0]
            self.all_molecule_graph = make_residue_graph(self.all_molecule_system, go=True)
            # compute the contacts; this also creates
            # the exclusions
            self.contact_selector(system)
            # super().run_system(system)
            self.compute_go_interaction()
