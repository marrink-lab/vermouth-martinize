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
    The .run_molecule function only generates the 
    dictionary self.molecule_graphs where the keys
    are the mol_idx and the values the molecules of
    the system as nx.Graphs. The computed structural 
    bias parameters are stored in 
    `system.gmx_topology_params` and can be written 
    out using the `vermouth.gmx.write_topology`
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
                 cutoff_short_inter,
                 cutoff_long_inter,
                 go_eps_inter,
                 cutoff_short_intra,
                 cutoff_long_intra,
                 go_eps_intra,
                 res_dist,
                 moltype,
                 go_anchor_bead,
                 go_ff
                 ):
        """
        Initialize the Processor with arguments required
        to setup the Go model structural bias.

        Parameters
        ----------
        contact_map: list[(str, int, int, str, int, int)]
            list of contacts defined as by the chain
            identifier, residue index and molecule index
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
        system: :class:`vermouth.system.System`
            the system
        magic_number: float
            magic number for Go contacts from the old
            GoVirt script.
        backbone: str
            name of backbone atom where virtual site is placed
        go_ff: Path
            file name where the martini3 non bonded interactions are stored
        """
        self.cutoff_short_inter=cutoff_short_inter
        self.cutoff_long_inter=cutoff_long_inter
        self.go_eps_inter=go_eps_inter
        self.cutoff_short_intra=cutoff_short_intra
        self.cutoff_long_intra=cutoff_long_intra
        self.go_eps_intra=go_eps_intra
        self.res_dist = res_dist
        self.moltype = moltype
        self.backbone = go_anchor_bead
        self.ff_file = go_ff
        # don't modify
        self.molecule_graphs = {}
        self.symmetrical_matrix = []
        self.lennard_jones = []
        self.__chain_id_to_resnode = {}
        self.conversion_factor = 2**(1/6)

    # do not overwrite when subclassing
    def _chain_id_to_resnode(self, chain, resid, molid):
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
        molid: int
            molecule index

        Returns
        -------
        dict
            a dict matching the chain,resid,molid to the node of the 
            corresponding graph in self.molecule_graphs
        """
        if self.__chain_id_to_resnode:
            if self.__chain_id_to_resnode.get((chain, resid, molid), None) is not None:
                return self.__chain_id_to_resnode[(chain, resid, molid)]
            else:
                LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')

        # for each residue collect the chain and residue in a dict
        # we use this later for identifying the residues from the
        # contact map

        for mol_id, molecule_graph in self.molecule_graphs.items():
            for resnode in molecule_graph.nodes:
                chain_key = molecule_graph.nodes[resnode].get('chain', None)
                resid_key = molecule_graph.nodes[resnode].get('_old_resid')
                mol_key = molecule_graph.nodes[resnode].get('mol_idx')
                if mol_key != mol_id:
                    print('error, mol index is not the same')
                self.__chain_id_to_resnode[(chain_key, resid_key, mol_key)] = resnode

        if self.__chain_id_to_resnode.get((chain, resid, molid), None) is not None:
            return self.__chain_id_to_resnode[(chain, resid, molid)]
        else:
            LOGGER.debug(stacklevel=5, msg='chain-resid pair not found in molecule')


    def contact_selector(self):
        """
        Select all contacts from the contact map
        that according to their distance and graph
        connectivity are eligible to form a Go
        bond and create exclusions between the
        backbone beads of those contacts where needed.
        Appends (vs_a, vs_b, beadA, beadB, bond_type, dist)
        to the self.symmetrical_matrix. where the vs_a is 
        a list of the 4 virtual sites corresponding to 
        'bead a' and vice versa vor vs_b. beadA and beadB
        are the beadtypes of the backbone bead. Useful for
        looking up the normal martini3 interaction, 
        for cancellation of that interaction. bond_type is
        intra, inter or other. intra-molecular inter-molecular
        or inter-molecular (between non-homogeneous molecules), 
        respectively. dist is the distance between the backbone
        beads.
        """

        mol_id_map = self.system.go_params['unique_mols_map']
        reference_mol_map = self.system.go_params['reference_mol_map']
        added_contacts = {}
        added_exclusions = {}

        # distance_matrix of eligible pairs as tuple(node, node, dist)
        contact_matrix = []

        connected_pairs = {}

        # make connected pairs so the contacts are not selected if they are 
        # in self.resdist 'graph distance' of each other
        for mol_idx, mol_graph in self.molecule_graphs.items():
            connected_pairs_mol = dict(nx.all_pairs_shortest_path_length(mol_graph, cutoff=self.res_dist))
            connected_pairs[mol_idx] = connected_pairs_mol

        bad_chains_warning = False

        # go through all contacts that were generated or read in contact_map.py
        for contact in self.system.go_params["go_map"][0]:
            # identify what the node ids (resA, resB) are in their respective molecule graphs
            # with the self._chain_id_to_resnode() function/dictionary
            resIDA, chainA, molIDA, resIDB, chainB, molIDB = contact
            molIDA, molIDB = int(molIDA), int(molIDB)
            resA = self._chain_id_to_resnode(chainA, resIDA, molIDA)
            resB = self._chain_id_to_resnode(chainB, resIDB, molIDB)
            if (resA is not None) and (resB is not None):
                if molIDA != molIDB or (molIDA == molIDB and resB not in connected_pairs[molIDA][resA]):
                    # now we lookup the backbone nodes within the residue contact
                    try:
                        bb_node_A = next(filter_minimal(self.molecule_graphs[molIDA].nodes[resA]['graph'],
                                                        select_backbone,
                                                        bb_atomname=self.backbone))
                        bb_node_B = next(filter_minimal(self.molecule_graphs[molIDB].nodes[resB]['graph'],
                                                        select_backbone,
                                                        bb_atomname=self.backbone))
                        
                        # here get the bead names of the backbones, to later know which LJ
                        # sigma and epsilon are related to this bond
                        graphA = self.molecule_graphs[molIDA].nodes[resA]['graph']
                        for atom_id, atom_attrs in graphA.nodes.items():
                            if atom_attrs.get('atomname') == 'BB':
                                if 'atype' in atom_attrs:
                                    backbone_atype = atom_attrs['atype']
                                    beadA = backbone_atype

                        graphB = self.molecule_graphs[molIDB].nodes[resB]['graph']
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

                    # compute distance between nodes/backbone beads
                    dist = np.linalg.norm(self.system.molecules[molIDA].nodes[bb_node_A]['position'] -
                                        self.system.molecules[molIDB].nodes[bb_node_B]['position'])
                    
                    # set the low and high cutoff, based on the bondtype of the contact
                    if molIDA == molIDB:
                        low = self.cutoff_short_intra
                        high = self.cutoff_long_intra
                        bond_type = 'intra'
                    elif molIDA != molIDB: 
                        low = self.cutoff_short_inter
                        high = self.cutoff_long_inter
                        if molIDA in mol_id_map[molIDB]:
                            bond_type = 'inter'
                        else:
                            bond_type = 'other'

                    # check if the distance satisfies the cutoff conditions
                    if high > dist > low:
                        if self.molecule_graphs[molIDA]:
                            vs_a, excl_a = get_go_type_from_attributes(self.molecule_graphs[molIDA].nodes[resA]['graph'],
                                                                    _old_resid=resIDA,
                                                                    chain=chainA,
                                                                    prefix=self.moltype)
                        else:
                            print(f'Warning there no graph generated with mol_id: {molIDA}')

                        if self.molecule_graphs[molIDB]:
                            vs_b, excl_b = get_go_type_from_attributes(self.molecule_graphs[molIDB].nodes[resB]['graph'],
                                                                    _old_resid=resIDB,
                                                                    chain=chainB,
                                                                    prefix=self.moltype)
                        else:
                            print(f'Warning there no graph generated with mol_id: {molIDB}')
                        
                        # here the contact (vs_b, vs_a, beadB, beadA, dist) is checked, if the
                        # contact was already seen before (in contact_matrix)
                        if (molIDB, molIDA, vs_b, vs_a, beadB, beadA, dist) in contact_matrix:

                            if (bond_type, tuple(vs_b), tuple(vs_a), beadB, beadA) in added_contacts:
                                added_contacts[(bond_type, tuple(vs_b), tuple(vs_a), beadB, beadA)].append(dist)
                            elif (bond_type, tuple(vs_a), tuple(vs_b), beadA, beadB) in added_contacts:
                                added_contacts[(bond_type, tuple(vs_a), tuple(vs_b), beadA, beadB)].append(dist)
                            else:
                                added_contacts[(bond_type, tuple(vs_b), tuple(vs_a), beadB, beadA)] = [dist]
                            
                            # generate backbone-backbone exclusions if the contact
                            # is between homologous molecules
                            if bond_type == 'intra' or bond_type == 'inter':
                                ref_molIDA = reference_mol_map[molIDA]
                                molecule = self.system.molecules[ref_molIDA]

                                if ref_molIDA not in added_exclusions:
                                    added_exclusions[ref_molIDA] = []

                                if (excl_a, excl_b) not in added_exclusions[ref_molIDA] and (excl_b, excl_a) not in added_exclusions[ref_molIDA]:
                                    if excl_a != excl_b:
                                        added_exclusions[ref_molIDA].append((excl_a, excl_b))

                            elif bond_type == 'other':
                                pass

                            # The bead types (beadA, beadB) of the backbone bead are added
                            # to the self.lennard_jones list
                            if (beadA, beadB) not in self.lennard_jones and (beadB, beadA) not in self.lennard_jones:
                                self.lennard_jones.append((beadA, beadB))

                        else:
                            contact_matrix.append((molIDA, molIDB, vs_a, vs_b, beadA, beadB, dist))
            else:
                if bad_chains_warning == False:
                    LOGGER.warning("Mismatch between chain IDs in pdb and contact map. This probably means the "
                                   "chain IDs are missing in the pdb and the contact map has all chains = Z.")
                    bad_chains_warning = True

        for mol_id, all_exclusions in added_exclusions.items():
            for exclusions in all_exclusions:
                excl_a, excl_b = exclusions
                excl_site_b = Interaction(atoms=(excl_a[0], excl_b[0]),
                                parameters=[], meta={"group": "Go model exclusion for virtual sites 'b'"})
                excl_site_d = Interaction(atoms=(excl_a[1], excl_b[1]),
                                parameters=[], meta={"group": "Go model exclusion for virtual sites 'd'"})
                molecule.interactions['exclusions'].append(excl_site_b)
                molecule.interactions['exclusions'].append(excl_site_d)

        for keys, dists in added_contacts.items():
            if int(max(dists) - min(dists)) >= 0.2:
                print("Warning: the distance between a pair of backbone beads vary more than 0.2 nm")
            average_dist = sum(dists) / len(dists)
            keys = keys + (average_dist,)
            self.symmetrical_matrix.append(keys)

        return

    def compute_go_interaction(self):
        """
        Compute the sigma value given a distance between
        two nodes, figure out the atomtype name and store
        it in the systems attribute gmx_topology_params.

        First a dict of all bead combinations with go-bonds
        is made where the keys are the bead combination,
        gotten from self.lennard_jones. and the values are
        the tuple(sigma, epsilon) values. 

        self.symetrical_matrix gives a list of tuples:
        (vs_a, vs_b, beadA, beadB, bond_type, dist) here
        the vs_a is a list of the 4 virtual sites corresponding 
        to 'bead a' and vice versa vor vs_b. beadA and beadB
        are the beadtypes of the backbone bead. Useful for
        looking up the normal martini3 interaction, 
        for cancellation of that interaction. bond_type is
        intra, inter or other. intra-molecular inter-molecular
        or inter-molecular (between non-homologous molecules), 
        respectively. dist is the distance between the backbone
        beads.

        The gmx_topology_params for each virtual site are
        set based on the values of these tuples.
        """

        contacts = self.symmetrical_matrix
        LJ_combies = self.lennard_jones
        LJ_sigma_epsilon_dict = {}
        contact_bias_list = []

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

        for bond_type, vs_a, vs_b, beadA, beadB, dist in contacts:
            
            if (beadA, beadB) in LJ_sigma_epsilon_dict:
                LJ_sigma, LJ_epsilon = LJ_sigma_epsilon_dict[(beadA, beadB)]
                LJ_sigma, LJ_epsilon = float(LJ_sigma), float(LJ_epsilon)
            elif (beadB, beadA) in LJ_sigma_epsilon_dict:
                LJ_sigma, LJ_epsilon = LJ_sigma_epsilon_dict[(beadB, beadA)]
                LJ_sigma, LJ_epsilon = float(LJ_sigma), float(LJ_epsilon)
            else:
                print('Warning there is a bead combi that is not in the LJ_sigma_epsilon_dict')
                print(beadA, beadB)

            if bond_type == 'intra':
                go_epsilon = self.go_eps_intra
            elif bond_type == 'inter' or bond_type == 'other':
                go_epsilon = self.go_eps_inter
            go_sigma = dist / self.conversion_factor

            vs_already_added = False
            for counter, (virtual_site_a, virtual_site_b) in enumerate(zip(vs_a, vs_b)):
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

        graph = make_residue_graph(molecule)
        first_node = list(molecule.nodes)[0]
        mol_idx = molecule.nodes[first_node]['mol_idx']
        self.molecule_graphs[mol_idx] = graph
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
        super().run_system(system)
        # compute the contacts; this also creates
        # the exclusions
        self.contact_selector()
        self.compute_go_interaction()
