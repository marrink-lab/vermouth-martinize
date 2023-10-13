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
from ..molecule import Interaction
from ..processors.processor import Processor
from ..selectors import filter_minimal, select_backbone
from .go_utils import get_go_type_from_attributes

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

    ** Subclassing **
    In order to customize the Go-model structural bias
    it is recommended to subclass this function and
    overwrite the .contact_selector method and/or
    the .compute_bias method. This subclassed Processor
    then has to be added to the into the /bin/martinize2
    pipline in place of the StructuralBiasWriter or as
    replacement in the GoPipeline.
    """
    def __init__(self,
                 contact_map,
                 cutoff_short,
                 cutoff_long,
                 go_eps,
                 res_dist,
                 res_graph=None,
                 prefix="VS"):
        """
        Initialize the Processor with arguments required
        to setup the Go model structural bias.

        Parameters
        ----------
        contact_map: list[(str, int, str, int)]
            list of contacts defined as by the chain
            identifier and residue index
        cutoff_short: float
            distances smaller than this are ignored
        cutoff_long: float
            distances larger than this are ignored
        go_eps: float
            epsilon value of the structural bias in
            kJ/mol
        res_dist: int
            if nodes are closer than res_dist along
            the residue graph they are ignored; this
            is similar to sequence distance but takes
            into account disulfide bridges for example
        res_graph: :class:vermouth.Molecule
            residue graph of the molecule; if None it
            get's generated automatically
        prefix: str
            prefix for all virtual-site atomtypes
        """
        self.contact_map = contact_map
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.res_dist = res_dist
        self.prefix = prefix
        self.res_graph = res_graph
        # don't modify
        self.__chain_id_to_resnode = {}

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
            return self.__chain_id_to_resnode

        # for each residue collect the chain and residue in a dict
        # we use this later for identifying the residues from the
        # contact map
        for resnode in self.res_graph.nodes:
            chain = self.res_graph.nodes[resnode].get('chain', None)
            # in vermouth within a molecule all resid are unique
            # when merging multiple chains we store the old resid
            # the go model always references the input resid i.e.
            # the _old_resid
            resid = self.res_graph.nodes[resnode].get('_old_resid')
            self.__chain_id_to_resnode[(chain, resid)] = resnode
        return self.__chain_id_to_resnode

    def contact_selector(self, molecule):
        """
        Select all contacts from the contact map
        that according to their distance and graph
        connectivity are elegible to form a Go
        bond and create exclusions between the
        backbone beads of those contacts.

        Parameters
        ----------
        molecule: :class:vermouth.Molecule

        Returns
        -------
        list[(abc.hashable, abc.hashable, float)]
            list of node keys and distance
        """
        # distance_matrix of elegible pairs as tuple(node, node, dist)
        contact_matrix = []
        # find all pairs of residues that are within bonded distance of
        # self.res_dist
        connected_pairs = dict(nx.all_pairs_shortest_path_length(self.res_graph,
                                                            cutoff=self.res_dist))
        for contact in self.contact_map:
            resIDA, chainA, resIDB, chainB = contact
            # identify the contact in the residue graph based on
            # chain ID and resid
            resA = self.chain_id_to_resnode[(chainA, resIDA)]
            resB = self.chain_id_to_resnode[(chainB, resIDB)]
            # make sure that both residues are not connected
            # note: contacts should be symmteric so we only
            # check against one
            if resB not in connected_pairs[resA]:
                # now we lookup the backbone nodes within the residue contact
                bb_node_A = next(filter_minimal(self.res_graph.nodes[resB], select_backbone))
                bb_node_B = next(filter_minimal(self.res_graph.nodes[resB], select_backbone))
                # compute the distance between bb-beads
                dist = np.linalg.norm(molecule.nodes[bb_node_A]['position'] -
                                      molecule.nodes[bb_node_B]['position'])
                # verify that the distance between BB-beads satisfies the
                # cut-off criteria
                if self.cut_off_large > dist > self.cutoff_short:
                    # find the go virtual-sites for this residue
                    # probably can be done smarter but mehhhh
                    atype_a = get_go_type_from_attributes(molecule,
                                                          resid=resIDA,
                                                          chain=chainA,
                                                          prefix=self.prefix)
                    atype_b = get_go_type_from_attributes(molecule,
                                                          resid=resIDB,
                                                          chain=chainB,
                                                          prefix=self.prefix)
                    # generate backbone backbone exclusions
                    # perhaps one day can be it's own function
                    excl = Interaction(atoms=(bb_node_A, bb_node_B), parameters=[], meta={"group": "Go model exclusion"})
                    molecule.interactions['exclusions'].append(excl)

                    contact_matrix.append((atype_a, atype_b, dist))
        return contact_matrix

    def compute_go_interaction(self, contacts):
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
        go_inters = {}
        for atype_a, atype_b, dist in contacts:
            # compute the LJ sigma paramter for this contact
            # 1.12246204830 is a magic number by Sebastian
            sigma = dist / 1.12246204830
            # find the go virtual-sites for this residue
            # probably can be done smarter but mehhhh
            go_inters[frozenset(atype_a, atype_b)] = (sigma, self.go_eps)
        return go_inters

    def run_system(self, system):
        """
        Process `system`.

        Parameters
        ----------
        system: vermouth.system.System
            The system to process. Is modified in-place.
        """
        for molecule in system.molecules:
            # compute the contacts; this also creates
            # the exclusions
            contacts = self.contact_selector(molecule)
            # compute the interaction parameters
            inters = self.compute_go_interaction(contacts)
            # update the contacts for the system
            self.system.gmx_topology_params.upadte(inters)
