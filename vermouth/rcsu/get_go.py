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
Process the Go contact pairs.
"""
import numpy as np
import networkx as nx
from vermouth.molecule import Interaction

def _get_bb_nodes(molecule, nodes):
    for node in nodes:
        if molecule.nodes[node]['atomname'] == "BB":
            return node
    return None

def _get_go_type(molecule, resid, chain, prefix):
    for node in molecule.nodes:
        attrs = molecule.nodes[node]
        if attrs['resid'] == resid and attrs['chain'] == chain:
            if prefix in attrs['atype']:
                return attrs['atype']
    raise IOError(f"Could not find GoVs with resid {resid} in chain {chain}.")

def _in_region(resid, regions):
    """
    Check if resid falls in regions interval.
    """
    for low, up in regions:
        if low <= resid <= up:
            return True
    return False

class GetGo():
    """
    Generate the Go model interaction parameters.
    """
    def __init__(self,
                 contact_map,
                 cutoff_short,
                 cutoff_long,
                 go_eps,
                 res_dist,
                 res_graph,
                 regions=None,
                 water_bias=None,
                 idp_water_bias=None,
                 prefix="VS"):

        self.contact_map = contact_map
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.res_dist = res_dist
        self.itp_file = open('GoVirtIncludes.itp', "w", encoding='UTF-8')
        self.prefix = prefix
        self.res_graph = res_graph
        self.water_bias = water_bias
        self.idp_water_bias = idp_water_bias
        self.regions = regions

    def write_go_structural_bias(self, molecule):
        chain_id_to_resnode = {}
        # find all pairs of residues that are within bonded distance of
        # self.res_dist
        connected_pairs = dict(nx.all_pairs_shortest_path_length(self.res_graph,
                                                            cutoff=self.res_dist))

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
            chain_id_to_resnode[(chain, resid)] = resnode

        for contact in self.contact_map:
            resIDA, chainA, resIDB, chainB = contact
            # identify the contact in the residue graph based on
            # chain ID and resid
            resA = chain_id_to_resnode[(chainA, resIDA)]
            resB = chain_id_to_resnode[(chainB, resIDB)]
            # make sure that both residues are not connected
            # note: contacts should be symmteric so we should be
            # able to get rid of the second if clause
            if resB not in connected_pairs[resA] and resA not in connected_pairs[resB]:
                # now we lookup the backbone nodes within the residue contact
                bb_node_A = _get_bb_nodes(molecule, self.res_graph.nodes[resA]['graph'].nodes)
                bb_node_B = _get_bb_nodes(molecule, self.res_graph.nodes[resB]['graph'].nodes)
                # compute the distance between bb-beads
                dist = np.linalg.norm(molecule.nodes[bb_node_A]['position'] -
                                      molecule.nodes[bb_node_B]['position'])
                # verify that the distance between BB-beads satisfies the
                # cut-off criteria
                if dist > self.cutoff_short or dist < self.cutoff_long:
                    # compute the LJ sigma paramter for this contact
                    sigma = dist / 1.12246204830
                    # find the go virtual-sites for this residue
                    # probably can be done smarter but mehhhh
                    nodeA = _get_go_type(molecule, resid=resIDA, chain=chainA, prefix=self.prefix)
                    nodeB = _get_go_type(molecule, resid=resIDB, chain=chainB, prefix=self.prefix)
                    # write itp file with go interaction parameters
                    self.itp_file.write(f"{nodeA} {nodeB} 1 {sigma:3.8F} {self.go_eps:3.8F} ; structural Go bias\n")
                    # generate the exclusions between pairs of bb beads
                    excl = Interaction(atoms=(bb_node_A, bb_node_B), parameters=[], meta={"group": "Go model exclusion"})
                    molecule.interactions['exclusions'].append(excl)

    def write_water_bias_auto(self, molecule):
        """
        Automatically calculate the water bias depending on the secondary structure.
        """
        self.itp_file.write('; additional Lennard-Jones interaction between virtual BB bead and W\n')
        for res_node in self.res_graph.nodes:
            resid = self.res_graph.nodes[res_node]['resid']
            chain = self.res_graph.nodes[res_node]['chain']
            vs_go_node = _get_go_type(molecule, resid=resid, chain=chain, prefix=self.prefix)
            sec_struc = self.res_graph.nodes[res_node]['sec_struc']
            eps = self.water_bias.get(sec_struc, 0.0)
            if self.res_graph.nodes[res_node]['resname'] in ['GLY', 'ALA', 'VAL', 'PRO']:
                sigma = 0.430
            else:
                sigma = 0.470

            self.itp_file.write(f"W {vs_go_node} 1 {sigma:3.8F} {eps:3.8F} ; secondary structure water bias\n")

    def write_water_bias_idr(self, molecule):
        """
        Write bias for intrinsically disordered domains from manual selection.
        """
        self.itp_file.write('; additional Lennard-Jones interaction between virtual BB bead and W\n')
        for res_node in self.res_graph.nodes:
            if _in_region(self.res_graph.nodes[res_node], self.regions):
                resid = self.res_graph.nodes[res_node]['resid']
                chain = self.res_graph.nodes[res_node]['chain']
                vs_go_node = _get_go_type(molecule, resid=resid, chain=chain, prefix=self.prefix)
                eps = self.idp_water_bias
                if self.res_graph.nodes[res_node]['resname'] in ['GLY', 'ALA', 'VAL', 'PRO']:
                    sigma = 0.430
                else:
                    sigma = 0.470

                self.itp_file.write(f"W {vs_go_node} 1 {sigma:3.8F} {eps:3.8F} ; idp water bias\n")

    def run_system(self, system):
        for molecule in system.molecules:
            self.write_go_structural_bias(molecule)
            if self.water_bias:
                self.write_water_bias_auto(self, molecule)

        self.itp_file.close()
