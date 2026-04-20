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

from .processor import Processor
from .annotate_idrs import parse_residues
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _in_chain_and_resid_region
import numpy as np

class IDRInteractionOptimising(Processor):
    def __init__(self, 
                 go, 
                 elastic,
                 id_regions,
                 elastic_res_distance=None):
        """
        Parameters
        ----------
        go: bool
            this system contains a go model
        elastic: bool
            this system contains an elastic network
        id_regions: list
            list of tuples of residue regions defining the IDRs
        system: vermouth.system.System
            the system of the molecules is used for
            storing the nonbonded parameters
        """
        self.go = go
        self.elastic = elastic
        self.id_regions = []
        for region in id_regions:
            self.id_regions.append(parse_residues(region))
        self.system = None
        self.elastic_res_distance = elastic_res_distance
    def remove_cross_nb_interactions(self, molecule, res_graph):
        """
        Remove Go interactions between folded and disordered regions of a molecule

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule
        res_graph: :class:`networkx.Graph`
            the residue graph of the molecule
        """
        #list of all the Go pairs in the molecule
        all_go_pairs = np.array([list(i.atoms) for i in self.system.gmx_topology_params["nonbond_params"] if 'W' not in list(i.atoms)])
        # list to record which items we don't want. cross = go potential between folded and disordered domain.
        all_cross_pairs = []
        # for each IDR that we have
        for region in self.id_regions:
            # for each residue in the molecule
            for res_node in res_graph.nodes:
                # get the current resid, the original one, and the chain
                resid = res_graph.nodes[res_node]['resid']
                _old_resid = res_graph.nodes[res_node]['stash']['resid']
                chain = res_graph.nodes[res_node]['chain']
                # if the region is in the original resid and chain
                if _in_chain_and_resid_region(region, _old_resid, chain):
                    # get the 
                    vs_go_node = next(get_go_type_from_attributes(res_graph.nodes[res_node]['graph'],
                                                                  resid=resid,
                                                                  chain=chain,
                                                                  prefix=molecule.meta.get('moltype')))
                    all_cross_pairs.append(np.where(all_go_pairs == vs_go_node)[0]) #just need the first one

        # make sure we only have one entry in case a site has more than one interaction
        all_cross_pairs = np.unique([x for xs in all_cross_pairs for x in xs])

        # delete the folded-disordered Go interactions from the list.
        # go backwards otherwise list order gets messed up.
        for i in reversed(all_cross_pairs):
            del self.system.gmx_topology_params["nonbond_params"][i]

    def remove_cross_elastic(self, molecule, elastic_bond_residue_distance):
        """
        Remove elastic network bonds between folded and disordered regions of a molecule

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule
        res_graph: :class:`networkx.Graph`
            the residue graph of the molecule
        """

        # list of all atom pairs with bonds between them in the molecule
        all_bonds = np.array([i.atoms for i in molecule.get_interaction('bonds')])

        # make a map between bond indicies and their residues. index starts from 1 so -1 for idx
        nodes_map = {i: {"resid": molecule.nodes[j].get('stash').get('resid'),
                                "chain": molecule.nodes[j].get('chain')
                                }
                    for i,j in enumerate(molecule.nodes, 1)}
        
        # list of bonds, but which residues they link not nodes
        residue_bonds = np.array([[nodes_map[i].get('resid') for i in j] for j in all_bonds])
        # same but for chains
        chains = np.array([[nodes_map[i].get('chain') for i in j] for j in all_bonds])
        
        # mask of whether a bond is elastic or not.
        elastic = np.diff(residue_bonds, axis=1).ravel()>=elastic_bond_residue_distance
        # need (n,2) shape array to use np.where
        is_elastic = np.stack((elastic, elastic)).T
        idx = np.arange(len(all_bonds))

        # list to record which items we don't want. cross = go potential between folded and disordered domain.
        all_cross_pairs = []

        for region in self.id_regions:
            # get info about the region
            lower, upper = sorted(region['resids'][0])
            chain = region['chain']

            removal_idx = np.where((residue_bonds >= lower) &       # residues >= lower residue bound of region 
                                   (residue_bonds <= upper) &       # residues <= upper residue bound of region
                                   (is_elastic == True) &           # the bond is categorised as elastic by residue distance
                                   (chains == chain)  # chain is correct
                                   )[0]                             # only need the first index of the search
            
            # make a mask within the bonds list for bonds that need to be removed
            bonds_idx = np.array([True if i in removal_idx else False for i in idx])
            # get the bonds that need to be removed
            for_removal = all_bonds[bonds_idx] 
            all_cross_pairs.extend(for_removal)

        # delete the folded-disordered elastic network bonds from the list
        # go backwards otherwise list order gets messed up.
        for i in reversed(all_cross_pairs):
            molecule.remove_interaction('bonds', tuple(i))


    def run_molecule(self, molecule):
        if not self.system:
            raise IOError('This processor requires a system.')

        if hasattr(molecule, 'res_graph'):
            res_graph = molecule.res_graph
        else:
            res_graph = make_residue_graph(molecule)

        if self.go:
            self.remove_cross_nb_interactions(molecule=molecule, res_graph=res_graph)
        elif self.elastic:
            self.remove_cross_elastic(molecule=molecule, elastic_bond_residue_distance=self.elastic_res_distance)

        return molecule

    def run_system(self, system):
        # no disordered regions, no bother
        if not self.id_regions:
            return system
        self.system = system
        super().run_system(system)

