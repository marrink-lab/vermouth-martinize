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

from itertools import compress
from .processor import Processor
from .annotate_idrs import parse_residues
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _in_chain_and_resid_region
import numpy as np
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

class IDRCrossDomainInteractionRemoval(Processor):
    def __init__(self, 
                 go, 
                 elastic,
                 id_regions):
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

    def remove_cross_nb_interactions(self, molecule, res_graph, nbparams):
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
        all_go_pairs = np.array([list(i.atoms) for i in nbparams if 'W' not in list(i.atoms)])
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

        return all_cross_pairs

    def remove_cross_elastic(self, molecule):
        """
        Remove elastic network bonds between folded and disordered regions of a molecule

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule
        res_graph: :class:`networkx.Graph`
            the residue graph of the molecule
        """
        # find elastic bonds by meta
        elastic_bonds = [list(i.atoms) for i in molecule.interactions['bonds'] if i.meta.get('group') == "Rubber band"]
        print(elastic_bonds)
        # list to record which items we don't want. cross = elastic bond between folded and disordered domain.
        all_cross_pairs = []
        # make a map between bond indicies and their residues. index starts from 1 so -1 for idx        
        nodes_map = {i: {"resid": molecule.nodes[j].get('stash').get('resid'),                                
                         "chain": molecule.nodes[j].get('chain')                                
                         }                    
                         for i,j in enumerate(molecule.nodes, 1)}
                
        # list of bonds, but which residues they link not nodes
        residue_bonds = [[nodes_map[i].get('resid') for i in j] for j in elastic_bonds]
        # same but for chains
        chains = [[nodes_map[i].get('chain') for i in j] for j in elastic_bonds]
        
        for region in self.id_regions:
            # get info about the region
            lower, upper = sorted(region['resids'][0])
            chain = region['chain']
            # get bonds and chain in molecule which match this region
            bond_in_region = [any([(i>= lower) & (i<= upper) for i in j]) for j in residue_bonds]
            chain_in_region = [any([i == chain for i in j]) for j in chains]
            # list of which residue bonds meet both criteria
            bonds_rm_idx = [all([i,j]) for i,j in zip(bond_in_region, chain_in_region)]

            # get the bonds that need to be removed
            for_removal = compress(elastic_bonds, bonds_rm_idx) 
            all_cross_pairs.extend(for_removal)

        # delete the folded-disordered elastic network bonds from the list
        # go backwards otherwise list order gets messed up.
        for i in reversed(all_cross_pairs):
            molecule.remove_interaction('bonds', tuple(i))


    def _remove_nb_interactions(self, system, interactions_list):
        """
        remove non bonded interactions at the system level
        """
        # delete the folded-disordered Go interactions from the list.
        # go backwards otherwise list order gets messed up.
        for i in reversed(interactions_list):
            del system.gmx_topology_params["nonbond_params"][i]


    def run_system(self, system):
        # no disordered regions, no bother
        if not self.id_regions:
            return system

        for molecule in system.molecules:

            if self.go:
                if hasattr(molecule, 'res_graph'):
                    res_graph = molecule.res_graph
                else:
                    res_graph = make_residue_graph(molecule)

                interactions_for_removal = self.remove_cross_nb_interactions(molecule=molecule, 
                                                                             res_graph=res_graph,
                                                                             nbparams=system.gmx_topology_params["nonbond_params"])
                self._remove_nb_interactions(system, interactions_for_removal)

            elif self.elastic:
                self.remove_cross_elastic(molecule=molecule)
