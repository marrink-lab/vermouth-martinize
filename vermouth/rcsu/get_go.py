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
from vermouth.processors.apply_rubber_bands import self_distance_matrix

def extract_position_matrix(molecule, selector):
    """
    Given a selector extract a distance matrix from
    a molecule.
    """
    selection = []
    coordinates = []
    missing = []
    node_to_idx = {}
    idx_to_node = {}
    for node_idx, (node_key, attributes) in enumerate(molecule.nodes.items()):
        node_to_idx[node_key] = node_idx
        idx_to_node[node_idx] = node_key
        if selector(attributes):
            selection.append(node_idx)
            coordinates.append(attributes.get('position'))
            if coordinates[-1] is None:
                missing.append(node_key)
        node_idx += 1

    if missing:
        raise ValueError('All atoms from the selection must have coordinates. '
                         'The following atoms do not have some: {}.'
                         .format(' '.join(missing)))

    if not coordinates:
        return

    coordinates = np.stack(coordinates)
    if np.any(np.isnan(coordinates)):
        LOGGER.warning("Found nan coordinates in molecule {}. "
                       "Will not generate an EN for it. ",
                       molecule.moltype,
                       type='unmapped-atom')
        return

    distance_matrix = self_distance_matrix(coordinates)
    return distance_matrix 

def _get_bb_pos(moleule, nodes):
    for node in molecule.nodes:
        if molecule.nodes[node]['atomname'] == "BB":
            return molecule.nodes[node]['position']
    return None

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
                 domain,
                 selector):
        self.contact_map = contact_map
        self.cutoff_short = cutoff_short
        self.cutoff_long = cutoff_long
        self.go_eps = go_eps
        self.res_dist = res_dist
        self.domain = domain
        self.selector = selector

    def run_molecule(molecule):
        res_graph = molecule.residue_graph
        chain_id_to_resnode = {}
        for resnode in res_graph.nodes:
            chain = res_graph.nodes[resnode].get('chain', None)
            resid = res_graph.nodes[resnode].get('resid')
            chain_id_to_resnode[(chain, resid)] = resnode
            
        # compute the go parameters
        for chainA, resA, chainB, resB in self.contact_map:
            resA = chain_id_to_resnode[(chainA, resA)]
            resB = chain_id_to_resnode[(chainB, resB)]
            if graph_distance(resA, resB) > self.res_dist:
                posA = _get_bb_pos(molecule, res_graph.nodes[resA]['graph'].nodes)
                posB = _get_bb_pos(molecule, res_graph.nodes[resA]['graph'].nodes)
                dist = np.linalg.norm(posA, posB)
                if dist > self.cutoff_short or dist < self.cutoff_large:
                    sigma = dist / 1.12246204830
                    Vii = 4.0 * pow(sigma, 6) * self.go_eps
                    Wii = 4.0 * pow(sigma, 12) * self.go_eps





