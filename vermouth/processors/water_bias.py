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

from .processor import Processor
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _get_bead_size, _in_chain_and_resid_region
from ..gmx.topology import NonbondParam
from .annotate_idrs import parse_residues
import numpy as np

class ComputeWaterBias(Processor):
    """
    Processor which computes the water  bias for
    the Martini Go and Martini IDP model.

    The water bias strength is defined per secondary
    structure element in `water_bias` and assigned if
    `auto_bias` is set to True. Using the `idr_regions`
    argument the water_bias can be changed for
    intrinsically disordered regions (IDRs). The IDR
    bias superseeds the auto bias.

    This Processor updates the system.gmx_topology_params
    attribute.

    Subclassing
    -----------
    If the procedure by which to assign the water bias is
    to be changed this processor is best subclassed and the
    assign_residue_water_bias method overwritten.
    """

    def __init__(self,
                 auto_bias,
                 water_bias,
                 idr_regions):
        """


        Parameters
        ----------
        auto_bias: bool
            apply the automatic secondary structure
            dependent water biasing
        water_bias: dict[str, float]
            a dict of secondary structure codes and
            epsilon value for the water bias in kJ/mol
        idr_regions: list
            list of tuples of residue regions defining the IDRs
        prefix: str
            prefix of the Go virtual-site atomtypes
        system: vermouth.system.System
            the system of the molecules is used for
            storing the nonbonded parameters
        """
        self.water_bias = water_bias
        self.auto_bias = auto_bias
        self.idr_regions = []
        for region in idr_regions:
            self.idr_regions.append(parse_residues(region))
        self.system = None

    def assign_residue_water_bias(self, molecule, res_graph):
        """
        Assign the residue water bias for all residues
        with a secondary structure element or that are
        defined by the region selector. Region selectors
        supercede the auto assignment.

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule
        res_graph: :class:`vermouth.molecule.Molecule`
            the residue graph of the molecule
        """
        for res_node in res_graph.nodes:
            resid = res_graph.nodes[res_node]['resid']
            _old_resid = res_graph.nodes[res_node]['_old_resid']
            chain = res_graph.nodes[res_node]['chain']
            resname = res_graph.nodes[res_node]['resname']
            eps = 0.0

            if self.auto_bias:
                sec_struc = res_graph.nodes[res_node]['cgsecstruct']
                eps = self.water_bias.get(sec_struc, 0.0)
            for region in self.idr_regions:
                if _in_chain_and_resid_region(region, _old_resid, chain):
                    eps = self.water_bias.get('idr', 0.0)
                    sec_struc = res_graph.nodes[res_node]['cgsecstruct']
            if abs(eps) <= 1e-12:
                continue

            vs_go_node = next(get_go_type_from_attributes(res_graph.nodes[res_node]['graph'],
                                                          resid=resid,
                                                          chain=chain,
                                                          prefix=molecule.meta.get('moltype')))

            # what is the blocks bb-type
            bb_type = molecule.force_field.blocks[resname].nodes['BB']['atype']
            size = _get_bead_size(bb_type)
            # bead sizes are defined in the force-field file as
            # regular, small and tiny
            sigma = float(molecule.force_field.variables[size])
            # update interaction parameters
            atoms = (molecule.force_field.variables['water_type'], vs_go_node)
            water_bias = NonbondParam(atoms=atoms,
                                      sigma=sigma,
                                      epsilon=eps,
                                      meta={"comment": ["water bias", sec_struc]})
            self.system.gmx_topology_params["nonbond_params"].append(water_bias)

    def remove_cross_nb_interactions(self, molecule, res_graph):
        """
        Remove Go bonds between folded and disordered regions of a molecule

        Parameters
        ----------
        molecule: :class:`vermouth.molecule.Molecule`
            the molecule
        res_graph: :class:`vermouth.molecule.Molecule`
            the residue graph of the molecule
        """
        #list of all the Go pairs in the molecule
        all_go_pairs = np.array([list(i.atoms) for i in self.system.gmx_topology_params["nonbond_params"] if 'W' not in list(i.atoms)])
        # list to record which items we don't want. cross = go potential between folded and disordered domain.
        all_cross_pairs = []

        for region in self.idr_regions:
            for res_node in res_graph.nodes:
                resid = res_graph.nodes[res_node]['resid']
                _old_resid = res_graph.nodes[res_node]['_old_resid']
                chain = res_graph.nodes[res_node]['chain']
                if _in_chain_and_resid_region(region, _old_resid, chain):
                    vs_go_node = next(get_go_type_from_attributes(res_graph.nodes[res_node]['graph'],
                                                                  resid=resid,
                                                                  chain=chain,
                                                                  prefix=molecule.meta.get('moltype')))
                    all_cross_pairs.append(np.where(all_go_pairs == vs_go_node)[0]) #just need the first one

        # make sure we only have one entry in case a site has more than one interaction
        all_cross_pairs = np.unique([x for xs in all_cross_pairs for x in xs])
        # delete the folded-disordered Go interactions from the list going backwards.
        # otherwise list order gets messed up.
        for i in reversed(all_cross_pairs):
            del self.system.gmx_topology_params["nonbond_params"][i]

    def run_molecule(self, molecule):
        """
        Assign water bias for a single molecule
        """
        if not self.system:
            raise IOError('This processor requires a system.')

        if not molecule.meta.get('moltype'):
            raise ValueError('The molecule does not have a moltype name.')

        if hasattr(molecule, 'res_graph'):
            res_graph = molecule.res_graph
        else:
            res_graph = make_residue_graph(molecule)

        self.assign_residue_water_bias(molecule, res_graph)
        self.remove_cross_nb_interactions(molecule, res_graph)

        return molecule

    def run_system(self, system):
        """
        Assign the water bias of the Go model to file. Biasing
        is always molecule specific i.e. no two different
        vermouth molecules can have the same bias.

        Parameters
        ----------
        system: :class:`vermouth.system.System`
        """
        if not (self.idr_regions or self.auto_bias):
            return system
        self.system = system
        super().run_system(system)
