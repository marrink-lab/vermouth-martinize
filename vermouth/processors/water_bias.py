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

from .processors.processor import Processor
from ..graph_utils import make_residue_graph
from ..rcsu.go_utils import get_go_type_from_attributes, _get_bead_size

class ComputeWaterBias(Processor):
    """
    Processor which computes the water  bias for
    the Martini Go and Martini IDP model.

    The water bias streght is defined per secondary
    structure element in `water_bias` and assinged if
    `auto_bias` is set to True. Using the `idr_regions`
    argument the water_bias can be changed for
    intrinsically disordered regions (IDRs). The IDR
    bias superseeds the auto bias.

    This Processor updates the system.gmx_topology_params
    attribute.

    ** Subclassing **
    If the procedure by which to assign the water bias is
    to be changed this processor is best subclassed and the
    assign_residue_water_bias method overwritten.
    """

    def __init__(self,
                 water_bias,
                 auto_bias,
                 idr_regions,
                 prefix=""):
        """


        Parameters
        ----------
        water_bias: dict[str, float]
            a dict of secondary structure codes and
            epsilon value for the water bias in kJ/mol
        auto_bias: bool
        idr_regions:
            regions defining the IDRs
        prefix: str
            prefix of the Go virtual-site atomtypes
        """
        self.water_bias = water_bias
        self.auto_bias = auto_bias
        self.idr_regions = idr_regions
        self.prefix = prefix

    def assign_residue_water_bias(self, molecule, res_graph):
        """
        Assign the residue water bias for all residues
        with a secondary structure element or that are
        defined by the region selector. Region selectors
        superceed the auto assignement.

        Parameters
        ----------
        molecule: :class:vermouth.Molecule
            the molecule
        res_graph: :class:vermouth.Molecule
            the residue graph of the molecule
        """
        bias_params = {}
        for res_node in self.res_graph.nodes:
            resid = self.res_graph.nodes[res_node]['resid']
            chain = self.res_graph.nodes[res_node]['chain']
            resname = self.res_gaph.nodes[res_node]['resname']

            if self.selector(res_graph, res_node):
                eps = self.water_bias.get('idr', 0.0)
            elif self.auto_bias:
                sec_struc = self.res_graph.nodes[res_node]['sec_struc']
                eps = self.water_bias.get(sec_struc, 0.0)
            else:
                continue

            vs_go_node = get_go_type_from_attributes(res_graph.nodes[res_node],
                                                     resid=resid,
                                                     chain=chain,
                                                     prefix=self.prefix)

            # what is the blocks bb-type
            bb_type = molecule.force_field.blocks[resname]['BB']['atype']
            size = _get_bead_size(bb_type)
            sigma = molecule.force_field.variables['bead_sizes'][size]
            bias_params[frozenset([molecule.force_field.water_name, vs_go_node])] = (sigma, eps)

        return bias_params

    def run_system(self, system):
        """
        Assign the water bias of the Go model to file. Biasing
        is always molecule specific i.e. no two different
        vermouth molecules can have the same bias.

        Parameters
        ----------
        molecule: :class:`vermouth.Molecule`
        """
        if self.idr_regions or self.auto:
            for molecule in system.molecules:
                if hasattr(molecule.res_graph):
                    res_graph = molecule.res_graph
                else:
                    make_residue_graph(molecule)

                bias_params = self.determine_residue_water_bias(self, res_graph)
                system.gmx_topology_params["nonbond_params"].update(bias_params)
        return system
