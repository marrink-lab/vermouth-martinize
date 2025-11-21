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
Wrapper of Processors defining the GoPipline.
"""
import networkx as nx
import inspect
import vermouth
from ..processors.processor import Processor
from .go_vs_includes import VirtualSiteCreator
from .go_structure_bias import ComputeStructuralGoBias
from ..processors import SetMoleculeMeta
from collections import defaultdict

class GoProcessorPipeline(Processor):
    """
    Wrapping all processors for the go model.
    """
    def __init__(self, processor_list):
        self.processor_list = processor_list
        self.kwargs = {}

    def prepare_run(self, system, moltype):
        structure_map = defaultdict(list)
        unique_mols_map = {}
        reference_mol_map = {}

        for mol in system.molecules:
            mol_id = next(iter(mol.nodes.values()))['mol_idx']
            resseq = tuple((n['resname'], n['atomname']) for _, n in sorted(mol.nodes.items()))
            bonds = tuple(sorted((min(a, b), max(a, b)) for a, b in mol.edges))
            sig = (resseq, bonds)
            structure_map[sig].append((mol_id, mol))

        for mol_list in structure_map.values():
            reference_mol_id = mol_list[0][0]  # mol id of the first molecule in the group
            mol_ids = [mol_id for mol_id, _ in mol_list]
            for mol_id, mol in mol_list:
                mol.meta['moltype'] = f"{moltype}_{reference_mol_id}"
                unique_mols_map[mol_id] = [m_id for m_id in mol_ids if m_id != mol_id]
                reference_mol_map[mol_id] = reference_mol_id

        system.go_params['reference_mol_map'] = reference_mol_map
        system.go_params['unique_mols_map'] = unique_mols_map

    def run_system(self, system, **kwargs):
        self.kwargs = kwargs
        self.prepare_run(system, moltype=kwargs['moltype'])

        for processor in self.processor_list:
            process_args = inspect.getfullargspec(processor).args
            process_args_values = {arg: self.kwargs[arg] for arg in kwargs.keys() if arg in process_args}
            processor(**process_args_values).run_system(system)
        return system


GoPipeline = GoProcessorPipeline([SetMoleculeMeta,
                                  VirtualSiteCreator,
                                  ComputeStructuralGoBias])
