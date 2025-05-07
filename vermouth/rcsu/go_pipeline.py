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

    def prepare_run_method_0(self, system, moltype):
        """
        Things to do before running the pipeline.
        """
        # merge all molecules in the system
        # this will eventually become deprecated
        # with the proper Go-model for multimers
        vermouth.MergeAllMolecules().run_system(system)
        molecule = system.molecules[0]
        molecule.meta['moltype'] = moltype

    def prepare_run_method_1(self, system, moltype):
        structure_map = defaultdict(list)
        equivalence = {}
        chain_to_mol_id = {}
        reference_chain_map = {}

        for index, mol in enumerate(system.molecules):
            chain_id = next(iter(mol.nodes.values()))['chain']
            resseq = tuple((n['resname'], n['atomname']) for _, n in sorted(mol.nodes.items()))
            bonds = tuple(sorted((min(a, b), max(a, b)) for a, b in mol.edges))
            sig = (resseq, bonds)
            structure_map[sig].append((chain_id, mol))

            chain_to_mol_id[chain_id] = index

        for chain_mol_list in structure_map.values():
            reference_chain_id = chain_mol_list[0][0]  # chain of the first molecule in the group
            chains = [chain_id for chain_id, _ in chain_mol_list]
            for chain_id, mol in chain_mol_list:
                mol.meta['moltype'] = f"{moltype}_{reference_chain_id}"
                equivalence[chain_id] = [c for c in chains if c != chain_id]
                reference_chain_map[chain_id] = reference_chain_id

        system.go_params['reference_chain_map'] = reference_chain_map
        system.go_params['chain_to_mol_id'] = chain_to_mol_id
        system.go_params['unique_mols_map'] = equivalence

    def run_system(self, system, **kwargs):
        self.kwargs = kwargs
        if kwargs['go_method'] == 0:
            self.prepare_run_method_0(system, moltype=kwargs['moltype'])
        if kwargs['go_method'] == 1:
            self.prepare_run_method_1(system, moltype=kwargs['moltype'])
        for processor in self.processor_list:
            process_args = inspect.getfullargspec(processor).args
            process_args_values = {arg: self.kwargs[arg] for arg in kwargs.keys() if arg in process_args}
            processor(**process_args_values).run_system(system)

        return system


GoPipeline = GoProcessorPipeline([SetMoleculeMeta,
                                  VirtualSiteCreator,
                                  ComputeStructuralGoBias])
