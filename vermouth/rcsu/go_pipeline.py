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
import inspect
import vermouth
from ..processors.processor import Processor
from .go_vs_includes import VirtualSiteCreator
from .go_structure_bias import ComputeStructuralGoBias
from ..processors import SetMoleculeMeta

class GoProcessorPipline(Processor):
    """
    Wrapping all processors for the go model.
    """
    def __init__(self, processor_list):
        self.processor_list = processor_list
        self.kwargs = {}

    def prepare_run(self, system, moltype):
        """
        Things to do before running the pipeline.
        """
        # merge all molecules in the system
        # this will eventually become deprecated
        # with the proper Go-model for multimers
        vermouth.MergeAllMolecules().run_system(system)
        molecule = system.molecules[0]
#        res_graph = vermouth.graph_utils.make_residue_graph(molecule)
#        molecule.res_graph = res_graph
        molecule.meta['moltype'] = moltype

    def run_system(self, system, **kwargs):
        self.kwargs = kwargs
        self.prepare_run(system, moltype=kwargs['moltype'])
        for processor in self.processor_list:
            process_args = inspect.getfullargspec(processor).args
            process_args_values = {arg:self.kwargs[arg] for arg in kwargs.keys() if arg in process_args}
            processor(**process_args_values).run_system(system)
        return system

GoPipeline = GoProcessorPipline([SetMoleculeMeta,
                                 VirtualSiteCreator,
                                 ComputeStructuralGoBias])
