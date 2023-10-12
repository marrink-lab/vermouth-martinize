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
from vermouth.processors import Processor
from .go_vs_includes import GoVirtIncludes
from .go_structure_bias import ComputeStructuralGoBias
from .go_water_bias import ComputeWaterGoBias
from ..processors import SetMoleculeMeta

class GoProcessorPipline(Processor):
    """
    Wrapping all processors for the go model.
    """
    def __init__(self, processor_list, **kwargs):
        self.processor_list = processor_list

    def prepare_run(self, system):
        """
        Things to do before running the pipeline.
        """
        # merge all molecules in the system
        # this will eventually become deprecated
        # with the proper Go-model for multimers
        vermouth.MergeAllMolecules().run_system(system)
        for molecule in system.molecules:
            res_graph = vermouth.graph_utils.make_residue_graph(system.molecules[0])
            molecule.residue_graph = res_graph

    def postprocess_run(self, system):
        """
        Do required post-processing.
        """
        pass

    def run_system(self, system):
        self.prepare_run(self, system)
        for processor in processor_list:
            process_args = inspect.getfullargspec(processor).args
            process_args_values = {self.kwargs[arg] for arg in process_args}
            processor(**process_args_values).run_system(system)
    return system

GoPipeline = ProcessorPipline([SetMoleculeMeta, GoVirtIncludes, ComputeStructuralGoBias, ComputeWaterGoBias])
