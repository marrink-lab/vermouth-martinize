# Copyright 2018 University of Groningen
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
Provides a processor that merges all the molecules from a system.
"""

from .processor import Processor


class MergeAllMolecules(Processor):
    """
    Merge all the molecules from a system.

    The molecules are merged into the first molecule of the system. Nothing is
    done if there are no molecules.
    """
    def run_system(self, system):
        if not system.molecules:
            return system

        molecule = system.molecules[0]
        for other in system.molecules[1:]:
            molecule.merge_molecule(other)

        system.molecules = [molecule]
        return system

    @staticmethod
    def run_molecule(molecule):
        raise NotImplementedError('MergeAllMolecules only works on systems.')
