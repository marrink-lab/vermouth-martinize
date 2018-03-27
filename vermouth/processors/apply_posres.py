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

from .processor import Processor


def apply_posres(molecule, selector, force_constant, functype=1, ifdef='POSRES'):
    for key, node in molecule.nodes.items():
        if selector(node):
            parameters = [functype, ] + [force_constant, ] * 3
            if ifdef is not None:
                meta = {'ifdef': ifdef}
            else:
                meta = {}
            molecule.add_interaction('position_restraints',
                                     (key, ), parameters, meta)
    return molecule


class ApplyPosres(Processor):
    def __init__(self, selector, force_constant, functype=1, ifdef='POSRES'):
        super().__init__()
        self.selector = selector
        self.force_constant = force_constant
        self.functype = functype
        self.ifdef = ifdef

    def run_molecule(self, molecule):
        return apply_posres(
            molecule,
            self.selector,
            self.force_constant,
            functype=self.functype,
            ifdef=self.ifdef,
        )
