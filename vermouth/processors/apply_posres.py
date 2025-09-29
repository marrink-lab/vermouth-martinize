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


def apply_posres(molecule, selector, atomnames, force_constant, functype=1, ifdef='POSRES'):
    """
    Apply position restraint interactions to a molecule.
    Position restraints are written as a constant to be processed by gmx grompp
    from a default force constant.

    molecule: vermouth.molecule.Molecule
        molecule to which to apply position restraints
    selector: vermouth.selector
        selector to use for node selection for interaction application
    atomnames: str
        name of target atom to use in selector
    force_constant: int
        default force constant to be used
    functype: int
        gromacs function type for position restraint
    ifdef: str
        ifdef statement for interaction meta
    """
    for key, node in molecule.nodes.items():
        if selector(node, atomnames):
            parameters = [functype, ] + ["POSRES_FC", ] * 3
            if ifdef is not None:
                meta = {'ifdef': ifdef}
            else:
                meta = {}
            molecule.add_interaction('position_restraints',
                                     (key, ), parameters, meta)
    molecule.meta['define'] = {'POSRES_FC': force_constant}
    return molecule


class ApplyPosres(Processor):
    def __init__(self, selector, force_constant, functype=1, ifdef='POSRES'):
        super().__init__()
        self.selector, self.atomnames = selector
        self.force_constant = force_constant
        self.functype = functype
        self.ifdef = ifdef

    def run_molecule(self, molecule):
        return apply_posres(
            molecule,
            self.selector,
            self.atomnames,
            self.force_constant,
            functype=self.functype,
            ifdef=self.ifdef,
        )
