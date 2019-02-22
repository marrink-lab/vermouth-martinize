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
Provides a processor to assign molecule type names to molecules.

A molecule type (moltype) is Gromacs's concept of a molecule. Providing a name
for a molecule type is required to write an ITP file for that molecule. We also
use the molecule type name to group molecules sharing the same molecule type.
Molecule type identity is tested based on
:meth:`vermouth.molecule.Molecule.share_moltype_with`.
"""

from .processor import Processor


class NameMolType(Processor):
    """
    Assigns molecule type (moltype) names to molecules.

    Moltype names are the names given to molecules in an ITP file. This
    processor assign consecutive names to the molecule. If the `deduplicate`
    argument is set to `True`, then the processor assigns the same name to all
    molecules with the same topology.

    By default, the moltype name is written under the "moltype" key of the
    molecule meta attributes. This key can be changed with the `meta_key`
    argument.

    Parameters
    ----------
    deduplicate: bool
        If `True`, the same name is given to all the molecules that share the
        same topology. Else, each molecule is given a different name.
    meta_key: str
        The name of the key in the molecule `meta` dictionary under which the
        moltype must be stored.

    See Also
    --------
    vermouth.processors.set_molecule_meta.SetMoleculeMeta
        This processor can set key/value pairs in the meta attributes of one
        molecule, or all molecules in a system. It can be used to set the
        moltype manually.
    vermouth.gmx.itp.write_molecule_itp
        Writes the ITP file for a molecule, and use the 'moltype' meta to name
        the molecule.
    """
    # TODO: See issue #35
    def __init__(self, deduplicate=True, meta_key='moltype'):
        self.deduplicate = deduplicate
        self.meta_key = meta_key
        super().__init__()

    def run_system(self, system):
        if not system.molecules:
            return system

        if self.deduplicate:
            self._name_with_deduplication(system)
        else:
            self._name_without_deduplication(system)

        return system

    def _name_with_deduplication(self, system):
        group_id = 0
        representatives = [(group_id, system.molecules[0])]
        for molecule in system.molecules:
            for match_id, template in representatives:
                if molecule.share_moltype_with(template):
                    break
            else:  # no break
                group_id += 1
                representatives.append((group_id, molecule))
                match_id = group_id
            molecule.meta[self.meta_key] = 'molecule_{}'.format(match_id)

    def _name_without_deduplication(self, system):
        for molecule_id, molecule in enumerate(system.molecules):
            molecule.meta[self.meta_key] = 'molecule_{}'.format(molecule_id)
