# Copyright 2026 University of Groningen
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
Provides a processor that averages structure dependent dihedral angles over
copies of a molecule.
"""

import numpy as np
from .processor import Processor
from ..log_helpers import StyleAdapter, get_logger

LOGGER = StyleAdapter(get_logger(__name__))

# change in degrees of the reference angle of a copy above which we warn
MAX_SHIFT = 30


def find_copies(molecules, mask):
    """
    Group the molecules that share a moltype once `mask` is applied to them,
    using the same criterion as
    :class:`~vermouth.processors.name_moltype.NameMolType`.

    Parameters
    ----------
    molecules: collections.abc.Iterable[vermouth.molecule.Molecule]
    mask: collections.abc.Callable
        Returns a copy of a molecule without the properties that may differ
        between copies.

    Returns
    -------
    list[list[vermouth.molecule.Molecule]]
        The groups of copies, in order of first occurrence.
    """
    groups = []
    for molecule in molecules:
        masked = mask(molecule)
        for representative, copies in groups:
            if masked.share_moltype_with(representative):
                copies.append(molecule)
                break
        else:
            groups.append((masked, [molecule]))
    return [copies for _, copies in groups]


class AverageCopyDihedrals(Processor):
    """
    Average the reference angle of dihedrals that are computed from the
    structure over all copies of a molecule, such that the copies can share
    a single topology.

    Molecules are copies if they share a moltype once these reference angles
    are set aside, i.e. if they get one moltype from
    :class:`~vermouth.processors.name_moltype.NameMolType` after the
    averaging. The reference angle (second parameter) of every such dihedral
    is replaced by its circular mean over the copies, written with the same
    number of decimals. A warning is given if this changes the angle of a
    copy by more than `MAX_SHIFT` degrees.

    Parameters
    ----------
    group: str
        Meta group of the dihedrals to average.
    """
    def __init__(self, group="SC-BB-BB-SC scFix"):
        self.group = group

    def run_system(self, system):
        for molecules in find_copies(system.molecules, self._without_angles):
            if len(molecules) > 1:
                self.average(molecules)
        return system

    def _without_angles(self, molecule):
        """
        Copy of `molecule` without the reference angles of the dihedrals of
        `group`.
        """
        masked = molecule.copy()
        dihedrals = masked.interactions.get('dihedrals', [])
        for idx, dihedral in enumerate(dihedrals):
            if dihedral.meta.get('group') == self.group:
                parameters = dihedral.parameters[:1] + dihedral.parameters[2:]
                dihedrals[idx] = dihedral._replace(parameters=parameters)
        return masked

    def average(self, molecules):
        """
        Average the dihedrals of `group` over `molecules` in-place.

        Parameters
        ----------
        molecules: list[vermouth.molecule.Molecule]
            Copies of the same molecule, which have the same dihedrals in the
            same order.
        """
        dihedral_lists = [molecule.interactions.get('dihedrals', []) for molecule in molecules]
        for idx, dihedral in enumerate(dihedral_lists[0]):
            if dihedral.meta.get('group') != self.group:
                continue
            values = [dihedrals[idx].parameters[1] for dihedrals in dihedral_lists]
            angles = np.radians([float(value) for value in values])
            mean = np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
            # change of the angle of each copy, between 0 and 180 degrees
            shifts = np.abs((np.degrees(angles) - mean + 180) % 360 - 180)
            if shifts.max() > MAX_SHIFT:
                self._warn(molecules, dihedral.atoms, shifts)
            decimals = len(str(values[0]).partition('.')[2])
            for dihedrals in dihedral_lists:
                parameters = list(dihedrals[idx].parameters)
                parameters[1] = f"{mean:.{decimals}f}"
                dihedrals[idx] = dihedrals[idx]._replace(parameters=parameters)

    def _warn(self, molecules, atoms, shifts):
        """
        Warn that the averaging changes the angle of the dihedral between
        `atoms` by `shifts` degrees in the respective copies.
        """
        nodes = [molecules[0].nodes[atom] for atom in atoms]
        residues = dict.fromkeys(f"{node['resname']}{node.get('stash', {}).get('resid', node['resid'])}"
                                 for node in nodes)
        chains = [str(molecule.nodes[atoms[0]].get('chain')) for molecule in molecules]
        LOGGER.warning("Averaging over the copies (chains {}) changes the reference angle of the {} "
                       "dihedral {} of {} by {:.0f} degrees in chain {}.", ', '.join(chains), self.group,
                       '-'.join(node['atomname'] for node in nodes), '-'.join(residues), shifts.max(),
                       chains[np.argmax(shifts)], type='scfix-spread')
