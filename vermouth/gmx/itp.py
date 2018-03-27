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
Handle the ITP file format from Gromacs.
"""

import copy
import itertools

__all__ = ['write_molecule_itp', ]


def _attr_has_not_none_attr(obj, attr):
    """
    Raise a Value error is 'obj' does not have an attribute 'attr' or if its
    value is ``None``.
    """
    try:
        value = getattr(obj, attr)
    except AttributeError:
        value = None
    if value is None:
        raise ValueError('{} has no attribute "{}".'.format(obj, attr))


def _interaction_sorting_key(interaction):
    ifdef = interaction.meta.get('ifdef')
    ifndef = interaction.meta.get('ifndef')
    if ifdef is not None and ifndef is not None:
        raise ValueError('An interaction cannot have both an "ifdef" '
                         'and an "ifndef" meta attribute.')
    if ifdef is not None:
        conditional = (ifdef, True)
    elif ifndef is not None:
        conditional = (ifndef, False)
    else:
        conditional = ()

    group = interaction.meta.get('group')
    if group is None:
        group = ''

    return (conditional, group)


def write_molecule_itp(molecule, outfile):
    """
    Write a molecule in ITP format.

    The molecule must have a `moltype` and a `nrexcl` attribute. Each atom
    in the molecule must have at least the following keys: `atype`, `resid`,
    `resname`, `atomname`, and `charge_group`. Atoms can also have a
    `charge` and a `mass` key.

    Parameters
    ----------
    molecule: Molecule
        The molecule to write. See above for the minimal information the
        molecule must contain.
    outfile: file handler or file-like
        The file in which to write.

    Raises
    ------
    ValueError
        The molecule is missing required information.
    """
    # Make sure the molecule contains the information required to write the
    # header.
    _attr_has_not_none_attr(molecule, 'moltype')
    _attr_has_not_none_attr(molecule, 'nrexcl')

    # Make sure the molecule contains the information required to write the
    # [atoms] section. The charge and mass can be ommited, if so gromacs take
    # them from the [atomtypes] section of the ITP file.
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group'):
        if not all([attribute in atom for _, atom in molecule.atoms]):
            raise ValueError('Not all atom have a {}.'.format(attribute))

    # Get the maximum length of each atom field so we can align the fields.
    # Atom indexes are written as a consecutive series starting from 1.
    # The maximum index of a 0-based series is `len(x) - 1`; because the
    # series starts at 1, the maximum value is `len(x).
    max_length = {'idx': len(str(len(molecule)))}
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group', 'charge', 'mass'):
        max_length[attribute] = max(len(str(atom.get(attribute, '')))
                                    for _, atom in molecule.atoms)

    outfile.write('[ moleculetype ]\n')
    outfile.write('{} {}\n\n'.format(molecule.moltype, molecule.nrexcl))

    # The atoms in the [atoms] section must be consecutively numbered, yet
    # there is no guarantee that the molecule fulfill that constrain.
    # Therefore we renumber the atoms. The `correspondence` dict allows to
    # keep track of the correspondence between the original and the new
    # numbering so we can apply the renumbering to the interactions.
    # The resid and charge_group should also be consecutive, though this is
    # left as the user responsibility. Make sure residues and charge groups are
    # correctly numbered.
    correspondence = {}
    outfile.write('[ atoms ]\n')
    for idx, (original_idx, atom) in enumerate(molecule.atoms, start=1):
        correspondence[original_idx] = idx
        new_atom = copy.copy(atom)
        # The charge and the mass can be blank and read from the [atomtypes]
        # section of the ITP file.
        new_atom['charge'] = new_atom.get('charge', '')
        new_atom['mass'] = new_atom.get('mass', '')

        outfile.write('{idx:>{max_length[idx]}} '
                      '{atype:<{max_length[atype]}} '
                      '{resid:>{max_length[resid]}} '
                      '{resname:<{max_length[resname]}} '
                      '{atomname:<{max_length[atomname]}} '
                      '{charge_group:>{max_length[charge_group]}} '
                      '{charge:>{max_length[charge]}} '
                      '{mass:>{max_length[mass]}}\n'
                      .format(idx=idx, max_length=max_length, **new_atom))
    outfile.write('\n')

    # Write the interactions
    conditional_keys = {True: '#ifdef', False: '#ifndef'}
    for name, interactions in molecule.interactions.items():
        # Do not write an empty section.
        if not interactions:
            continue
        # Improper dihedral angles have their own section in the Molecule
        # object to distinguish them from the proper dihedrals. Yet, they
        # should be written under the [ dihedrals ] section of the ITP file.
        if name == 'impropers':
            name = 'dihedrals'
        outfile.write('[ {} ]\n'.format(name))
        interactions_group_sorted = sorted(
            interactions,
            key=_interaction_sorting_key
        )
        interaction_grouped = itertools.groupby(
            interactions_group_sorted,
            key=_interaction_sorting_key
        )
        for (conditional, group), interactions_in_group in interaction_grouped:
            if conditional:
                conditional_key = conditional_keys[conditional[1]]
                outfile.write('{} {}\n'.format(conditional_key, conditional[0]))
            if group:
                outfile.write('; {}\n'.format(group))
            for interaction in interactions_in_group:
                atoms = ' '.join('{atom_idx:>{max_length[idx]}}'
                                 .format(atom_idx=correspondence[x],
                                         max_length=max_length)
                                 for x in interaction.atoms)
                parameters = ' '.join(str(x) for x in interaction.parameters)
                comment = ''
                if 'comment' in interaction.meta:
                    comment = '; ' + interaction.meta['comment']
                outfile.write(' '.join((atoms, parameters, comment)) + '\n')
            if conditional:
                outfile.write('#endif\n')
            outfile.write('\n')
