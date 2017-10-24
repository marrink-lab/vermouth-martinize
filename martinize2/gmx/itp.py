"""
Handle the ITP file format from Gromacs.
"""

from __future__ import print_function
import copy

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

    print('[ moleculetype ]', file=outfile)
    print(molecule.moltype, molecule.nrexcl, file=outfile)
    print('', file=outfile)

    # The atoms in the [atoms] section must be consecutively numbered, yet
    # there is no guarantee that the molecule fulfill that constrain.
    # Therefore we renumber the atoms. The `correspondence` dict allows to
    # keep track of the correspondence between the original and the new
    # numbering so we can apply the renumbering to the interactions.
    # The resid and charge_group should also be consecutive, though this is
    # left as the user responsibility. Make sure residues and charge groups are
    # correctly numbered.
    correspondence = {}
    print('[ atoms ]', file=outfile)
    for idx, (original_idx, atom) in enumerate(molecule.atoms, start=1):
        correspondence[original_idx] = idx
        new_atom = copy.copy(atom)
        # The charge and the mass can be blank and read from the [atomtypes]
        # section of the ITP file.
        new_atom['charge'] = new_atom.get('charge', '')
        new_atom['mass'] = new_atom.get('mass', '')

        print('{idx:>{max_length[idx]}} '
              '{atype:<{max_length[atype]}} '
              '{resid:>{max_length[resid]}} '
              '{resname:<{max_length[resname]}} '
              '{atomname:<{max_length[atomname]}} '
              '{charge_group:>{max_length[charge_group]}} '
              '{charge:>{max_length[charge]}} '
              '{mass:>{max_length[mass]}}'
              .format(idx=idx, max_length=max_length, **new_atom),
              file=outfile)
    print('', file=outfile)

    for name, interactions in molecule.interactions.items():
        print('[ {} ]'.format(name), file=outfile)
        for interaction in interactions:
            atoms = ' '.join('{atom_idx:>{max_length[idx]}}'
                             .format(atom_idx=correspondence[x],
                                     max_length=max_length)
                             for x in interaction[0])
            parameters = ' '.join(str(x) for x in interaction[1])
            print(atoms, parameters, file=outfile)
        print('', file=outfile)
