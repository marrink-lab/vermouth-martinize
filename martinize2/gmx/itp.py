import copy

def write_itp(molecule, outfile):
    try:
        moltype = molecule.moltype
    except AttributeError:
        moltype = None
    if moltype is None:
        raise ValueError('Molecule has no moltype.')

    try:
        nrexcl = molecule.nrexcl
    except AttributeError:
        nrexcl = None
    if nrexcl is None:
        raise ValueError('Molecule has no nrexcl.')

    max_length = {}
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group'):
        if not all([attribute in atom for _, atom in molecule.atoms]):
            raise ValueError('Not all atom have a {}.'.format(attribute))
    for attribute in ('atype', 'resid', 'resname', 'atomname',
                      'charge_group', 'charge', 'mass'):
        max_length[attribute] = max(len(str(atom.get(attribute, '')))
                                    for _, atom in molecule.atoms)
    # Atom indexes are written as a continuous series starting from 1.
    # The maximum index of a 0-based series is `len(x) - 1`, because the
    # series starts at 1, the maximum value is `len(x).
    max_length['idx'] = len(str(len(molecule)))



    print('[ moleculetype ]', file=outfile)
    print(moltype, nrexcl, file=outfile)
    print('', file=outfile)

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
