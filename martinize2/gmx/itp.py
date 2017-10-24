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
                      'charge_group', 'charge', 'mass'):
        if not all([attribute in atom for atom in molecule.atoms]):
            raise ValueError('Not all atom have a {}.'.format(attribute))
        max_length[attribute] = max(len(str(atom[attribute]))
                                    for atom in molecule.atoms)
    max_length['idx'] = len(str(max(molecule.nodes())))



    print('[ moleculetype ]', file=outfile)
    print(moltype, nrexcl, file=outfile)
    print('', file=outfile)

    print(max_length)
    print('[ atoms ]', file=outfile)
    for idx in molecule.nodes():
        atom = molecule.node[idx]
        print('{idx:>{max_length[idx]}} '
              '{atype:<{max_length[atype]}} '
              '{resid:>{max_length[resid]}} '
              '{resname:<{max_length[resname]}} '
              '{atomname:<{max_length[atomname]}} '
              '{charge_group:>{max_length[charge_group]}} '
              '{charge:>{max_length[charge]}} '
              '{mass:>{max_length[mass]}}'
              .format(idx=idx, max_length=max_length, **atom), file=outfile)
    print('', file=outfile)

    for name, interactions in molecule.interactions.items():
        print('[ {} ]'.format(name), file=outfile)
        for interaction in interactions:
            print(' '.join('{x:>{max_length[idx]}}'
                           .format(x=x, max_length=max_length)
                           for x in interaction[0]),
                  ' '.join(str(x) for x in interaction[1]),
                  file=outfile)
        print('', file=outfile)
