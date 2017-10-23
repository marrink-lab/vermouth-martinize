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

    if not all(['mass' in atom for atom in molecule.atoms]):
        raise ValueError('Not all atom have a mass.')

    print('[ moleculetype ]', file=outfile)
    print(moltype, nrexcl, file=outfile)
    print('', file=outfile)

    print('[ atoms ]', file=outfile)
    for idx in molecule.nodes():
        atom = molecule.node[idx]
        print('{idx} {atype} {resid} {resname} {atomname} {charge_group} {charge} {mass}'
              .format(idx=idx, **atom), file=outfile)
    print('', file=outfile)

    for name, interactions in molecule.interactions.items():
        print('[ {} ]'.format(name), file=outfile)
        for interaction in interactions:
            print(' '.join(str(x) for x in interaction[0]),
                  ' '.join(str(x) for x in interaction[1]),
                  file=outfile)
        print('', file=outfile)
