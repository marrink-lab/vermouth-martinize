Gromacs variables
#################

Some parameters in force fields included in the Vermouth library make use of Gromacs
`preprocessing <https://manual.gromacs.org/documentation/current/user-guide/mdp-options.html#preprocessing>`_.

Preprocessing statements in gromacs mdp files may be used to customise topologies at the preprocessing stage
of simulation setup. The statements in the ``define`` statement of the input molecular dynamics parameters
(.mdp) file are passed to the topology file, where ``#ifdef`` statements are used to modify the exact topology
of the molecule. For more examples of ``#ifdef`` statements, see the
`gromacs manual <https://manual.gromacs.org/2024.2/reference-manual/topologies/topology-file-formats.html#ifdef-statements>`_.


FLEXIBLE
========

The FLEXIBLE defines is used to switch some bonds between harmonic bonds and constraints. This is important for
minimization purposes.


POSRES
======

The POSRES defines is used to activate any position restraints defined in system topologies. Position restraints
are applied to beads by Martinize2 via the ``-p`` flag. Position restraints may be useful for minimization and
equilibration purposes.


POSRES_FC
=========

For minmization and equilibration purposes, it may be useful to reduce the position restraint on a molecule across
several successive simulations. In Martinize2, the strength of the position restraint is controlled by the
``-pf`` flag, and is 1000 kJ/mol/nm^2 by default. However, as position restraints are written as a ``ifndef``
statement in topology files, its exact value can be passed from the mdp file. For example::

 define = -DPOSRES_FC=500

will reduce the position restraint to 500 kJ/mol/nm^2 directly.
