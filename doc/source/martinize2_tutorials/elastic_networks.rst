# Elastic Networks

The first method applied to maintain the secondary and tertiary structure
of Martini proteins was `Elastic Networks <https://doi.org/10.1021/ct9002114>`_.

Elastic networks are formed by finding contacts between protein backbone
beads within a particular cutoff distance, and applying harmonic bonds between them,
restraining them at the initial distance throughout your simulation.

Elastic networks are applied in Martinize2 as per the section of the help::


  -elastic              Write elastic bonds (default: False)
  -ef RB_FORCE_CONSTANT
                        Elastic bond force constant Fc in kJ/mol/nm^2 (default: 500)
  -el RB_LOWER_BOUND    Elastic bond lower cutoff: F = Fc if rij < lo (default: 0)
  -eu RB_UPPER_BOUND    Elastic bond upper cutoff: F = 0 if rij > up (default: 0.9)
  -ermd RES_MIN_DIST    The minimum separation between two residues to have an RB the default value is set by the force-field. (default: None)
  -ea RB_DECAY_FACTOR   Elastic bond decay factor a (default: 0)
  -ep RB_DECAY_POWER    Elastic bond decay power p (default: 1)
  -em RB_MINIMUM_FORCE  Remove elastic bonds with force constant lower than this (default: 0)
  -eb RB_SELECTION      Comma separated list of bead names for elastic bonds (default: None)
  -eunit RB_UNIT        Establish what is the structural unit for the elastic network. Bonds are only created within a unit. Options are molecule, chain,
                        all, or aspecified region defined by resids, with followingformat: <start_resid_1>:<end_resid_1>, <start_resid_2>:<end_resid_2>...
                        (default: molecule)

Let's have a look at how to combine these in more detail.


Basic usage
------
Without any further consideration, an elastic network can be added to your martinize2 command easily:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -elastic``

which will apply the default harmonic bond constant (500 kJ/mol/nm^2) between non-successive BB beads
which are < 0.9 nm apart in space.

NOTE! For proteins in martini 3, the default constant should be 700 kJ/mol/nm^2. Changing the default
elastic constant per force field will be fixed in future versions of martinize2.


Customising cutoffs
-------

If your system requires more of a custom cutoff to better reproduce the dynamics of your protein,
the region for the force to be applied in can be customised using the ``-el`` and ``-eu`` flags:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -elastic -el 0.1 -eu 0.5``

In this example, the elastic network will only be applied between backbone beads which are between 0.1 and 0.5 nm
apart.

Using decays
-----

The strength of the elastic bond can be tuned with distance using an exponential decay function,
which uses the ``-ea`` and ``-ep`` flags as input parameters:

.. math::
  decay = e^{(- f * ((x - l) ^p))

where:

- ``l`` = lower bound  (-el)
- ``f`` = decay factor (-ea)
- ``p`` = decay power  (-ep)

Combining parameters
------


.. image:: elastic_examples.png

Here we see several examples of how the strength of elastic network harmonic bonds can be tuned.

The first example uses default parameters (albeit a shorter cutoff), such that a constant force is
applied between all backbone beads within the cutoff.

The second and third examples use a slightly longer cutoff, and apply a gentle decay function
to the strength of the network. In the first case, the decay is applied naively, and as such its
strength decays from 0 distance. In the second case, combining the decay with a lower cutoff means that
for backbone beads that are close the elastic network strength is constand, but is lower between pairs slightly
further away.

The fourth example shows a similar function to the second example, but with a longer cutoff and a stronger decay.
(note the form of the exponential decay above)

The fifth example adds an additional parameter ``-em`` into the function. As described in the help, if forces are
calculated to be lower than this force, they are removed and set to zero. Note how the input values are almost identical
to the fourth example, which would otherwise get cutoff at 0.9 nm. Because the decay function reduces the force below
the minimum before the cutoff, it overrides it and the force is zeroed before the upper cutoff anyway.
