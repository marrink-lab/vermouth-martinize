=============
Water biasing
=============

One feature associated with the latest version of the
`Go model <https://www.nature.com/articles/s41467-025-58719-0>`_ is the ability to
bias the non-bonded interactions with water, specified by secondary structure. As the reference
demonstrates, this may be important in fixing several problems with the current model of proteins,
including over-compactness of intrinsically disordered regions.

The documentation describes these features::

  Apply water bias.:
    -water-bias           Automatically apply water bias to different secondary structure elements. (default: False)
    -water-bias-eps WATER_BIAS_EPS [WATER_BIAS_EPS ...]
                          Define the strength of the water bias by secondary structure type. For example, use `H:3.6 C:2.1` to bias helixes and coils. Using
                          the idr option (e.g. idr:2.1) intrinsically disordered regions are biased seperately. (default: [])
    -id-regions WATER_IDRS [WATER_IDRS ...]
                          Intrinsically disordered regions specified by resid.These parts are biased differently when applying a water bias.format:
                          <chain>-<start_resid_1>:<end_resid_1> <chain>-<start_resid_2>:<end_resid_2>... (default: [])

These flags can be specified in conjunction with the Go model.


Water biasing for secondary structure
-------------------------------------

To apply a water bias to your protein dependent on the secondary structure, the first two flags
described above must be used.

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -dssp -water-bias -water-bias-eps H:1``

This will produce a coarse-grained model of your protein, with virtual sites along the backbone.
The virtual sites will be defined in an external file, which should be included in your topology
as per the Go model instructions.

There will also be a second file, defining the additional non-bonded interactions between
water and the secondary structure elements defined in the command. In this case, any residue
identified as ``H`` (*i.e.* helix) by dssp will have an additional Lennard-Jones interaction of
epsilon = 1 kJ/mol between its backbone virtual site and water.

To define more interactions based on secondary structure, add more letter codes to the
``-water-bias-eps``:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -dssp -water-bias -water-bias-eps H:1 C:0.5 E:2``


Water biasing for intrinsically disordered regions/proteins
-----------------------------------------------------------

If you have disordered regions in your protein, then they can have additional bonded and nonbonded
parameters added (described more in the `Go model paper <https://www.nature.com/articles/s41467-025-58719-0>`_).

These regions need to firstly be annotated by the user, using the ``-id-regions`` flag to indicate resid segments
known to be disordered:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -dssp -id-regions A-1:10 B-65:92``

Ideally, as the paper describes, these should have their water bias and bonded parameters fixed too.
This can be done by combining the above command with the ones previously described about water biasing:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -dssp -id-regions A-1:10 B-65:92 -water-bias -water-bias-eps idr:0.5``

Here, ``-idr-tune`` makes sure that the additional bonded parameters are applied to the region specified by ``-id-regions``,
while ``-water-bias`` and ``-water-bias-eps idr:0.5`` ensures that for the idr region defined, an additional nonbonded parameter
with water is written to the nonbond_params.itp file.

For a single chain, or a homomultimer containing identical disordered regions, the chain specifier on the ``-id-regions`` flag is
not necessary. The command:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -dssp -id-regions 50:75 -water-bias -water-bias-eps idr:0.5``

will apply disordered parameters and biasing to residues 50:75 of all chains in the system.

If you're working extensively with proteins which are fully disordered in Martini, it may be more convenient to
use `Polyply <https://github.com/marrink-lab/polyply_1.0>`_ to generate the input parameters for your system
than Martinize2, as Polyply does not require an atomistic input structure to generate these parameters. The
`tutorial <https://github.com/marrink-lab/polyply_1.0/wiki/Tutorial:-Martini-3-IDPs>`_ on the Polyply wiki
may be a useful starting point as an indication for Polyply can be used for this.

