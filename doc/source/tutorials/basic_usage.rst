===========
Basic usage
===========

Overly basic usage
==================

Without any other additions, ``martinize2`` can take your protein, and make a ready coarse
grained model with some martini
forcefield:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb``

This command will (try) and convert your protein from the atomistic input to one
in the Martini3 force field.

Other force fields are available to convert your protein to. To view them you
can use ``martinize2 -list-ff``.

Minimal physical reality usage
==============================

Our knowledge of proteins and Martini tells us that we need to add some more
information to the topology to account for secondary structure.

Let martinize2 deal with secondary structure for you
----------------------------------------------------

Martinize2 can deal with secondary structure intelligently using dssp in one of two ways:

1) Use a dssp executable installed on your machine
2) Use the dssp implementation in `mdtraj <https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html>`_

To explain these more:

1) If you have dssp installed locally, note that martinize2 is only validated for particular versions of dssp.
   Currently the versions supported are 2.2.1 and 3.0.0.
   If a non-validated version is used, a warning will be raised and nothing is written.
   If you know what you're doing and are confident with what's been produced, you can override the warning
   with the ``-maxwarn`` flag. Otherwise, dssp can be used using the ``-dssp`` flag in martinize.

   Where you have a local installation, replace ``/path/to/dssp`` in the following command with the
   location of your installation:

   ``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp /path/to/dssp``

2) If you have `mdtraj <https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html>`_ installed in
   your python environment, the ``-dssp`` flag can be left blank, and martinize2 will attempt to use
   dssp from there:

   ``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp``

If you want to check how the secondary structure has been assigned, martinize2 will write the
secondary structure sequence into the header information of the output topology files, along
with the input command used.

User knows best
---------------

If you already know the secondary structure of your protein and don't want to worry about
dssp assigning it correctly, the ``-ss`` flag can be used instead.

The ``-ss`` flag must be one of either:

1)   The same length as the number of residues in your protein, with a dssp code for each residue
2)   A single letter (eg. '``H``'), which will apply the same secondary structure parameters to your entire protein.

For example:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ss HHHHHHHHHH``

will read the ``HHHHHHHHHH`` dssp formatted string, indicating that there are exactly 10 residues in the
input pdb which should all be treated as part of a helix. If the string provided does not contain the same
number of residues as the input file, an error will be raised.

Alternatively in this case, as we know everything should be a helix, the same result can be achieved through
using a single letter as described above:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ss H``

If instead we needed to assert that only the first five residues are in a helix, and the final five are coiled,
we must we the full length string again:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ss HHHHHCCCCC``

Other basic features of martinize2
==================================

Side chain fixing
-----------------

One important addition for the generation of correct Martini protein topologies is side chain fixing.
Side chain fixing is essential for ensuring correct sampling of side chain dynamics, and involves adding
additional bonded terms into the structure of the protein, relating to the angles and dihedrals around
side chain and backbone atoms. For further information on the background and motivation for these terms,
please read the paper by `Herzog et. al <https://pubs.acs.org/doi/full/10.1021/acs.jctc.6b00122>`_.

From martinize2 version ≥ 0.12.0, side chain fixing is done automatically. For martinize2 ≤ 0.11.0,
side chain fixing must be done for the martini 3 forcefield manually:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -scfix``

You can check the version of martinize2 that you have installed by running:

``martinize2 --version``

In martinize2 ≥ 0.12.0, side chain fixing is done by default. If you want to turn this behaviour off
for the forcefield that you're using, the `-noscfix` flag may be used instead.

Secondary and tertiary structure considerations
-----------------------------------------------

The examples given on this page show how to generate basic coarse grained topologies for the
martini force field using martinize2. Martinize2 has many further features to
transform your simulation from a physically naive coarse grained model to one that really
reproduces underlying atomistic behaviour. One of the most important considerations in this
is how to treat secondary structure in the absence of hydrogen bonding. Two such methods
are described in both the
`Martini protein tutorial <https://cgmartini.nl/docs/tutorials/Martini3/ProteinsI/>`_, which
should be an essential route in to conducting simulations with the martini force field.

We cover the documentation of these features in greater detail in the pages about
`Elastic Networks </tutorials/elastic_networks.html>`_ and `Gō models </tutorials/go_models.html>`_.

Cysteine bridges
----------------

If your protein contains cysteine bridges, martinize2 will attempt to identify linked residues
and add correct Martini parameters between them. When bridged residue pairs
are identified, a constraint of length 0.24 nm will be added between the side chains of the two
residues.

The `-cys` flag can read one of two types of argments. The default value `-cys auto` will look
for pairs of residues within a short cutoff. This is assumed by default, so if your protein
contains disulfide bridges at the correct distance, then they'll be found automatically just using:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp``

You can check if the correct bridges have been identified and added in the `[ constraints ]` directive
of the output itp file. Disulfide bonds are written at the top of the directive like so::

 [ constraints ]
  5 25 1 0.24 ; Disulfide bridge
 30 50 1 0.24 ; Disulfide bridge

Alternatively if you need to assert the identification of the bridges over a distance that isn't
automatically identified, a distance in nm can be supplied to `-cys`, e.g.:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -cys 5``

will look for cysteines within 5 nm of each other and apply the same disulfide bond as before.

Citations
---------

At the end of the execution of martinize2, the final output log writes general information with
requests to citate relevant papers. Martinize2 collects paper citation information dynamically
based on what features have been used, such as force fields, extra parameters,
how secondary structure has been determined, and so on. For posterity and to ensure ease of
reference, the same paper citations are also printed to the header information of the output
topology files.

As the correct references are collected dynamically, all the papers printed here by martinize2
should be cited, to ensure that relevant authors and features are credited. Please do so!
Martinize2 is both free and open source, and continued citations help us to keep it this way.


