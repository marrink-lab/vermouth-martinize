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
====

Our knowledge of proteins and Martini tells us that we need to add some more
information to the topology to account for secondary structure.

Let martinize2 deal with secondary structure for you
----

Martinize2 can deal with secondary structure intelligently using dssp in one of two ways:

1) Use a dssp executable installed on your machine
2) Use the dssp implementation in `mdtraj <https://mdtraj.org/1.9.4/api/generated/mdtraj.compute_dssp.html>`_

To explain these more:

1) If you have dssp installed locally, note that martinize2 is only validated for particular versions of dssp.
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


User knows best
-----------

If you already know the secondary structure of your protein and don't want to worry about
dssp calculating it correctly, the ``-ss`` flag can be used.

The ``-ss`` flag must be one of either:

1)   The same length as the number of residues in your protein, with a dssp code for each residue
2)   A single letter (eg. '``H``'), which will apply the same secondary structure parameters to your entire protein.

For example:

``martinize2 -f 181L_clean.pdb -o t4l_only.top -x t4l_cg.pdb -ss etcetc``

will use the ``etcetc`` dssp formatted string that the user provides to specify how the secondary structure is
treated, which must contain the same number of characters as the residues you have in the protein.











