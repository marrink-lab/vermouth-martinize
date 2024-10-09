===========================
Mutations and modifications
===========================

Martinize2 facilitates a powerful syntax for defining mutations and modifications to your input protein structure.
Here we'll look at some of the options for how your protein can be changed during the coarse graining process. While
chemically and structurally the two processes are different, the command line options share the same syntax.

General syntax
====

For both modifications and mutations, the syntax is specified as, e.g. A-PHE45:ALA, parts of which can be eliminated
depending on what is needed. As per the documentation, this syntax can be thought of as
<chain>-<resname><resid>:<new resname>.

Mutate a single residue
=====

If you have a single specific residue that you want to mutate, the you can use the mutate flag. For example, if you know
your protein has a phenylalanine on chain A at resid 45 that you want to mutate to an alanine, then you can use the
command above:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -mutate A-PHE45:ALA``

If you have additional mutations to make, then you can give the ``-mutate`` flag as many times as you want. For example,
in addition to the F45A mutation above, we have an arginine on chain B at resid 50 that should be mutated to lysine:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -mutate A-PHE45:ALA -mutate B-ARG50:LYS``



Mutate all residues
======

If you need to simulate a protein where every instance of a particular residue has been mutated to another, you can
leave out aspects of the syntax described above. To build on the previous command, if we no longer needed just the
arginine at resid 50 on chain B to be mutated to lysine, but instead *all* instances, then we can leave out the chain
and resid of the command:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -mutate A-PHE45:ALA -mutate ARG:LYS``

This can be specified by chain, so that only arginines on chain B are mutated:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -mutate A-PHE45:ALA -mutate B-ARG:LYS``





