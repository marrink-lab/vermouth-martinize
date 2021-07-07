General Overview
================
VerMoUTH and martinize2 are tools for setting up molecular dynamics (MD)
simulations starting from atomistic coordinates, with a special focus on
polymeric systems (including proteins and DNA). Existing tools that do this
are generally limited to strictly linear polymers, while VerMoUTH and
martinize2 make *no* assumptions regarding polymer structure. VerMoUTH is a
python library that can be used programmatically. Martinize2 is a command line
tool build on top of that.

VerMoUTH and martinize2 are also capable of dealing with structures where atom
names are not provided, and to some extent with incomplete structures where
atoms are missing from the input structure due to e.g. experimental limitations.
There is also support for post-translational modifications.

VerMoUTH and martinize2 can be used to generate both atomistic and
coarse-grained topologies and are the preferred method of generating topologies
for the [Martini3]_ force field.

Installation instructions
-------------------------
Vermouth and martinize2 are distribute through pypi and can be installed using
pip.

.. code-block:: bash

    pip install vermouth

The behavior of the ``pip`` command can vary depending of the specificity of your
python installation. See the `documentation on installing a python package
<https://packaging.python.org/tutorials/installing-packages/#installing-packages>`_
to learn more.

Vermouth has `SciPy <https://scipy.org>`_ as *optional* dependency. If available
it will be used to accelerate the distance calculations when `making bonds
<martinize2_workflow:Make bonds>`_

Quickstart
----------
The CLI of martinize2 is very similar to that of martinize1, and can often be
used as a drop-in replacement. For example:

.. code-block:: bash

    martinize2 -f lysozyme.pdb -x cg_protein.pdb -o topol.top
        -ff martini3001 -dssp -elastic

This will read an atomistic ``lysozyme.pdb`` and produce a Martini3_ compatible
structure and topology at ``cg_protein.pdb`` and ``topol.top`` respectively. It
will use the program DSSP to determine the proteins secondary structure (which
influences the topology), and produce an elastic network. See ``martinize2 -h``
for more options! Note that if ``martinize2`` runs into problems where the
produced topology might be invalid it will issue warnings. If this is the case
it won't write any output files, but also see the ``-maxwarn`` flag.

General layout
--------------
In VerMoUTH a force field is defined as a collection of Blocks, Links and
Modifications. Each of these is a graph, where nodes describe atoms (or
coarse-grained beads) and edges describe bonds between these. Blocks describe
idealized residues/monomeric repeat units and their MD parameters and
interactions. Links are molecular fragments that describe MD parameters and
interactions *between* residues/monomeric repeat units. Modifications are
molecular fragments that describe *deviations* from Blocks, such as
post-translational modifications and protonation states. Mappings describe how
molecular fragments can be converted between force fields.

Finally, martinize2 is a pipeline that is built up from Processors, which are
defined by VerMoUTH. Processors are isolated steps which function on either the
complete system, or single molecules.

Martinize2 identifies atoms mostly based on their *connectivity*. We read the
bonds present in the input file (as ``CONECT`` records), and besides that we
guess bonds based on atom names (within residues) and on distances (between
residues, using the same criteria as [VMD]_). This means that your input structure
must be reasonable.

Citing
------
A publication for vermouth and martinize 2 is currently being written.
For now, please cite the relevant chapter from the thesis of Peter C Kroon:

Kroon, P.C. (2020). Martinize 2 -- VerMoUTH. *Aggregate, automate, assemble* (pp. 16-53). ISBN:
978-94-034-2581-8.

References
----------
.. [Martini3] P.C.T. Souza, R. Alessandri, J. Barnoud, S. Thallmair, I. Faustino, F. Grünewald, et al., Martini 3: a general purpose force field for coarse-grained molecular dynamics, Nat. Methods. 18 (2021) 382–388. doi:10.1038/s41592-021-01098-3.
.. [VMD] W. Humphrey, A. Dalke and K. Schulten, "VMD - Visual Molecular Dynamics", J. Molec. Graphics, 1996, vol. 14, pp. 33-38. http://www.ks.uiuc.edu/Research/vmd/.
