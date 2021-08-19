General Overview
================
VerMoUTH and martinize2 are tools for setting up starting structures for
molecular dynamics (MD) simulations starting from atomistic coordinates, with a
special focus on polymeric systems (including proteins and DNA). Existing tools
that do this are generally limited to strictly linear polymers, while VerMoUTH
and martinize2 make *no* assumptions regarding polymer structure. VerMoUTH is a
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
Vermouth and martinize2 are distributed through pypi and can be installed using
pip.

.. code-block:: bash

    pip install vermouth

The behavior of the ``pip`` command can vary depending on the specificity of your
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
will use the program [DSSP]_ to determine the proteins secondary structure (which
influences the topology), and produce an elastic network. See ``martinize2 -h``
for more options! Note that if ``martinize2`` runs into problems where the
produced topology might be invalid it will issue warnings. If this is the case
it won't write any output files, but also see the ``-maxwarn`` flag.

General layout
--------------
In VerMoUTH a :ref:`force field <data:Force Field>` is defined as a collection
of :ref:`Blocks <data:Block>`, :ref:`Links <data:Link>` and
:ref:`Modifications <data:Modification>`. Each of these is a graph, where nodes
describe atoms (or coarse-grained beads) and edges describe bonds between these.
:ref:`Blocks <data:Block>` describe idealized residues/monomeric repeat units
and their MD parameters and interactions. :ref:`Links <data:Link>` are molecular
fragments that describe MD parameters and interactions *between*
residues/monomeric repeat units. :ref:`Modifications <data:Modification>` are
molecular fragments that describe *deviations* from :ref:`Blocks <data:Block>`,
such as post-translational modifications and protonation states.
:ref:`Mappings <data:Mapping>` describe how molecular fragments can be converted
between force fields.

Finally, martinize2 is a pipeline that is built up from
:ref:`Processors <processors:Processor>`, which are defined by VerMoUTH.
Processors are isolated steps which function on either the complete system, or
single molecules.

Martinize2 identifies atoms mostly based on their *connectivity*. We read the
bonds present in the input file (as ``CONECT`` records), and besides that we
:ref:`guess bonds <martinize2_workflow:Make bonds>` based on atom names (within
residues) and on distances (between residues, using the same criteria as
[VMD]_). This means that your input structure must be reasonable.

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
.. [DSSP] - W.G. Touw, C. Baakman, J. Black, T.A.H. te Beek, E. Krieger, R.P. Joosten, et al., A series of PDB-related databanks for everyday needs, Nucleic Acids Res. 43 (2015) D364–D368. doi:10.1093/nar/gku1028.
   - W. Kabsch, C. Sander, Dictionary of protein secondary structure: pattern recognition of hydrogen-bonded and geometrical features., Biopolymers. 22 (1983) 2577–637. doi:10.1002/bip.360221211.