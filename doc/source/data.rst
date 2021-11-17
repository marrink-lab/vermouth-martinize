Data
====
VerMoUTH knows several data structures, most of which describe atoms (or CG
beads) and connections between those. As such, these are modelled as
mathematical graphs, where the nodes describe the particles, and edges the bonds
between these. In addition, these data structures describe the MD parameters and
interactions, such as bonds, atom types, angles, etc.

Molecule
--------
A :class:`~vermouth.molecule.Molecule` is a :class:`~networkx.Graph` where nodes
are atoms/beads, and edges are the connections between theses (i.e. bonds [#]_)
Generally, molecules are a single connected components [#]_. Interactions are
accessible through the :attr:`~vermouth.molecule.Molecule.interactions`
attribute. Non-bonded parameters are not fully defined: nodes have an 'atype'
attribute describing the particle type to be used in an MD simulation, but we
don't store the associated e.g. Lennard-Jones parameters.

:class:`Molecules <vermouth.molecule.Molecule>` define a few notable convenience
methods:

 - :meth:`~vermouth.molecule.Molecule.merge_molecule`: Add all atoms and
   interactions from a molecule to this one. Note that this can also be used to
   add a :class:`vermouth.molecule.Block` to a molecule! This way you can
   incrementally build polymers from monomers. This method will always produce
   a disconnected graph, so be sure to add the appropriate edges afterwards.
 - :meth:`~vermouth.molecule.Molecule.make_edges_from_interactions`: To generate
   edges from bond, angle, dihedral, cmap and constraint interactions. This is
   the only way interactions and their parameters are interpreted in vermouth.

.. [#] But note that not every edge has to correspond to a bond and vice versa.
.. [#] I.e. there is a path from any node to any other node in the molecule.

Block
-----
A :class:`~vermouth.molecule.Block` can be seen as a canonical residue
containing all atoms and interactions, and where all atom names are correct.
A block should be a single connected component, and atom names within a block
are assumed to be unique.

Blocks can be defined through Gromacs' ``.itp`` and ``.rtp`` file formats.

:class:`Blocks <vermouth.molecule.Block>` define a few notable convenience
methods:

 - :meth:`~vermouth.molecule.Block.guess_angles`: Generate all possible angles
   based on the edges.
 - :meth:`~vermouth.molecule.Block.guess_dihedrals`: Generate all possible
   dihedral angles based on the edges.
 - :meth:`~vermouth.molecule.Block.to_molecule`: Create a new
   :class:`~vermouth.molecule.Molecule` based on this block.

Link
----
A :class:`~vermouth.molecule.Link` is used to describe interactions *between*
residues. As such, it consists of nodes and edges describing the molecular
fragment it should apply to, as well as the associated changes in MD parameters.
For example, a link can describe the addition, change or removal of specific
interactions or node attributes. They can also be used to remove nodes. Although
it is possible to generate *all* MD parameters and interactions using Links,
rather than taking them from constituent blocks, this is not the preferred
method. The approach where links only affect the parameters where they depend on
the local structure makes it easier to reason about how the final topology is
constructed, and the performance is better.

Besides nodes, edges and interactions links also describe non-edges, patterns
and removed interactions. Non-edges and patterns are used when matching the link
to a molecule. Where there is a non-edge in the link there cannot be an edge in
the molecule, and the atoms involved do not need to be present in the molecule.
Patterns provide a concise way where either one of multiple conditions must be
met. For example two neighbouring 'BB' beads, where one must have a helical
secondary structure, and the other should be a coil.

Links can be defined through :ref:`.ff files <file_formats:.ff file format>`.
See also: :ref:`Apply Links <martinize2_workflow:4) Apply Links>`.

Modification
------------
A :class:`~vermouth.molecule.Modification` describes how a residue deviates from
its associated :class:`~vermouth.molecule.Block`, such as non-standard
protonation states and termini. Modifications differentiate between
atoms/particles that should already be described by the block and atoms that are
only described by the modification.

A modification can add or remove nodes, change node attributes, and add, change,
or remove interactions; much like a `Link`_.

Modifications can be defined through :ref:`.ff files <file_formats:.ff file format>`.
See also: :ref:`Identify modifications <martinize2_workflow:Identify modifications>`.

Force Field
-----------
A :class:`force field <vermouth.forcefield.ForceField>` is a collection of
:ref:`Blocks <data:Block>`, :ref:`Links <data:Link>` and
:ref:`Modifications <data:Modification>`. Force fields are identified by their
:attr:`~vermouth.forcefield.ForceField.name`, which should be unique. Within a
force field blocks and modifications should also have unique names.

Note that this is only a subset of a force field in the MD sense: a VerMoUTH
:class:`force field <vermouth.forcefield.ForceField>` does not include e.g.
non-bonded parameters (only the particle types are included), or functional
forms.

The ``universal`` force field deserves special mention. If not overridden with
the ``-from`` flag this force field is used. This force field does not define
any MD parameters, but this is fine. Instead, this force field defines only atom
names and the associated connections.

Mapping
-------
A :class:`~vermouth.map_parser.Mapping` describes how molecular fragments can
be transformed from one force field to another.

Mappings can be provided through [backward]_ style ``.map`` files, or the more
powerful (but verbose) :ref:`.mapping <file_formats:.mapping file format>` format.
See also: :ref:`Resolution transformation <martinize2_workflow:3) Resolution transformation>`.

.. [backward] T.A. Wassenaar, K. Pluhackova, R.A. Böckmann, S.J. Marrink, D.P. Tieleman, Going Backward: A Flexible Geometric Approach to Reverse Transformation from Coarse Grained to Atomistic Models, J. Chem. Theory Comput. 10 (2014) 676–690. doi:10.1021/ct400617g.
