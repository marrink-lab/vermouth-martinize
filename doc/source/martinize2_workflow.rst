Martinize 2 workflow
####################
Pipeline
========
Martinize 2 is the main command line interface entry point for vermouth.
It effectively consists of 6 stages:

1) reading input files
2) repairing the input molecule
3) mapping the input molecule to the desired output force field and resolution
4) applying Links to generate inter-residue interactions
5) post-processing, such as building an elastic network
6) writing output files

We'll describe each stage in more detail here. It is good to bear in mind
however that in all stages the recognition/identification of atoms/particles is
based on their *connectivity* in addition to any atom properties.

Throughout this document, when we refer to an 'edge' we mean a connection
between two nodes in a graph. With 'bonds' we mean a chemical connection
including the corresponding simulation parameters. Similarly, with 'molecule' we
mean a connected graph consisting of atoms and edges. Note that this is not
necessarily the same as a protein chain, since these could be connected through
e.g. a disulphide bridge.

If martinize2 at some point encounters a situation that might result in an
incorrect topology it will issue a warning, and refuse to write output files so
that you are forced to examine the situation, but also see the ``-maxwarn`` CLI
option. The options ``-v`` and ``-vv`` can be used to print more debug output,
while the options ``-write-graph``, ``-write-repair`` and ``-write-canon`` can
be used to write out the system after `Make bonds`_, `Repair graph`_ and
`Identify modifications`_, respectively. All of these can help you track down
what's going wrong where.

1) Read input files
===================

Martinize2 can currently read input structures from .gro and .pdb files. .pdb
files are preferred however, since they contain more information, such as chain
identifiers, and ``TER`` and ``CONECT`` records.

Reading PDB files
-----------------
Reading PDB files is done by :class:`~vermouth.processors.pdb_reader.PDBInput`.
We take into account the following PDB records: ``MODEL`` and ``ENDMDL`` to
determine which model to parse; ``ATOM`` and ``HETATM``; ``TER``, which can be
used to separate molecules; ``CONECT``, which is used to add edges; and ``END``.

Will issue a ``pdb-alternate`` warning if any atoms in the PDB file have an
alternate conformation that is not 'A', since those will always be ignored.

Relevant CLI options: ``-f``; ``-model``; ``-ignore``; ``-ignh``.

Make bonds
----------
Since atom identification is governed by their connectivity we need to generate
bonds in the input structure. Where possible we get them from the input file
such as PDB ``CONECT`` records. Beyond that, edges are added by
:class:`~vermouth.processors.make_bonds.MakeBonds`. By default edges will be
added based on atom names and distances, but this behaviour can be changed via
the CLI option ``-bonds-from``.

To add edges based on atom names the :ref:`data:Block` from the input force
field is used as reference for every residue in the input structure where
possible. This is not possible when a residue contains multiple atoms with the
same name, nor when there is no :ref:`data:Block` corresponding to the residue
[#]_. Note that this will only ever create edges *within* residues.

Edges will be added based on distance when they are close enough together,
except for a few exceptions (below). Atoms will be considered close enough based
on their element (taken from either the PDB file directly, or deduced from atom
name [#]_). The distance threshold is multiplied by ``-bonds-fudge`` to allow
for conformations that are slightly out-of-equilibrium. Edges will not be added
from distances in two cases: 1) if edges could be added based on atom names no
edges will be added between atoms that are not bonded in the reference
:ref:`data:Block`. 2) No edges will be added between residues if one of the
atoms involved is a hydrogen atom. Edges added this way are logged as debug
output.

If your input structure is far from equilibrium and adding edges based on
distance is likely to produce erroneous results, make sure to provide ``CONECT``
records describing at least the edges between residues, and between atoms
involved in modifications, such as termini and PTMs.

Will issue a ``general`` warning when it is requested to add edges based on atom
names, but this cannot be done for some reason. This commonly happens when your
input structure is a homo multimer without ``TER`` record and identical residue
numbers and chain identifiers across the monomers. In this case martinize2
cannot distinguish the atom "N", residue ALA1, chain "A" from the atom "N",
residue ALA1, chain "A" in the next monomer. The easiest solution is to place
strategic ``TER`` records in your PDB file.

Relevant CLI options: ``-bond-from``; ``-bonds-fudge``

.. [#] Based on residue name.
.. [#] The method for deriving the element from an atom name is extremely
   simplistic: the first letter is used. This will go wrong for two-letter
   elements such as 'Fe', 'Cl', and 'Cu'. In those cases, make sure your PDB
   file specified the correct element. See also:
   :func:`~vermouth.graph_utils.add_element_attr`

Annotate mutations and modifications
------------------------------------
As a last step martinize2 allows you to make some changes to your input
structure from the CLI, for example to perform point mutations, or to apply
PTMs and termini. This is done in part by
:class:`~vermouth.processors.annotate_mut_mod.AnnotateMutMod`, and completed by
`Repair graph`_.

The ``-mutate`` option can be used to change the residue name of one or more
residues. For example, you can specify ``-mutate PHE42:ALA`` to mutate all
residues with residue name "PHE" and residue number 42 to "ALA". Or change all
"HSE" residues to "HIS": ``-mutate HSE:HIS``. Mutations can be specified in a
similar way.

The specifications ``nter`` and ``cter`` can be used to quickly refer to all N-
and C-terminal residues respectively [#]_. In addition, the CLI options
``-nter`` and ``-cter`` can be used to change the N- and C-termini. By default
martinize2 will try to apply charged protein termini ('N-ter' and 'C-ter'). If
this is not what you want, for example because your molecule is not a protein,
be sure to provide the appropriate ``-nter`` and ``-cter`` options. You can
specify the modification ``none`` to specify that a residue should not have any
modifications. Note that if you use this for the termini you may end up with
chemically invalid, uncapped, termini.

Relevant CLI options: ``-mutate``, ``-modify``, ``-nter``, ``-cter``, ``-nt``

.. [#] N- and C-termini are defined as residues with 1 neighbour and having a
   higher or lower residue number than the neighbour, respectively. Note that
   this does not include zwitterionic amino acids!
   This also means that if your protein has a chain break you'll end up with
   more termini than you would otherwise expect.

2) Repair the input graph
=========================
Depending on the origin of your input structure, there may be atoms missing, or
atoms may have non-standard names. In addition, some residues may include
modifications such as PTMs.

Repair graph
------------
The first step is to complete the graph so that it contains all atoms described
by the reference :ref:`data:Block`, and that all atoms have the correct names.
These blocks are taken from the input force field based on residue names (taking
any mutations and modifications into account).
:class:`~vermouth.processors.repair_graph.RepairGraph` takes care of all this.

To identify atoms in a residue we consider the
:ref:`graph_algorithms:maximum common induced subgraph` between the residue and
its reference since the residue can be both too small (atoms missing in the
input) and too large (atoms from PTMs) at the same time. Unfortunately, this is
a very expensive operation which scales exponentially with the size of the
residue. So if you know beforehand that your structure contains (very) large
PTMs, such as lipidations, consider specifying those as separate residues.

The maximum common induced subgraph is found using
:class:`~vermouth.ismags.ISMAGS`, where nodes are considered equal if their
elements are equal. Beforehand, the atoms in the residue will be sorted such
that the isomorphism where most atom names correspond with the reference is
found. This sorting also speeds up the calculation significantly, so if you're
working with a system containing large residues consider correcting some of the
atom names.

Will issue an ``unknown-residue`` warning if no Block can be retrieved for a
given residue name. In this case the entire molecule will be removed from the
system.

Identify modifications
----------------------
Secondly, all modifications are identified. `Repair graph`_ will also tag all
atoms it did not recognise, and those are processed by
:class:`~vermouth.processors.canonicalize_modifications.CanonicalizeModifications`.

This is done by finding the solution where all unknown atoms are covered by the
atoms of exactly one :ref:`data:Modification`, where the modification must be an
:ref:`induced subgraph <graph_algorithms:Induced subgraph isomorphism>` of the
molecule. Every modification must contain at least one "anchoring" atom, which
is an atom that is also described by a :ref:`data:Block`. Unknown atoms are
considered to be equal if their element is equal; anchor atoms are considered
equal if their atom name is equal. Because modifications must be
:ref:`induced subgraphs <graph_algorithms:Induced subgraph isomorphism>` of the
input structure there can be no missing atoms!

After this step all atoms will have correct atom names, and any residues that
are include modifications will be labelled. This information is later used
during the :ref:`resolution transformation <martinize2_workflow:3) Resolution transformation>`

An ``unknown-input`` warning will be issued if a modification cannot be
identified. In this case the atoms involved will be removed from the system.

Rebuild coordinates for missing atoms
-------------------------------------
Currently martinize2 is not capable of rebuilding coordinates for missing atoms.

3) Resolution transformation
============================
The resolution transformation is done by
:class:`~vermouth.processors.do_mapping.DoMapping`. This processor will produce
your molecules at the target resolution, based on the available mappings. These
mappings are read from the ``.map`` and ``.mapping`` files available in the
library [#]_. See also :ref:`file_formats:File formats`. In essence these
mappings describe how molecular fragments (atoms and bonds) correspond to a
block in the target force field. We find all the ways these mappings can fit
onto the input molecule, and add the corresponding blocks and modifications to
the resulting molecule.

For a molecular fragment to match the input molecule the atom and residue names
need to match [#]_. This is why we first :ref:`repair <martinize2_workflow:2) Repair the input graph>`
the input molecule so that you only need to consider the canonical atom names
when adding mappings. Mappings defined by ``.mapping`` files can also cross
residue boundaries (where specified).

Edges and interactions within the blocks will come from the target force field.
Edges between the blocks will be generated based on the connectivity of the
input molecule, i.e. if atoms A and B are connected in the input molecule, the
particles they map to in the output force field will also be connected.
Interactions across separate blocks will be added in the next step.

The processor will do some sanity checking on the resulting molecule, and issue
an ``unmapped-atom`` warning if there are modifications in the input molecule
for which no mapping can be found. In addition, this warning will also be issued
if there are any non-hydrogen atoms that are not mapped to the output molecule.
A more serious ``inconsistent-data`` warning will be issued for the following
cases:

- there are multiple modification mappings, which overlap
- there are multiple block mappings, which overlap
- there is an output particle that is constructed from multiple input atoms,
  and some "residue level" attributes (such as residue name and number) are not
  consistent between the constructing atoms.
- there is an atom which maps to multiple particles in the output, but these
  particles are disconnected
- there is an interaction that is being set by multiple mappings

Relevant CLI options: ``-ff``, ``-map-dir``

.. [#] When ``-ff`` (target force field) and ``-from`` (original force field)
   are the same the mappings will be generated automatically.
.. [#] This is only mostly true. All attributes except a few that are not always
   defined must match. Not all attributes (such as 'mass') are defined in all
   cases, depending on the source of the mappings. Note that we also take into
   account that atom names might have changed due to modifications: we use the
   atom name as it is defined by the :ref:`data:Block`.

4) Apply Links
==============
Next interactions *between* residues are added by
:class:`~vermouth.processors.do_links.DoLinks`. We do this based on the concept
of :ref:`Links <data:Link>`, which are molecular fragments that describe
interactions, and which atoms they should apply to. Links are very powerful and
flexible tools, and we use them to generate all interactions that depend on the
local structure of the polymer. For example, all interactions that depend on the
protein sequence or secondary structure are defined by :ref:`Links <data:Link>`.

Links can both add, change and remove interactions and nodes. Because of this,
the order in which links are applied matters for the final topology. We apply
them in the order in which they are defined in the force field files. Therefore
it is important to define links in the order of most general to most specific. A
link is applied in all the places where it fits onto the molecule produced by
:ref:`the mapping step <martinize2_workflow:3) Resolution transformation>`.

For a link to match all its node attributes must match, where the 'order'
attribute is a special case. The order attributes are translated to a
difference in residue numbers, so that nodes 'BB' and '+BB' must have a
difference in residue number of exactly 1 [#]_. Due to the reliance on residue
numbers this can cause complications for non-linear polymers. For those cases
order specifications such as '>' (greater than) and '*' (different from) [#]_
might be useful.

.. [#] Also '-BB' and 'BB', '+BB' and '++BB', etc.
.. [#] Remember that links can overlap! The link ``BB *BB`` will be applied both
   forwards and backwards!

5) Post processing
==================
There can be any number of post processing steps. For example to add an elastic
network, or to generate Go virtual sites. We will not describe their function
here in detail. Instead, see for example
:class:`~vermouth.processors.apply_rubber_band.ApplyRubberBand` and
:class:`~vermouth.processors.go_vs_includes.GoVirtIncludes`.

Relevant CLI options: ``-elastic``, ``-ef``, ``-el``, ``-eu``, ``-ermd``,
``-ea``, ``-ep``, ``-em``, ``-eb``, ``-eunit``, ``-govs-include``,
``-govs-moltype``

6) Write output
===============
Finally, the topology and conformation are written to files (if no warnings were
encountered along the way). Currently martinize2 and VerMoUTH can only write
Gromacs itp files. Martinize2 will write a separate itp file for every unique
molecule in the system.

Relevant CLI options: ``-x``, ``-o``, ``-sep``, ``-merge``
