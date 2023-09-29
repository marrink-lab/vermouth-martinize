Adding new modifications
========================
In :doc:`/tutorials/6_adding_residues_links/index` we added a whole new
:ref:`data:Block` in order to describe a phosphoserine residue.
This had the (dis)advantage that we had to redefine all the default serine
interactions and parameters as well as the inter-residue
:ref:`links <data:Link>`. There must be a better way!

Fortunately there is. Rather than describing a whole new SEP
:ref:`data:Block` and all that entails we can instead describe
just the way the phosphorylation *modified* the normal SER residue. This is
exactly what :ref:`modifications <data:Modification>` are for. As before, we
need to add the new modification in 3 places: input force field, output force
field, and mapping between the two. Note that the parameters
presented here are for demonstration purposes only and not fit for actual
science or simulations!

The input force field
---------------------
During :ref:`repair <martinize2_workflow:repair graph>` the regular SER atoms
will be repaired, the missing hydrogen (HG) will be added (!), and the phosphate
atoms will be annotated as being "nonstandard". During
:ref:`martinize2_workflow:Identify modifications` the :ref:`processors:Processor`
will try to identify these tagged atoms by finding a minimal set of
modifications that describe all relevant atoms. For a modification to apply
here it must be subgraph isomorphic to the input structure.

If we run ``martinize2 -f ala-sep-ala.pdb -o topol.top -x AJA.pdb`` we get, as
expected, the warning that not all modifications could be identified::

  WARNING - unknown-input - Could not identify the modifications for residues ['SER3'], involving atoms ['21-O1', '22-O2', '23-O3', '24-P']

So let's define the modification in ``forcefields/charmm/modification.ff``::

    ; THESE PARAMETERS ARE FOR DEMONSTRATION PURPOSES ONLY. DO NOT USE.
    [ modification ]
    SER-phos
    [ atoms ]
    O1 {"element": "O", "PTM_atom": true}
    O2 {"element": "O", "PTM_atom": true}
    O3 {"element": "O", "PTM_atom": true}
    P  {"element": "P", "PTM_atom": true}
    OG {"element": "O"}
    HG {"element": "H", "replace": {"atomname": null}}
    [ edges ]
    OG P
    P O1
    P O2
    P O3

As before, the input force field does not define any MD parameters or
interactions. This modification contains nodes and edges. The edges are not
very interesting, and just define the connections between nodes. Nodes on the
other hand define 2 things: 1) the atom name as is should be (first column),
and 2) any constraints the node must satisfy during the subgraph isomorphism
as a JSON formatted mapping. The constraints should define at least 2
properties: the ``element``, and ``PTM_atom``. The element property is
self explanatory, but the ``PTM_atom`` needs more explanation.

Modifications contain *2* types of nodes:

#. Nodes that are already described by the parent block (``PTM_atom`` is false, this is the default). We call these nodes "anchors".
#. Nodes that are not yet described by the parent block (``PTM_atom`` is true).

In addition, it's worth noting that :ref:`repair <martinize2_workflow:repair graph>`
reconstructed the HG atom (see ``-write-repair``) since it's not in the input
PDB. We use the "replace" property to describe all node attributes that need
to change because of this modification. In this case we indicate that this atom
should be removed again, by setting its atomname to "null". You can use the
"replace" property to change *any* node property, including *e.g.* atom type
and charge.

The output force field
----------------------
We have to add a similar modification for the output force field in
``forcefields/martini3001/modification.ff``::

    [ modification ]
    ; THESE PARAMETERS ARE FOR DEMONSTRATION PURPOSES ONLY. DO NOT USE.
    SER-PO4
    [ atoms ]
    BB  {"PTM_atom": false}
    SC1 {"PTM_atom": false, "resname": "SER", "replace": {"atype": "Q5n", "charge": -1}}
    [ bonds ]
    BB SC1 1 0.33 5000

Nothing new here compared to the modification for the input force field. Note
that here we *do* define the simulation parameters, and we define a bond.

The mapping
-----------
Finally, we need to add the mapping describing how to get from charmm to
martini3001 in ``mappings/SEP.mapping``::

    ; THESE PARAMETERS ARE FOR DEMONSTRATION PURPOSES ONLY. DO NOT USE.
    [ modification ]
    [ from ]
    charmm
    [ to ]
    martini3001
    [ from blocks ]
    SER-phos
    [ to blocks ]
    SER-PO4
    [ from nodes ]
    N
    HN
    CA
    HA
    C
    O
    CB
    HB1
    HB2
    [ from edges ]
    N  HN
    N  CA
    CA HA
    CA C
    C  O
    CA CB
    CB HB1
    CB HB2
    CB OG
    [ mapping ]
    N   BB
    HN  BB
    CA  BB
    HA  BB 0
    C   BB
    O   BB
    CB  BB
    CB  SC1
    HB1 SC1 0
    HB2 SC1 0
    OG  SC1
    P   SC1
    O1  SC1
    O2  SC1
    O3  SC1

Firstly, notice that this is a different file format than the backwards format
we used before. In this case we have to define between which force fields we're
going to define a mapping (``charmm`` and ``martini3001``), and between which
modifications (or blocks) (``SER-phos`` and ``SER-PO4``). This mapping has to
define how to map the phosphate moiety (at least). This moiety will be mapped
to the SC1 bead, so we will need to describe the complete mapping for that bead.
In addition this mapping affects the mapping of the BB bead (since CB will now
also contribute in part to it).

The charmm modification already define some nodes (see above), but not all the
nodes required to describe the complete mapping for the BB and SC1 nodes, so
these need to be described under `from nodes` and `from edges`. Finally, the
actual mapping section should be self explanatory.

Now if we run ``martinize2 -f ala-sep-ala.pdb -x AJA.pdb -o topol.top -ff-dir forcefields/ -map-dir mappings/``
we see ``INFO - general - Applying modification mapping ('SER-phos',)``

Now we need to check the produced itp file::

    ; THESE PARAMETERS ARE FOR DEMONSTRATION PURPOSES ONLY. DO NOT USE.
    [ atoms ]
    1 Q5  1 ALA BB  1   1
    2 TC3 1 ALA SC1 2 0.0
    3 P2  2 SER BB  3 0.0
    4 Q5n 2 SER SC1 4  -1
    5 Q5  3 ALA BB  5  -1
    6 TC3 3 ALA SC1 6 0.0

    [ bonds ]
    3 4 1 0.33 5000

    ; Backbone bonds
    1 3 1 0.350 4000
    3 5 1 0.350 4000

    #ifdef FLEXIBLE
    ; Side chain bonds
    1 2 1 0.270 1000000
    5 6 1 0.270 1000000
    #endif

    [ constraints ]
    #ifndef FLEXIBLE
    ; Side chain bonds
    1 2 1 0.270
    5 6 1 0.270
    #endif

    [ angles ]
    ; BBB angles
    1 3 5 10 127 20

    ; BBS angles regular martini
    1 3 4 2 100 25
    3 5 6 2 100 25

    ; First SBB regular martini
    2 1 3 2 100 25

What we see here is that the atom type and bond we specified in the
modification have been applied, and we can also no longer see the BB-SC1 bond
that comes with the normal serine residue (``BB SC1 1 0.287 7500``) is no
longer present. In addition, we find the usual backbone/protein interactions.
