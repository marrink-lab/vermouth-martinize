Adding new residues and links
=============================
Occasionally you may need a topology containing a residue that is not yet
described by the force fields that ship with vermouth. In this case you will
need to create the required data files yourself, and point martinize2 to them
with the `-ff-dir` and `-map-dir` flags. The key thing to remember is that you
will need to add/edit *three* files. You need to describe your new residue in
the input force field (default charmm); the output force field, and the mapping
between the two.

For this example we will add the required data files for a phosphorylated
serine residue (``OC(=O)C(N)COP(=O)(=O)[O-]``). Note that the parameters
presented here are for demonstration purposes only and not fit for actual
science or simulations!

The input force field
---------------------
The input force field is the force field best describing the structure and atom
names in your input PDB file. By default we use the charmm naming scheme. Since
the input force field will only be used to :ref:`repair <martinize2_workflow:2) Repair the input graph>`
your input structure, only the atom names and edges are relevant.

We'll start by creating a force fields folder we can use to create the tutorial
files; and in that folder we need to create a force field named ``charmm``::

  mkdir -p tutorial_ff/charmm

Now we need to add the SEP :ref:`data:block` to our ``charmm`` folder. Let's
put it in the file ``tutorial_ff/charmm/sep.rtp``::

  [ bondedtypes ]
  1 5 9 2 1 3 1 0
  [SEP]
    [ atoms ]
        N   N  0 0
        HN  H  0 1
        CA  C  0 2
        HA  H  0 3
        CB  C  0 4
        HB1 H  0 5
        HB2 H  0 6
        OG  O  0 7
        C   C  0 8
        O   O  0 9
        P   P  0 10
        O1  O  0 11
        O2  O  0 12
        O3  O -1 13
    [ bonds ]
        CB CA
        OG CB
        N  HN
        N  CA
        C  CA
        C  +N
        CA HA
        CB HB1
        CB HB2
        O  C
        OG P
        P  O1
        P  O2
        P  O3

Note that we only add atom names and bonds, since those are all we need. Also
note that we added all hydrogens.

The output force field
----------------------
We also need to add the SEP :ref:`data:block` to the output force field. Of
course we'll use Martini 3 for this. Let's again start by making a martini3001
folder::

    mkdir -p tutorial_ff/martini3001

Now to add the block to ``tutorial_ff/martini3001/sep.ff``::

    ;;; PHOSPHOSERINE
    [ moleculetype ]
    SEP 1

    ; THESE PARAMETERS ARE FOR DEMONSTRATION PURPOSES ONLY. DO NOT USE.
    [ atoms ]
    ; id type resnr residue atom cgnr charge
     1   P2   1     SEP     BB   1     0
     2   Q5n  1     SEP     SC1  1    -1

    [ bonds ]
    BB SC1 1 0.33 5000

At this point we can run ``martinize2 -ff-dir tutorial_ff -list-blocks`` to
check whether our new SEP blocks are picked up.

The mapping
-----------
Finally, we need to add the mapping describing how to get from charmm to
martini3001. We need to make a folder::

  mkdir mappings

In that folder, make a file ``mappings/sep.charmm36.map``::

    [ molecule ]
    SEP

    [ from ]
    charmm

    [ to ]
    martini3001

    [ martini ]
    BB SC1

    [ mapping ]
    charmm

    [ atoms ]
     1     N  BB
     2    HN  BB
     3    CA  BB
     4    HA  !BB
     5    CB  BB SC1
     6   HB1  !SC1
     7   HB2  !SC1
     8    OG  SC1
     9     C  BB
    10     O  BB
    11     P  SC1
    12    O1  SC1
    13    O2  SC1
    14    O3  SC1

A few things are worth noting here. The HA, HB1, and HB2 atoms are mentioned
here, but their mapping weight is 0, due to the exclamation point. In addition,
CB will contribute to BB and SC1 with equal weight.

Ok, this great! At this point we can run ``martinize2``::

    martinize2 -ff-dir tutorial_ff -map-dir mappings -f ala-sep-ala.pdb -x AJA.pdb -o topol.top

And inspect the resulting ``molecule_0.itp`` to make sure our final topology is
correct::

    [ moleculetype ]
    molecule_0 1

    [ atoms ]
    1 Q5  1 ALA BB  1    1
    2 TC3 1 ALA SC1 2  0.0
    3 P2  2 SEP BB  3  0.0
    4 Q5n 2 SEP SC1 3 -1.0
    5 Q5  3 ALA BB  4   -1
    6 TC3 3 ALA SC1 5  0.0

    [ bonds ]
    3 4 1 0.33 5000

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

We can see that we end up with the correct non-bonded parameters for our SEP
residue, the C- and N-termini are looking good, and we have the BB-SC1 bond we
specified.

There is a problem though, there are no bonds (or constraints) connecting the
SEP residue to its neighbouring ALA residues!

The Links
---------
In Vermouth and martinize2 we use :ref:`links <data:link>` to describe interactions
between residues. We need to these to the output force field---in this case
martini3001.

We can add the following to ``tutorial_ff/martini3001/sep.ff``::

    [ link ]
    [ bonds ]
    BB {"resname": "SEP"} +BB {"resname": "ALA"} 1 0.35 4000

    [ link ]
    [ bonds ]
    BB {"resname": "SEP"} -BB {"resname": "ALA"} 1 0.35 4000

    [ link ]
    [ angles ]
    -BB {"resname": "ALA"} BB {"resname": "SEP"} +BB {"resname": "ALA"} 10 100 20

    [ link ]
    [ angles ]
    -BB BB {"resname": "SEP"} SC1 2 100 25

Links are small molecular fragments. For example, the first one consists of 2
BB beads. The first one has to be part of a SEP residue, and the second has to
be part of an ALA residue. In addition, the ``+`` means the second BB has to
have a resid of exactly one higher than the first BB. In our example, this link
will apply a backbone bond between the SEP residue and ALA3.

The second link is almost identical, and applies a backbone bond between ALA1
and SEP. The two angles work in a similar fashion.

This would result in the following topology::

    [ moleculetype ]
    molecule_0 1

    [ atoms ]
    1 Q5  1 ALA BB  1    1
    2 TC3 1 ALA SC1 2  0.0
    3 P2  2 SEP BB  3  0.0
    4 Q5n 2 SEP SC1 3 -1.0
    5 Q5  3 ALA BB  4   -1
    6 TC3 3 ALA SC1 5  0.0

    [ bonds ]
    3 4 1 0.33 5000
    3 5 1 0.35 4000
    3 1 1 0.35 4000

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
    1 3 5 10 100 20
    1 3 4 2 100 25

We now have bonds between the backbone beads, as well as the 2 angles we need.
In this case, since we don't intend to use this residue for anything other than
an ALA-SEP-ALA peptide, we can combine these links::

    [ link ]
    [ atoms ]
    -BB {"resname": "ALA"}
    BB {"resname": "SEP"}
    SC1 {"resname": "SEP"}
    +BB {"resname": "ALA"}
    [ bonds ]
    BB +BB 1 0.35 4000
    BB -BB 1 0.35 4000
    [ angles ]
    -BB BB +BB 10 100 20
    -BB BB SC1 2 100 25

Which will produce the exact same topology.
If you *do* need to add a residue that can be used in any kind of protein
please take a look at how the Martini 3 force field is implemented, and deals
with e.g. the secondary structure dependence.
