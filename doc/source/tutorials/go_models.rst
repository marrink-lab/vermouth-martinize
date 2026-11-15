=========
Gō models
=========

The MartiniGō model is a method of maintaining secondary and tertiary structure using native contacts of proteins
to create a `Gō-like model <https://pubs.acs.org/doi/full/10.1021/acs.jctc.6b00986>`_ between beads.
In contrast to an elastic network, the Gō model applies non-bonded interactions between pairs of
beads within a protein based on residue overlap and restricted chemical structural unit criteria.

The latest version of Martinize2 (version ≥ 0.10.0) implements the newest version of the
`Gō model <https://www.biorxiv.org/content/10.1101/2024.04.15.589479v1>`_. In this version of the Gō model, interactions
are mediated through the addition of extra virtual sites on top of backbone beads in the protein. Interactions are in
the form of Lennard-Jones interactions, which are written as an extra file to be included in the protein topology.

The Gō model is described in the help::

 Virtual site based GōMartini:
   -go [GO]              Use Martini Go model. Accepts either an input file from the server, or just provide the flag to
                         calculate as part of Martinize. (default: None)
   -go-multimer          Keep the molecules separate and write one itp per unique molecule, distinguishing intra- and
                         inter-molecular Go contacts. (default: False)
   -go-ff GO_FF          Force field itp file (e.g. martini_v3.0.0.itp) from which the backbone nonbonded parameters are
                         read. Required with -go-multimer. (default: None)
   -go-eps GO_EPS        The strength of the Go model structural bias in kJ/mol. (default: 9.414)
   -go-eps-inter GO_EPS_INTER
                         The strength of the Go bias in kJ/mol between molecules with the same sequence. Only with
                         -go-multimer; defaults to -go-eps. (default: None)
   -go-eps-hetero GO_EPS_HETERO
                         The strength of the Go bias in kJ/mol between molecules with different sequences. Only with
                         -go-multimer; defaults to -go-eps-inter. (default: None)
   -go-shell-width GO_SHELL_WIDTH
                         Tolerance in nm for pooling a Go contact between molecules. The pairs of molecules in which the
                         contact distance is within this tolerance of its shortest value are the reference pairs. The
                         contact is pooled over all pairs of molecules of the same two itps that are as far apart as a
                         reference pair, to within this tolerance. Distances between molecules are measured between the
                         centres of their backbone beads. Its sigma comes from the contact distances in these pairs.
                         Only with -go-multimer; defaults to 0.05. (default: None)
   -go-pooling {union,intersection}
                         Within a molecule, use the Go contacts that are found in any (union) or in every (intersection)
                         copy of that molecule. Between molecules, use the Go contacts that are found in any or every
                         pair of molecules over which they are pooled (see -go-shell-width). Only with -go-multimer;
                         defaults to union. (default: None)
   -go-low GO_LOW        Minimum distance (nm) below which contacts are removed. (default: 0.3)
   -go-up GO_UP          Maximum distance (nm) above which contacts are removed. (default: 1.1)
   -go-res-dist GO_RES_DIST
                         Minimum graph distance (similar sequence distance) below which contacts are removed. (default: 3)
   -go-write-file [GO_WRITE_FILE]
                         Write out contact map to file if calculating as part of Martinize2. (default: None)
   -go-backbone GO_BACKBONE
                         Name of protein backbone bead on which to place a virtual interaction go site (default: BB)
   -go-atomname GO_ATOMNAME
                         Name of the virtual interaction go site atom (default: CA)

To add a Gō model to your protein, the first step is to calculate the contact map of your protein.
The contact map can be obtained in two ways. Firstly, by uploading your pdb structure
to the `web server <http://pomalab.ippt.pan.pl/GoContactMap/>`_, and downloading the associated ``contact_map.out`` file.
Alternatively, the contact map can be calculated directly without the need for
any external processes. While the implementations of the contact algorithm are identical, the Martinize2 implementation
may be relatively slow for larger systems. Typically for proteins with fewer than 1000 residues, the calculation of the
contact map as part of Martinize2 will add up to a minute of extra calculation. Note that while the implementations of
the main algorithm are identical, there may be small differences in the resulting contact map files due to assumptions
the server makes about the format of input pdb files, which the implementation in Martinize2 overcomes. If you want
to check the contact map that Martinize2 has calculated, you can write it out using the ``-go-write-file`` argument.
While the contact map files may have small differences, it is likely that they will still result in the same non-bonded
file outputs, a result of how symmetrical contacts are further identified in the definition of the Gō model.

The Gō model is added to the input system using the ``-go`` argument of martinize2. If you have used a contact map
from the server, give the path to the contact map file as the argument:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -go contact_map.out``

Otherwise the contact map is calculated as part of Martinize2 by just specifying the ``-go`` argument:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -go``


Without any further additions, this will:
 1) Generate virtual sites with the atomname ``CA`` directly on top of all the backbone beads ``BB`` in your protein.
    The atomname and the name of the backbone bead are controlled using the ``-go-atomname`` and ``-go-backbone`` flags
    respectively. Each ``CA`` atom has an underlying atomtype (see below), which has a default name inherited from the
    ``-name`` flag.
 2) Use the contact map to generate a set of non-bonded parameters between specific pairs of ``CA`` atoms in your molecule
    with strength 9.414 kJ/mol (changed through the ``-go-eps`` flag).
 3) Eliminate any contacts which are shorter than 0.3 nm and longer than 1.1 nm, or are closer than 3 residues in the
    molecular graph. These options are flexible through the ``-go-low`` and ``-go-up`` flags.
 4) If the contact map finds any atoms within contact range defined, but are *also* within 3 residues of each other,
    then the contacts are removed. This is defined through the ``-go-res-dist`` flag.

As a result, along with the standard output of martinize2 (*i.e.* itp files for your molecules, a generic .top file,
a coarse grained structure file), you will get two extra files: ``go_atomtypes.itp`` and ``go_nbparams.itp``. The atomtypes
file defines the new virtual sites as atoms for your system, and the nbparams file defines specific non-bonded
interactions between them.

For example, ``go_atomtypes.itp`` looks like any other ``[ atomtypes ]`` directive::

 [ atomtypes ]
 molecule_1 0.0 0 A 0.00000000 0.00000000
 molecule_2 0.0 0 A 0.00000000 0.00000000
 molecule_3 0.0 0 A 0.00000000 0.00000000
 molecule_4 0.0 0 A 0.00000000 0.00000000
 molecule_5 0.0 0 A 0.00000000 0.00000000
 ...

Similarly, ``go_nbparams.itp`` looks like any ``[ nonbond_params ]`` directive (obviously, the exact parameters here
depend on your protein)::

 [ nonbond_params ]
 molecule_17 molecule_0_13 1 0.59354169 9.41400000 ;go bond 0.666228018941817
 molecule_18 molecule_0_14 1 0.53798937 9.41400000 ;go bond 0.6038726468003999
 molecule_19 molecule_0_15 1 0.51270658 9.41400000 ;go bond 0.5754936778307316
 molecule_22 molecule_0_15 1 0.73815666 9.41400000 ;go bond 0.8285528398039018
 molecule_22 molecule_0_18 1 0.54218134 9.41400000 ;go bond 0.6085779754055839
 molecule_23 molecule_0_19 1 0.53307395 9.41400000 ;go bond 0.5983552758317587
 ...

To activate your Gō model for use in Gromacs, the `martini_v3.0.0.itp` master itp needs the additional files included.
The additional atomtypes defined in the ``go_atomtypes.itp`` file should be included at the end of the `[ atomtypes ]`
directive as::


 [ atomtypes ]
 ...
 TX1er 36.0 0.000 A 0.0 0.0
 W  72.0 0.000 A 0.0 0.0
 SW 54.0 0.000 A 0.0 0.0
 TW 36.0 0.000 A 0.0 0.0
 U  24.0 0.000 A 0.0 0.0

 #ifdef GO_VIRT
 #include "go_atomtypes.itp"
 #endif

 [ nonbond_params ]
 P6    P6  1 4.700000e-01    4.990000e+00
 P6    P5  1 4.700000e-01    4.730000e+00
 P6    P4  1 4.700000e-01    4.480000e+00
 ...

Similarly, the nonbonded parameters should be included at the end of the `[ nonbond_params ]`
directive::

 ...
 TX2er  SQ1n  1 3.660000e-01    3.528000e+00
 TX2er  TQ1n  1 3.520000e-01    5.158000e+00
 TX1er   Q1n  1 3.950000e-01    1.981000e+00
 TX1er  SQ1n  1 3.780000e-01    3.098000e+00
 TX1er  TQ1n  1 3.660000e-01    4.422000e+00

 #ifdef GO_VIRT
 #include "go_nbparams.itp"
 #endif

Then in the .top file for your system, simply include `#define GO_VIRT` along with the other files
to be included to active the Gō network in your model.

As a shortcut for writing the include statements above, you can simply include these files in your master
``martini_v3.0.0.itp`` file with the following commands in a bash shell::

 sed -i "s/\[ nonbond_params \]/\#ifdef GO_VIRT\n\#include \"go_atomtypes.itp\"\n\#endif\n\n\[ nonbond_params \]/" martini_v3.0.0.itp

 echo -e "\n#ifdef GO_VIRT \n#include \"go_nbparams.itp\"\n#endif" >> martini_v3.0.0.itp

The Gō model should then be usable in your simulations following the `general protein tutorial <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00677>`_.
But careful! While the Gō model specifies nonbonded interactions, the interactions are only defined
internally for each molecule. This means that if you have multiple copies of your Gō model protein
in the system, the Gō bonds are still only specified internally for each copy of the molecule,
not truly as intermolecular forces in the system as a whole. For more detail on this phenomenon,
see the paper by `Korshunova et al. <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00677>`_. Gō bonds between the
copies of a protein can be generated with ``-go-multimer``, see
:ref:`tutorials/go_models:Gō models for multimers`.


Multiple Gō models in the same system
-------------------------------------

If you have several different proteins that you have martinized with their own Gō models, then several extra steps are
required to ensure that they can be simulated together.

Firstly, when the proteins are coarse grained with Martinize2, the `-name` flag must be used to ensure that all the virtual
sites created for the purposes of the Gō model have unique names and atomtypes. For example::

 martinize2 -f protein.pdb -x cg.pdb -o topol.top -go contact_map.out -name my_protein

will ensure that the atoms created to apply Gō sites to are names `my_protein_{0..n}` for a protein of n residues.

Having martinized the proteins in this way, all the `go_atomtypes.itp` and `go_nbparams.itp` files generated for each
protein should be concatenated into a single file, which may then be included in the master force field file as
described above.

Gō models for multimers
-----------------------

With ``-go``, all molecules in the input are merged into a single molecule, so the chains of a multimer or a fibril
end up in one large itp and copies of a protein cannot share a topology. With ``-go-multimer``, the molecules are kept
separate and martinize2 writes one itp per protein, which all its copies share, while the Gō bonds between the
molecules are kept. The model follows `Korshunova et al. <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00677>`_. For
example::

 martinize2 -f fibril.pdb -o topol.top -x cg_fibril.pdb -ff martini3001 -ss <secondary structure> -go -go-multimer -go-ff martini_v3.0.0.itp

On top of the setup of the Gō model described above, this requires:

 1) The force field file with ``-go-ff``. Only its ``[ nonbond_params ]`` are read, to cancel the regular interaction
    between backbone beads that form a Gō bond between molecules (site D below), so it must be the force field file
    used in the simulation.
 2) Complete chains with unique chain IDs, since residues are identified by their chain ID and residue number, without
    the insertion code. Chains with missing residues give an error, and so do residue numbers that occur twice in a
    chain (for example 52 and 52A).
 3) One secondary structure for all copies of a protein, as they share one itp. With ``-dssp``, DSSP is run once on
    the whole assembly, so that secondary structure formed between molecules, such as the β-sheets of a fibril, is
    found. Residues that still get a different assignment in different copies get the most common assignment over the
    copies (ties go to the first copy), and every change is logged. Copies are the chains that end up sharing an itp:
    a chain that differs in any other way, for example through ``-modify``, keeps its own secondary structure.
    Alternatively, the secondary structure can be given with ``-ss``, for example one obtained from atomistic
    simulations. Copies that are given different secondary structures get separate itps.

Gromacs applies ``[ nonbond_params ]`` per pair of atomtypes, so a Gō bond between atomtypes that copies share acts
both within each copy and between copies, while ``[ exclusions ]`` only act within a molecule. The multimer Gō model
therefore uses up to four virtual sites per residue, all placed on the backbone bead. For a contact between the
residues *i* and *j*:

.. list-table::
   :header-rows: 1

   * - Site
     - Added for
     - Line in ``go_nbparams.itp``
     - Exclusion within the molecule
     - Effect
   * - A
     - intra-molecular contact
     - A\ :sub:`i` - A\ :sub:`j` with +ε (``-go-eps``)
     - none
     - Gō bond within each copy and between copies
   * - B
     - intra-molecular contact
     - B\ :sub:`i` - B\ :sub:`j` with −ε
     - B\ :sub:`i` - B\ :sub:`j`
     - cancels A between copies
   * - C
     - inter-molecular contact
     - C\ :sub:`i` - C\ :sub:`j` with +ε\ :sub:`inter` (``-go-eps-inter`` or ``-go-eps-hetero``)
     - C\ :sub:`i` - C\ :sub:`j` if *i* ≠ *j* and both molecules share an itp
     - Gō bond between molecules
   * - D
     - inter-molecular contact
     - D\ :sub:`i` - D\ :sub:`j` with minus the backbone-backbone interaction from ``-go-ff``
     - as for C
     - cancels the regular backbone interaction between molecules

Site A is the usual Gō virtual site, with the atomname ``CA`` and the atomtype ``<moltype>_<resid>``. The sites B, C and
D are only added where they are needed, with the atomtypes ``<moltype>_<resid>_B``, ``_C`` and ``_D`` and the atomnames
``VWB``, ``VWC`` and ``VWD``. The Gō bonds use the same σ as in the regular Gō model, and as there, the backbone beads of
an intra-molecular contact are excluded from each other. A contact between the same residue of two copies (*i* = *j*)
gets the lines C\ :sub:`i` - C\ :sub:`i` and D\ :sub:`i` - D\ :sub:`i`. As a molecule has only one of each site and a
site does not interact with itself, these lines only act between copies and need no exclusion. Site B is also added for
proteins with a single copy. This allows the output itp file to be used in simulations of multiple copies of a protein
with only intra-molecular Gō bonds.

The output consists of the same files as with ``-go``: ``go_atomtypes.itp`` and ``go_nbparams.itp`` are included
through the force field file as described above, and the ``[ molecules ]`` directive of the .top file lists every
itp with its number of copies. For every itp, martinize2 reports the number of copies, of intra- and
inter-molecular contacts and of added sites.

Compared with a Gō model in which all molecules are merged, the multimer Gō model differs in the following ways:

 1) As the copies of a protein share one itp, their contacts are combined (pooled) per pair of residues. A contact
    between residues *i* and *j* within a molecule that is found in any copy becomes a Gō bond in every copy. A
    contact between molecules becomes a Gō bond between any two copies that come within the cut-off, not only between
    the copies that were in contact in the input structure. The σ of a Gō bond is the mean of the contact's distances
    over the copies, or pairs of copies, in which it was found, and a warning is given if these distances differ by
    more than 0.1 nm. By default, a contact is used if it was found at least once (union). With
    ``-go-pooling intersection``, a contact within a molecule is only used if it was found in every copy, and a contact
    between molecules only if it was found in every pair of molecules over which it is pooled (see the next point).
 2) The same contact between molecules can also be found between molecules that are further apart, for example
    between the next-nearest strands of a fibril, where it is longer. To keep such different kinds of contacts from
    being mixed into one σ, the pairs of molecules over which a contact between residues *i* and *j* is pooled are
    chosen in two steps:

    - first, the pairs of residues *i* and *j* whose distance is within ``-go-shell-width`` (default 0.05 nm) of the
      shortest one are found;
    - then, the contact is pooled over all pairs of molecules of the same itps that are as far apart as the molecules
      of one of these pairs of residues, to within the same width. The distance between two molecules is measured
      between the centres of their backbone beads.

    The σ of the contact is the mean of its distances in these pairs of molecules, and a warning is given if these
    distances differ by more than 0.1 nm. In a fibril, for example, a contact between neighbouring strands is pooled
    over all pairs of neighbouring strands, but not over the pairs of next-nearest strands. If the pairs of residues
    found in the first step lie in pairs of molecules at different distances, the contact is about equally short in
    different kinds of pairs of molecules and cannot be assigned to one of them. A warning is then given, and the
    contact is pooled over the pairs of molecules at all of these distances. Pairs of molecules that are equally far
    apart are treated as equivalent, so the input should be a well-defined structure, and different interfaces are
    best simulated in separate systems with separate itps. In a fibril with several protofilaments, molecules one
    layer apart in neighbouring protofilaments are only slightly further apart than molecules in the same layer, so
    they can end up among the pairs of molecules that a contact is pooled over; intersection may then need a smaller
    width.
 3) Copies share one secondary structure (requirement 3), and their structure dependent side chain dihedrals
    (``SC-BB-BB-SC scFix``) are replaced by their circular mean. A warning is given if this changes the angle of a copy
    by more than 30°. To keep the dihedrals of every chain, use ``-sep``; ``-noscfix`` leaves out the side chain
    corrections altogether.
 4) An exclusion also removes the plain Coulomb term :math:`f q_i q_j/(\varepsilon_r r)`, which site D does not
    cancel. A contact between molecules that joins two charged backbone beads, such as two charged termini, therefore
    gets no Gō bond, and a warning is given for every such contact.
 5) Intrinsically disordered regions given with ``-id-regions`` lose their Gō bonds (sites A, B and C) in all copies
    that share an itp; site D is kept, as the backbone exclusion is kept in a merged Gō model.

With ``-sep``, every molecule gets its own itp, so only the fourth point remains, apart from ``-dssp`` being run on the
whole assembly. Chains merged with ``-merge`` form one molecule, so the contacts between them are intra-molecular. The
warnings of the first four points have the types ``go-contact-spread``, ``go-contact-shells``, ``scfix-spread`` and
``go-charged-contact``, and all warnings of one of these types together count as one towards ``-maxwarn``. After
checking them, you can ignore them with, for example,
``-maxwarn go-contact-spread go-contact-shells scfix-spread go-charged-contact``.

Note on multiple Gō models
--------------------------

While the Gō model is expressed as a set of interactions between Gō sites on a protein, interactions are not
generally extended over all copies of a protein. That is, if a simulation is set up with several copies of a
multimeric protein where the monomers are held together by a Gō model, Gromacs will not permit the multimers
to "fall apart" for monomers to find other monomers that were initially in other complexes. This limitation is
a result of the nature of the Gromacs itp format, with the Gō interactions described within each `[ moleculetype ]`
directive. This issue is discussed more extensively in a recent paper by `Korshunova et al. <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00677>`_.
With ``-go-multimer``, the Gō bonds between molecules act between all copies of a protein, see
:ref:`tutorials/go_models:Gō models for multimers`.



Visualising Gō networks
----------------------------

If you want to look at your Gō network in VMD to confirm that it's been constructed in the
way that you're expecting, the `MartiniGlass <https://github.com/Martini-Force-Field-Initiative/MartiniGlass>`_
package can help write visualisable topologies to view.
