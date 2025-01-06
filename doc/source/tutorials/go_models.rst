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

 Virtual site based GoMartini:
  -go [GO]              Use Martini Go model. Accepts either an input file from the server, or just provide the flag to
                        calculate as part of Martinize. (default: None)
   -go-eps GO_EPS        The strength of the Go model structural bias in kJ/mol. (default: 9.414)
   -go-moltype GOVS_MOLTYPE
                         Set the name of the molecule when using Virtual Sites GoMartini. (default: molecule_0)
   -go-low GO_LOW        Minimum distance (nm) below which contacts are removed. (default: 0.3)
   -go-up GO_UP          Maximum distance (nm) above which contacts are removed. (default: 1.1)
   -go-res-dist GO_RES_DIST
                         Minimum graph distance (similar sequence distance) below which contacts are removed. (default: 3)
  -go-write-file [GO_WRITE_FILE]
                        Write out contact map to file if calculating as part of Martinize2. (default: None)

To add a Gō model to your protein, the first step is to calculate the contact map of your protein.
The contact map can be obtained in two ways. Firstly, by uploading it
to the `web server <http://pomalab.ippt.pan.pl/GoContactMap/>`_, and downloading the associated ``contact_map.out`` file.
Alternatively, with a version of Martinize2 ≥ 0.13.0 the contact map can be calculated directly without the need for
any external processes. While the implementations of the contact algorithm are identical, the Martinize2 implementation
may be relatively slow for larger systems. Typically for proteins with fewer than 1000 residues, the calculation of the
contact map as part of Martinize2 will add up to a minute of extra calculation. Note that while the implementations of
the main algorithm are identical, there may be small differences in the resulting contact map files due to assumptions
the server makes about the format of input pdb files, which the implementation in Martinize2 overcomes. If you want
to check the contact map that Martinize2 has calculated, you can write it out using the ``-go-write-file`` argument.
While the contact map files may have small differences, it is likely that they will still result in the same non-bonded
file outputs, a result of how symmetrical contacts are further identified in the definition of the Gō model.

The go model is then applied to the protein using the ``-go`` argument of martinize2. If you have used a contact map
from the server, give the path to the contact map file as the argument:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -go contact_map.out``

Otherwise the contact map is calculated as part of Martinize2 by just specifying the ``-go`` argument:

``martinize2 -f protein.pdb -o topol.top -x cg_protein.pdb -ff martini3001 -dssp -go``


Without any further additions, this will:
 1) Generate virtual sites with the atomname ``CA`` directly on top of all the backbone beads in your protein.
    Each ``CA`` atom has an underlying atomtype (see below), which has a default name specified using the
    ``-go-moltype`` flag.
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
 molecule_0_1 0.0 0 A 0.00000000 0.00000000
 molecule_0_2 0.0 0 A 0.00000000 0.00000000
 molecule_0_3 0.0 0 A 0.00000000 0.00000000
 molecule_0_4 0.0 0 A 0.00000000 0.00000000
 molecule_0_5 0.0 0 A 0.00000000 0.00000000
 ...

Similarly, ``go_nbparams.itp`` looks like any ``[ nonbond_params ]`` directive (obviously, the exact parameters here
depend on your protein)::

 [ nonbond_params ]
 molecule_0_17 molecule_0_13 1 0.59354169 9.41400000 ;go bond 0.666228018941817
 molecule_0_18 molecule_0_14 1 0.53798937 9.41400000 ;go bond 0.6038726468003999
 molecule_0_19 molecule_0_15 1 0.51270658 9.41400000 ;go bond 0.5754936778307316
 molecule_0_22 molecule_0_15 1 0.73815666 9.41400000 ;go bond 0.8285528398039018
 molecule_0_22 molecule_0_18 1 0.54218134 9.41400000 ;go bond 0.6085779754055839
 molecule_0_23 molecule_0_19 1 0.53307395 9.41400000 ;go bond 0.5983552758317587
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
see the paper by `Korshunova et al. <https://pubs.acs.org/doi/10.1021/acs.jctc.4c00677>`_.


Visualising Go networks
----------------------------

If you want to look at your Gō network in VMD to confirm that it's been constructed in the
way that you're expecting, the `MartiniGlass <https://github.com/Martini-Force-Field-Initiative/MartiniGlass>`_
package can help write visualisable topologies to view.
