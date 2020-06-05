Martinize 2 workflow
####################
Martinize 2 is the main command line interface entry point for vermouth.
It does many interesting things which will be explained here later, and in the
end you should end up with a topology for your system.
Graph central in atom recognition/identification

Pipeline
========

Make bonds
----------
:class:`~vermouth.processors.make_bonds.MakeBonds`

Repair graph
------------
:class:`~vermouth.processors.repair_graph.RepairGraph`

:class:`~vermouth.processors.canonicalize_modifications.CanonicalizeModifications`

Resolution transformation
-------------------------
:class:`~vermouth.processors.do_mapping.DoMapping`

Apply Links
-----------
:class:`~vermouth.processors.do_links.DoLinks`

Post processing
---------------
:class:`~vermouth.processors.apply_rubber_band.ApplyRubberBand`

Important command line options
==============================
Martinize2 CLI options
