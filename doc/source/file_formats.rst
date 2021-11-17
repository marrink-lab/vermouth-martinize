File formats
============
VerMoUTH introduces two new file formats. The ``.ff`` format for defining
:ref:`blocks <data:block>`, :ref:`links <data:link>` and :
ref:`modifications <data:modification>`. Note that you can also define blocks
(and basic links) with Gromacs ``.itp`` and ``.rtp`` files. The ``.mapping``
format can be used to define :ref:`mappings <data:mapping>`. Mappings that don't
cross residue boundaries can also be defined using ``.map`` files.

These file formats are still not finalized and subject to change. Therefore
these file formats are not yet documented. If you need to implement (mappings
for) your own residues you'll need to reverse engineer the format from the
existing files.

.ff file format
---------------
Used for defining :ref:`blocks <data:block>`, :ref:`links <data:link>` and
:ref:`modifications <data:modification>`.

.mapping file format
--------------------
Used for defining :ref:`mappings <data:mapping>` for single blocks,
modifications, and block mappings that cross residue boundaries.
