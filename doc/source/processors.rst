Processor
=========
:class:`Processors <vermouth.processors.processor.Processor>` are relatively
simple. They form the fundamental steps of the martinize2 pipeline. Processors
are called via their :meth:`~vermouth.processors.processor.Processor.run_system`
method. The default implementation of this method iterates over the molecules
in the system, and runs the :meth:`~vermouth.processors.processor.Processor.run_molecule`
method on them. This means that implementations of Processors must implement
either a ``run_system`` method, or a ``run_molecule`` method. If the processor
can be run on independent molecules the ``run_molecule`` method is preferred;
``run_system`` should be used only for cases where the problem at hand cannot
be separated in tasks-per-molecule.

In their ``run_molecule`` method Processor implementations are free to either
modify :class:`molecules <vermouth.molecule.Molecule>` or create new ones.
Either way, they must return a :class:`~vermouth.molecule.Molecule`. The
``run_system`` will be called with a :class:`~vermouth.system.System`, which
will be modified in place.
