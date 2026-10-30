YAML pipeline configuration
===========================

A pipeline configuration determines which processors Martinize2 runs, in
which order, and with which constructor arguments. It also contributes options
to the command-line interface. Configuration is composed first; the resulting
options are parsed, conditions and arguments are resolved, and the executable
pipeline is built before processing begins.

The examples in the basic section illustrate pipeline fragments. Martinize2
combines source and target fragments with its own runtime objects; an isolated
fragment is not necessarily a complete replacement for a Martinize2 run.
Working configurations are available in the
`force-field data directory <https://github.com/marrink-lab/vermouth-martinize/tree/main/vermouth/data/force_fields>`_
and in ``vermouth/data/pipelines``.

Basic usage
-----------

Selecting pipelines
~~~~~~~~~~~~~~~~~~~

Martinize2 selects the source and target force fields and their pipeline
fragments together:

.. code-block:: console

   martinize2 -from charmm -ff martini3001 \
       -f input.pdb -x output.pdb

Each selected force field supplies a ``pipeline.yaml`` in its own directory,
including force fields loaded from an extra force-field directory. Martinize2
selects the source file's ``from`` step and the target file's ``to`` step.
Additional pipeline fragments can extend the configuration. They are included
at the root of the generated configuration.
They are composed before the source and target steps, rather than simply run
after them. The composition rules are described below.

Pipeline names can be found in additional search directories, or a pipeline can
be supplied as a file path:

.. code-block:: console

   martinize2 -pipeline-dir my-pipelines -pipeline custom -f input.pdb
   martinize2 -pipeline path/to/custom.yaml -f input.pdb

The preliminary parser handles selection and other options needed before the
pipeline-specific interface exists. These include ``-from``, ``-ff``,
``-pipeline``, ``-pipeline-dir``, ``-ff-dir``, ``-map-dir``, ``-list-ff``,
``-list-blocks``, ``-v``, ``-maxwarn``, and ``-version``. Other options are
contributed by the composed pipeline.

Steps and execution
~~~~~~~~~~~~~~~~~~~

A configuration has a ``martinize2`` pipeline. Its ordered steps can be
processors or named sub-pipelines; a sub-pipeline contains its own ordered
steps. Each processor receives the system produced by the previous step. A
processor may also modify the system in place.

.. code-block:: yaml

   martinize2:
     steps: !!omap
       - read:
           processor: vermouth.pipeline_processors.ReadSystem
           args:
             path:
               cli: input
           cli:
             flags:
               input:
                 cli: f
                 type: path
                 required: true
       - prepare:
           steps: !!omap
             - vermouth.RepairGraph:
                 args:
                   delete_unknown:
                     value: true

The step key identifies the step within its parent, including during
composition. In the example above the step keys are ``martinize2``, ``read``,
``prepare``, and ``vermouth.RepairGraph``. If no explicit ``processor`` is given,
that step key is also used as the processor's Python import path. An explicit
processor path allows a stable step name and allows the same processor class
to appear under different names. Processor classes are imported when the
pipeline is built. Arguments are passed to their constructors as keyword
arguments; omitted arguments use the constructor's defaults. See
:doc:`processors` for the processors.

A condition on a processor or sub-pipeline determines whether it runs.
Conditions are evaluated when the pipeline is built, using the parsed
command-line values and runtime variables, not repeatedly as the system
changes. A false sub-pipeline condition skips its entire subtree during
execution. Conditions do not make a fragment exempt from validation, and
processor imports and argument resolution still happen when building the
pipeline.

.. code-block:: yaml

   - repair:
       processor: vermouth.RepairGraph
       cli:
         flags:
           repair:
             action: store_true
             default: false
       condition:
         equal:
           cli: repair
           value: true
       log:
         info:
           msg: Repairing the molecular graph
           type: step
       args: {}

The log entry is emitted immediately before an enabled step runs; it is not
emitted when that step is skipped. Its level determines the logging severity,
and its type categorizes the message for Martinize2's logging and warning
handling. The supported levels are debug, info, warning, error, and critical.
Logging does not itself control whether a step runs.

Conditions support equality between exactly two operands drawn from a CLI
value, a runtime variable, and a literal value. ``all`` and ``any`` combine
at least two conditions, and ``not`` negates one. Equality uses Python's
equality semantics; a missing variable is not a supported existence test.

.. code-block:: yaml

   condition:
     all:
       - equal:
           cli: repair
           value: true
       - not:
           equal:
             variable: ff
             value: null

Equality can also compare a CLI value directly with a variable:

.. code-block:: yaml

   condition:
     equal:
       cli: selected
       variable: expected

Command-line options
~~~~~~~~~~~~~~~~~~~~

An option definition contributes to the generated command-line
interface. Its internal name identifies the parsed value in processor
arguments and conditions; it need not be the same as the spelling typed by
the user.

.. code-block:: yaml

   cli:
     flags:
       input:
         cli: f
         aliases: [-input]
         type: path
         required: true
         help: Input structure
       ignore_h:
         cli: ignh
         action: store_true
         default: false
         help: Ignore hydrogen atoms
   steps: !!omap
     - vermouth.pipeline_processors.ReadSystem:
         args:
           path:
             cli: input
           ignh:
             cli: ignore_h

Here the user supplies ``-f input.pdb`` or ``-input input.pdb``; the
processor receives the parsed path as its ``path`` argument. The primary
spelling gets a single-dash prefix; aliases are complete option strings.
Argparse interprets actions, defaults, required options, choices, argument
counts, and help text.

Flags can be grouped for help output. A mutually exclusive group additionally
lets argparse reject incompatible choices. Grouping does not create a
separate namespace or change visibility within the pipeline.

.. code-block:: yaml

   cli:
     groups:
       - title: Structure handling
         exclusive_groups:
           - flags:
               dssp:
                 nargs: "?"
               ss:
                 type: str

An option can be declared on a pipeline, sub-pipeline, or processor step. It
is visible in that scope and its descendants, including the condition on
the declaring step. A flag declared on one sibling is not visible to another
sibling, even though both contribute to the same command-line interface.
Declare a shared flag on their nearest common parent, or repeat the identical
definition in each sibling. Repeated definitions of the same internal name
that remain after composition must have fully identical settings; the generated
interface adds the option only once.

Arguments and runtime variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each processor argument has one value source: a literal from the
configuration, a parsed command-line value, or a runtime variable. These
sources determine the actual Python values passed to the constructor.

.. code-block:: yaml

   variables: [ff, mappings]
   steps: !!omap
     - vermouth.DoMapping:
         args:
           to_ff:
             variable: ff
           mappings:
             variable: mappings
           delete_unknown:
             value: true

Variables are names for objects supplied by the program building the pipeline;
their declarations do not create or assign those objects. Martinize2 supplies
``source_ff`` for the source force field, ``ff`` for the target force field,
and ``mappings`` for the mappings. The example uses the latter two in a
target-side scope. A custom embedding of the pipeline builder can supply
different objects by name.

Like flags, variable declarations are visible in their own scope and
descendants, not across siblings. An argument or condition must refer to a
variable declared in scope, and the builder must actually supply a value
under that name. All scopes refer to the same runtime value for a given name;
declarations do not create scope-specific values. Declaring a name does not
make an arbitrary new runtime object available. For include-specific
renaming, see below.

Literal values are passed through as YAML values. A declarative path is an
exception: it becomes a Python path resolved against the file that supplied
it, or against Vermouth's data directory. This provenance is retained when a
fragment is included.

.. code-block:: yaml

   args:
     input_file:
       value:
         $path: inputs/reference.pdb
     data_file:
       value:
         $data: mappings/example.map
     label:
       value: inputs/reference.pdb

The first path is relative to the declaring YAML file; the second must stay
inside the data directory. The label remains an ordinary string. Absolute
paths and parent-directory components are allowed with ``$path``.

Configuration invariants
~~~~~~~~~~~~~~~~~~~~~~~~

Within each pipeline, step keys must be unique. Two distinct steps may still
use the same processor class. CLI and variable references in arguments and
conditions must be declared in the same or an enclosing scope; a definition
elsewhere in the global CLI is not sufficient. Repeated CLI definitions are
checked for consistency across the composed pipeline. Conditions must have
one supported operator and equality must have exactly two operands.

When a document declares a schema, it is validated *after* its includes have
been composed. The bundled schema also constrains, for example, argument
sources and CLI definitions. Schema paths are relative to the document that
declares them. The generated Martinize2 document uses the bundled schema, and
Martinize2 validates the composed references before building the interface.
Invalid imports, missing runtime values, and invalid constructor arguments
can additionally fail when the pipeline is built.

Advanced composition
--------------------

Includes and precedence
~~~~~~~~~~~~~~~~~~~~~~~

An include brings a selected mapping into the mapping containing the
directive. Includes are expanded depth-first, in their listed order, before
the local content is applied. Later includes take precedence over earlier
ones, and local content takes precedence over all includes, according to the
merge rules below. Include cycles are errors.

.. code-block:: yaml

   martinize2:
     $include:
       - common.yaml:martinize2
       - optional.yaml:martinize2
     steps: !!omap
       - vermouth.SortMoleculeAtoms:
           args: {}

An include without a fragment selects the whole document, so it belongs at a
compatible document-level mapping. Selecting the pipeline mapping instead
allows it to be composed into another pipeline. Includes can also be used
in other mappings, such as a step, its CLI, or its arguments. The selected
fragment must be a mapping, and the result must be valid at its destination.

Paths and fragments
~~~~~~~~~~~~~~~~~~~

For an include, a relative file path is first resolved relative to the
including YAML file. Otherwise the loader checks the supplied path and
searches the including directory, additional pipeline directories, the
bundled pipeline directory, and finally bundled force-field pipelines by
name. Additional ``-pipeline`` selections use the same name/path lookup,
without an including file's relative-path context.

The part after the first colon selects within the fully composed included
document. Components traverse mapping keys; numeric indexes select entries in
order. A step can therefore be selected by its key or by its zero-based
position:

.. code-block:: yaml

   $include:
     - common.yaml:martinize2.steps.read_input.args
     - common.yaml:martinize2.steps[0].args

These select the same mapping if ``read_input`` is the first step. Bracketed
selectors are numeric, not step names. A dotted component cannot represent a
single key containing a dot; use a numeric step index where appropriate.
Selectors do not import the selected fragment's parent scope: declarations
outside the fragment must be supplied by the destination scope.

Variable renaming
~~~~~~~~~~~~~~~~~

An include can adapt a fragment's variable names to objects available in its
destination. Renaming applies to declarations and references in the selected
fragment, not to the including document or other includes. Unlisted names
stay unchanged; attempting to rename a name absent from the fragment is an
error.

.. code-block:: yaml

   martinize2:
     variables: [source_ff]
     steps: !!omap
       - read:
           $include:
             - path: source.yaml:martinize2.steps.from.steps.read
               rename_variables:
                 ff: source_ff

For example, Martinize2 renames ``ff`` to ``source_ff`` when it imports the
source force field's ``from`` step, leaving the target's ``ff`` references
unchanged. Renaming is not a mechanism for creating a runtime value; the
destination name must still be declared in scope and supplied by the builder.

Merge and replace
~~~~~~~~~~~~~~~~~

The default strategy is recursive merge. Existing mappings are merged by key;
an incoming scalar or list replaces the existing value. A new key is appended,
while an existing key keeps its position. Consequently, matching step keys
modify the same step rather than adding a second execution. Lists such as
variable declarations and CLI groups are replaced, not concatenated.

.. code-block:: yaml

   # base.yaml
   martinize2:
     steps: !!omap
       - repair:
           processor: vermouth.RepairGraph
           args:
             delete_unknown:
               value: false
             include_graph:
               value: true

   # derived.yaml
   martinize2:
     $include:
       - base.yaml:martinize2
     steps: !!omap
       - repair:
           args:
             delete_unknown:
               value: true

The resulting repair step keeps the processor and ``include_graph`` from the
base and uses the new ``delete_unknown`` value. In contrast, the replace
strategy clears the existing mapping at the point where it is applied before
adding the incoming keys. It does not replace the entire document unless
applied at that level.

.. code-block:: yaml

   martinize2:
     $include:
       - base.yaml:martinize2
     steps: !!omap
       - repair:
           args:
             $strategy: replace
             delete_unknown:
               value: true

Now the repair step has only the displayed argument. Replacement is especially
important when changing an argument's source: ordinary merge would retain
the old source as well, producing an invalid argument. Replace that argument
mapping, or explicitly remove the old source.

.. code-block:: yaml

   args:
     path:
       $strategy: replace
       value: input.pdb

Positioning and removing entries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New mapping entries normally appear after existing entries. An insertion
directive on a new entry places it immediately before or after an existing
key in the same mapping. This is particularly useful for ordered steps:

.. code-block:: yaml

   martinize2:
     $include:
       - base.yaml:martinize2
     steps: !!omap
       - average:
           processor: vermouth.DoAverageBead
           $insert_after: repair
           args: {}
       - obsolete: $remove

The anchor must already exist when the entry is merged; it is a sibling key,
not a processor path in another pipeline. With both before and after anchors,
the anchors must be adjacent and specify the same insertion slot. If the
incoming key already exists, it is merged in place, not moved: any supplied
anchors must already describe its immediate position or composition fails.
Missing anchors are errors.

Removing a key deletes its entire value, including any nested steps. Removing
a missing key has no effect. Removal and insertion are applied during
composition, so use them in a fragment that is merged into a mapping containing
the relevant keys; they are not executable processor settings.

Directive names and the removal marker start with a dollar sign. To preserve
a literal leading dollar sign in a YAML string or key rather than treating it
as a composition directive, double it:

.. code-block:: yaml

   args:
     label:
       value: $$remove

The processor receives the literal string ``$remove``.
