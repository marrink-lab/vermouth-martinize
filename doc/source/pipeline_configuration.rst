YAML pipeline configuration
===========================

Overview
--------

Martinize2 pipelines define which processors are run, their order, their
arguments, and the conditions under which they are executed.

Pipeline configurations can be written in YAML. The YAML files are read by
the :class:`~vermouth.pipeline.PipelineConfigBuilder` and converted into an internal pipeline
configuration. This configuration is then used to construct the command-line
interface and the executable pipeline.

A pipeline can be composed from multiple YAML files. This makes it possible
to separate general processing steps from force-field-specific or optional
steps.

Pipeline builders
-----------------

The pipeline configuration is processed by three builder classes:

:class:`~vermouth.pipeline.PipelineConfigBuilder`
    Loads one or more YAML pipeline files and combines them into a single
    pipeline configuration.

:class:`~vermouth.pipeline.CLIBuilder`
    Builds the command-line interface from the CLI options defined in the
    pipeline configuration and parses the provided command-line arguments.

:class:`~vermouth.pipeline.PipelineBuilder`
    Resolves CLI values, runtime variables, and conditions in the pipeline
    configuration and constructs the executable pipeline.

Running Martinize2 with a pipeline
----------------------------------

Pipeline files are selected with the ``-pipeline`` option::

    martinize2 \
        -pipeline charmm martini3001 \
        -inpath input.pdb \
        -outpath output.pdb

Pipeline names such as ``charmm`` and ``martini3001`` are resolved from the
default pipeline directory. The default pipeline configurations distributed
with Vermouth are located in ``vermouth/data/pipelines``. 
A path to a custom YAML file can also be used.

Complete, working configurations can be found in the
`pipeline configuration directory <https://github.com/marrink-lab/vermouth-martinize/tree/main/vermouth/data/pipelines>`_.
These can be used as examples when writing new pipeline fragments.

Multiple pipeline files
-----------------------

Multiple YAML files can be supplied after ``-pipeline``::

    martinize2 \
        -pipeline charmm water martini3001 \
        -inpath input.pdb \
        -outpath output.pdb

The configurations are combined in the order in which they are provided.
Processor steps are appended, command-line flags are combined, and variables
are namespaced per YAML file.

Pipeline structure
------------------

A pipeline configuration contains a root ``martinize2`` section. It can define
variables, command-line flags, command-line groups, and processor steps.

A simplified example is shown below:

.. code-block:: yaml

  martinize2:
     variables:
       - mappings
     cli_flags:
       elastic:
         action: store_true
         default: false
         help: Generate an elastic network.
       inpath:
         type: str
         help: Path to read
     steps:
       - vermouth.pipeline_processors.ReadSystem:
           args:
             inpath:
               cli: inpath
       - vermouth.DoMapping:
           condition:
             equal:
               cli: elastic
               value: true
           args:
             mappings:
               variable: mappings

Nested steps
~~~~~~~~~~~~

Steps can contain additional ``steps``, allowing pipeline configurations
to be nested. The structure is recursive, so nested groups are handled in
the same way as the top-level pipeline.

For example:

.. code-block:: yaml

   martinize2:
     steps: !!omap
       - prepare_system:
           steps: !!omap
             - vermouth.pipeline_processors.ReadSystem:
                 args:
                   path:
                     value: input.pdb
             - vermouth.RepairGraph:
                 args:
                   delete_unknown:
                     value: true
                   include_graph:
                     value: false

       - vermouth.DoMapping:
           args: {}

       - post_process:
           steps: !!omap
             - vermouth.SortMoleculeAtoms:
                 args: {}

Processor steps
---------------

Each item in ``steps`` defines a processor (or sub-pipeline) and its configuration. Processors
are executed in the order in which they occur in the YAML file.

The key is the import path of the processor:

.. code-block:: yaml

   - vermouth.DoMapping:
       args:
         mappings:
           variable: mappings

For more information about Vermouth processors, see :doc:`processors`.

Processor arguments
-------------------

Arguments are defined under ``args``. An argument can receive its value from:

* a fixed value;
* a command-line option;
* a pipeline variable.

Fixed value
~~~~~~~~~~~

A fixed value is passed directly from the YAML configuration to the processor.

.. code-block:: yaml

   args:
     force_constant:
       value: 700

Command-line value
~~~~~~~~~~~~~~~~~~

A command-line value is taken from the option provided by the user through the CLI.

.. code-block:: yaml

   args:
     force_constant:
       cli: rb_force_constant

Variable value
~~~~~~~~~~~~~~

Variables can be used for runtime values or Python objects that cannot be
represented directly in YAML. For example, mappings or force-field objects
can be created by Martinize2 and referenced from the pipeline configuration.

.. code-block:: yaml

   args:
     mappings:
       variable: mappings

Command-line flags
------------------

Pipeline files can define command-line options under ``cli_flags``:

.. code-block:: yaml

   cli_flags:
     elastic:
       action: store_true
       default: false
       help: Generate an elastic network.

The :class:`~vermouth.pipeline.PipelineConfigBuilder` reads these definitions into the internal
pipeline configuration. The :class:`~vermouth.pipeline.CLIBuilder` uses this configuration to
construct the Martinize2 command-line interface.

Hardcoded command-line options
------------------------------

Most Martinize2 command-line options are defined by the pipeline YAML files
and are added dynamically by the :class:`~vermouth.pipeline.CLIBuilder`. A small number of options
are defined directly by Martinize2 because they are required before the
pipeline configuration and the dynamic command-line interface can be built.

These options are:

``--pipeline``
    Selects one or more pipeline YAML files.

``--override``
    Specifies an optional YAML file containing pipeline overrides.

``--pipeline-dir``
    Adds a directory in which pipeline YAML files are searched.

``-extra_ff_dir``
    Adds an additional force-field directory.

``-extra_map_dir``
    Adds an additional mapping directory.

``-list_ff``
    Lists the available force fields.

Conditions
----------

A processor can be enabled conditionally with the ``condition`` field.

The supported condition types are:

* ``equal``;
* ``has_variable``;
* ``all``;
* ``any``;
* ``not``.

For example:

.. code-block:: yaml

   condition:
     equal:
       cli: elastic
       value: true

Multiple conditions can be combined:

.. code-block:: yaml

   condition:
     all:
       - equal:
           cli: elastic
           value: true
       - not:
           equal:
             cli: go
             value: true

Overrides
---------

An override file can modify a combined pipeline configuration without editing
the original pipeline files:

.. code-block:: console

   martinize2 \
       -pipeline charmm water martini3001 \
       -override change.yaml \
       -inpath input.pdb \
       -outpath output.pdb

Override entries target processor IDs. If a processor does not define an
explicit ``id``, its processor name is used as the default ID.

Changing values
~~~~~~~~~~~~~~~

.. code-block:: yaml

   override:
     elastic:
       args:
         force_constant:
           value: 500

Removing values
~~~~~~~~~~~~~~~

The special value ``$remove`` removes a key:

.. code-block:: yaml

   override:
     elastic:
       cli_flags:
         rb_force_constant:
           help: "$remove"

Merge and replace strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dictionaries are merged by default. If an override contains one of the
argument source keys ``cli``, ``value``, or ``variable``, the default strategy
is ``replace``.

The strategy can be selected explicitly with ``$strategy``:

.. code-block:: yaml

   override:
     elastic:
       cli_flags:
         rb_force_constant:
           $strategy: replace
           default: 500

Inserting processors
~~~~~~~~~~~~~~~~~~~~

New processors can be inserted before or after an existing processor:

.. code-block:: yaml

   override:
     sort_after_mapping:
       processor: vermouth.SortMoleculeAtoms
       $insert_after: vermouth.DoMapping
       args: {}

If the target processor is not unique, explicit processor IDs must be used.

Custom pipeline directories
---------------------------

Additional directories containing pipeline YAML files can be supplied with
``-pipeline-dir``.

A custom pipeline file can also be provided directly:

.. code-block:: console

   martinize2 \
       -pipeline path/to/custom_pipeline.yaml \
       -inpath input.pdb \
       -outpath output.pdb

YAML syntax reference
---------------------

This section describes the complete YAML structure understood by the pipeline
configuration code. Complete, working configurations are distributed in
``vermouth/data/pipelines`` and can be used as examples when writing new
pipeline fragments.

Top-level structure
~~~~~~~~~~~~~~~~~~~

A pipeline YAML file must contain a ``martinize2`` mapping::

    martinize2:
      variables: []
      cli_flags: {}
      cli_groups: []
      steps: []

The ``martinize2`` mapping, and nested pipeline mappings below it, can contain
the following fields:

``variables``
    A list of runtime variable names that can be referenced by ``args`` and
    ``condition`` entries.

``cli_flags``
    A mapping of internal command-line option names to argparse option
    definitions.

``cli_groups``
    A list of mutually exclusive command-line option groups.

``steps``
    An ordered list of processor steps or nested pipeline groups.

A processor step can additionally contain ``id``, ``args``, and
``condition``. A nested pipeline group contains another ``steps`` mapping and
may also define CLI options, variables, or a condition.

``steps``
~~~~~~~~~

``steps`` is an ordered mapping. Entries are executed in the order in which
they occur. Processor entries use the fully qualified Python import path of
the processor as their key::

    steps: !!omap
      - vermouth.DoMapping:
          args:
            mappings:
              variable: mappings
      - vermouth.DoAverageBead:
          args: {}

The processor class is imported when the pipeline is built.

An entry can instead be a named nested pipeline. A nested pipeline is
identified by the presence of another ``steps`` field::

    - mapping:
        steps: !!omap
          - vermouth.DoMapping:
              args: {}
          - vermouth.DoAverageBead:
              args: {}

Nested pipelines can themselves contain nested pipelines. CLI options and
variables defined in a parent scope are available to its child steps.

``id``
~~~~~~

``id`` is an optional identifier for a processor step::

    - vermouth.pipeline_processors.ElasticWrapper:
        id: elastic
        args: {}

IDs are used by override files to locate a processor. If ``id`` is omitted,
the fully qualified processor name is used as the effective ID. An ID used as
an override target must identify exactly one processor. If the same processor
occurs more than once, explicit unique IDs are required to address the steps
individually.

``cli_flags``
~~~~~~~~~~~~~

``cli_flags`` defines command-line arguments. The mapping key is the internal
name under which the parsed value is stored::

    cli_flags:
      rb_force_constant:
        cli: ef
        type: float
        default: 700
        help: Elastic bond force constant Fc in kJ/mol/nm^2

The following fields are supported by the pipeline configuration layer:

``cli``
    Optional alternative spelling of the command-line flag. For example,
    ``cli: ef`` makes ``-ef`` available while the parsed value is stored as
    ``rb_force_constant``. The internal name is also accepted as a flag.

``type``
    Converts the command-line value. Supported string values are ``str``,
    ``int``, ``float``, ``path``, ``cys_argument``, ``water_bias``,
    ``ignore_resname``, and ``maxwarn``.

``default``
    Value used when the option is not supplied.

``action``
    Argparse action, for example ``store_true`` or ``append``.

``help``
    Help text shown by the command-line parser.

``required``
    Whether the option must be supplied.

``choices``
    Sequence of accepted values.

``nargs``
    Number of command-line values consumed by the option.

``const``
    Constant value used by argparse actions that support it, for example an
    option using ``nargs: '?'``.

Except for ``cli`` and the string-to-type translation performed for ``type``,
the option fields are passed to ``argparse.ArgumentParser.add_argument``.
Therefore argparse-compatible fields can be used where appropriate.

CLI flags may be defined at the ``martinize2`` level or inside nested pipeline
and processor definitions. A CLI option is visible in the scope where it is
defined and in child scopes. When multiple pipeline YAML files are combined,
a CLI flag may occur more than once only when its definitions are identical.

``cli_groups``
~~~~~~~~~~~~~~

``cli_groups`` defines mutually exclusive command-line options. Each group has
a ``flags`` mapping containing the options in the group::

    cli_groups:
      - type: mutually_exclusive
        flags:
          dssp:
            nargs: "?"
            const: true
          ss:
            type: str
            default: null
          collagen:
            action: store_true
            default: false

The currently supported group semantics are mutually exclusive: argparse
rejects a command line that supplies more than one flag from the same group.
The flag definitions use the same argparse option fields as ``cli_flags``,
except that the ``cli`` alias field is not interpreted for flags inside a
``cli_groups`` entry.

``variables``
~~~~~~~~~~~~~

``variables`` lists values that are supplied by Martinize2 at runtime rather
than read from YAML or the command line::

    variables:
      - ff
      - mappings

A variable can be referenced from processor arguments and conditions. When
multiple pipeline files are combined, variable references are namespaced with
the stem of the YAML file. For example, ``ff`` in ``charmm.yaml`` becomes
``charmm.ff`` internally. This allows multiple fragments to define variables
with the same local name without collisions.

``args``
~~~~~~~~

``args`` maps processor constructor argument names to their value source. A
processor argument should use one of the following source fields.

``value``
    A fixed YAML value::

        args:
          delete_unknown:
            value: true

``cli``
    The value of a command-line option::

        args:
          path:
            cli: inpath

``variable``
    A runtime variable::

        args:
          force_field:
            variable: ff

CLI and variable references must be defined in the current scope or a parent
scope. These references are validated before the executable pipeline is
constructed.

``condition``
~~~~~~~~~~~~~

``condition`` controls whether a step is executed. A condition mapping must
contain exactly one condition operator. The following operators are supported.

``equal``
    Compare a CLI option or runtime variable to a fixed value::

        condition:
          equal:
            cli: elastic
            value: true

    A variable can be compared instead by replacing ``cli`` with
    ``variable``::

        condition:
          equal:
            variable: some_variable
            value: some_value

``has_variable``
    Test whether the object referenced by ``variable`` contains a force-field
    variable with the specified ``key``::

        condition:
          has_variable:
            variable: ff
            key: bondedtypes

``all``
    Evaluate to true only if all nested conditions are true::

        condition:
          all:
            - equal:
                cli: elastic
                value: true
            - equal:
                cli: go
                value: null

``any``
    Evaluate to true if at least one nested condition is true::

        condition:
          any:
            - equal:
                cli: elastic
                value: true
            - equal:
                cli: another_option
                value: true

``not``
    Negate one nested condition::

        condition:
          not:
            equal:
              cli: dssp
              value: null

``all``, ``any``, and ``not`` can be nested to construct more complex
conditions. CLI options and variables referenced by conditions are validated
before pipeline construction.

Override YAML syntax reference
------------------------------

An override file has an ``override`` mapping. Each key below ``override`` is
the ID of an existing processor step, or the ID assigned to a newly inserted
step::

    override:
      elastic:
        args:
          rb_force_constant:
            value: 500

Existing steps
~~~~~~~~~~~~~~

For an existing step, the mapping below its ID is merged into that processor's
configuration. Ordinary dictionaries are merged recursively and ordinary
values and lists are replaced.

``$remove``
~~~~~~~~~~~

The special string ``$remove`` removes the corresponding key::

    override:
      elastic:
        cli_flags:
          rb_force_constant:
            help: "$remove"

``$strategy``
~~~~~~~~~~~~~

``$strategy`` controls how a dictionary is applied. Supported values are
``merge`` and ``replace``.

``merge``
    Recursively merge the override into the existing dictionary.

``replace``
    Clear the existing dictionary before applying the override::

        override:
          elastic:
            cli_flags:
              rb_force_constant:
                $strategy: replace
                default: 500

The default strategy is ``merge``. A dictionary containing any of the
argument-source keys ``cli``, ``value``, or ``variable`` instead defaults to
``replace``. This prevents two different argument sources from being retained
when changing, for example, an argument from a CLI value to a fixed value.
``$strategy`` can be used to override this default explicitly.

Inserting a processor
~~~~~~~~~~~~~~~~~~~~~

If an override ID does not match an existing step, the entry is interpreted
as a new processor definition. It must contain ``processor`` and exactly one
of ``$insert_before`` or ``$insert_after``::

    override:
      sort_after_mapping:
        processor: vermouth.SortMoleculeAtoms
        $insert_after: vermouth.DoMapping
        args: {}

``processor``
    Fully qualified import path of the processor to insert.

``$insert_before``
    ID or effective processor ID before which the new processor is inserted.

``$insert_after``
    ID or effective processor ID after which the new processor is inserted.

The new override key becomes the explicit ``id`` of the inserted processor.
The insertion target must identify exactly one existing step.
