YAML pipeline configuration
===========================

Overview
--------

Martinize2 pipelines define which processors are run, their order, their
arguments, and the conditions under which they are executed.

Pipeline configurations can be written in YAML. The YAML files are read by
the ``PipelineConfigBuilder`` and converted into an internal pipeline
configuration. This configuration is then used to construct the command-line
interface and the executable pipeline.

A pipeline can be composed from multiple YAML files. This makes it possible
to separate general processing steps from force-field-specific or optional
steps.

Running Martinize2 with a pipeline
----------------------------------

Pipeline files are selected with the ``--pipeline`` option::

    martinize2 \
        --pipeline charmm water martini3001 \
        -inpath input.pdb \
        -outpath output.pdb

Pipeline names such as ``charmm`` and ``martini3001`` are resolved from the
default pipeline directory. The default pipeline configurations distributed
with Vermouth are located in ``vermouth/data/pipelines``. 
A path to a custom YAML file can also be used.

Multiple pipeline files
-----------------------

Multiple YAML files can be supplied after ``--pipeline``::

    martinize2 \
        --pipeline charmm water martini3001 \
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
       - ff

     cli_flags:
       elastic:
         action: store_true
         default: false
         help: Generate an elastic network.

     steps:
       - vermouth.pipeline_processors.ReadSystem:
           args:
             inpath:
               cli: inpath

       - vermouth.DoMapping:
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

   - mapping:
       steps:
         - vermouth.DoMapping:
             args:
               mappings:
                 variable: mappings
         - vermouth.DoAverageBead:
             args: {}

Processor steps
---------------

Each item in ``steps`` defines a processor and its configuration. Processors
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

The ``PipelineConfigBuilder`` reads these definitions into the internal
pipeline configuration. The ``CLIBuilder`` uses this configuration to
construct the Martinize2 command-line interface.

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
       --pipeline charmm water martini3001 \
       --override change.yaml \
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
``--pipeline-dir``.

A custom pipeline file can also be provided directly:

.. code-block:: console

   martinize2 \
       --pipeline path/to/custom_pipeline.yaml \
       -inpath input.pdb \
       -outpath output.pdb