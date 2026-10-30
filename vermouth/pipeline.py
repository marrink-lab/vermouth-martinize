from pathlib import Path
from copy import deepcopy
from collections import OrderedDict
from collections.abc import Collection, Mapping, MutableMapping, MutableSequence
import vermouth 
import argparse
import importlib
import jsonschema
from functools import lru_cache
import yaml
from vermouth.processors.processor import Pipeline
from vermouth.log_helpers import TypeAdapter, StyleAdapter
import logging
LOGGER = StyleAdapter(TypeAdapter(logging.getLogger("vermouth")))

DEFAULT_CLI_TYPE_MAP = {
    "str": str,
    "int": int,
    "float": float,
    "path": Path,
}


class _LiteralDollarString(str):
    """A string whose leading dollar sign was escaped in YAML."""


_STRING_LIKE = (str, bytes, bytearray)
SOURCE_KEY = "$source"


def _escape_literal_dollars(value):
    """
    Mark YAML strings whose leading dollar sign is escaped.

    A leading ``$$`` represents a literal leading ``$``. The marker preserves
    that distinction until composition directives have been processed.
    """
    if isinstance(value, Mapping):
        escaped = {}
        for key, item in value.items():
            escaped_key = _escape_literal_dollars(key)
            if escaped_key in escaped:
                raise ValueError(
                    f"Escaped mapping key {escaped_key!r} collides with an "
                    "existing key."
                )
            escaped[escaped_key] = _escape_literal_dollars(item)
        return escaped
    if isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        return value.__class__(_escape_literal_dollars(item) for item in value)
    if isinstance(value, str) and value.startswith("$$"):
        return _LiteralDollarString(value[1:])
    return value


def resolve_literal_dollars(value):
    """
    Convert escaped YAML dollar strings to ordinary strings recursively.

    This is called after composition directives have been evaluated, before
    the pipeline configuration is executed.
    """
    if isinstance(value, Mapping):
        return {
            resolve_literal_dollars(key): resolve_literal_dollars(item)
            for key, item in value.items()
        }
    if isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        return value.__class__(resolve_literal_dollars(item) for item in value)
    if isinstance(value, _LiteralDollarString):
        return str(value)
    return value


def _strip_source_metadata(value):
    """Remove internal composition provenance before pipeline execution."""
    if isinstance(value, MutableMapping):
        value.pop(SOURCE_KEY, None)
        for item in value.values():
            _strip_source_metadata(item)
    elif isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        for item in value:
            _strip_source_metadata(item)
    return value


def _is_directive_key(key, directive):
    """Return whether a mapping key is an unescaped composition directive."""
    return (
        isinstance(key, str)
        and not isinstance(key, _LiteralDollarString)
        and key == directive
    )


def _pop_directive(mapping, directive, default):
    """Remove and return an unescaped directive from a mapping."""
    for key in mapping:
        if _is_directive_key(key, directive):
            return mapping.pop(key)
    return default


def _has_escaped_key_collision(mapping, key):
    """Return whether equal mapping keys differ only by dollar escaping."""
    return any(
        existing_key == key
        and (
            isinstance(existing_key, _LiteralDollarString)
            != isinstance(key, _LiteralDollarString)
        )
        for existing_key in mapping
    )



# validate conditions  
def _options_used_in_condition(condition):
    """
    Collect CLI option and variable references used in a condition.

    Parameters
    ----------
    condition : dict
        Condition definition from the pipeline configuration.

    Returns
    -------
    tuple[set[str], set[str]]
        Referenced CLI option names and variable names.

    Raises
    ------
    ValueError
        If the condition type is unknown or an ``equal`` condition does not
        reference a CLI option or variable.
    """
    type_, cond = next(
        (key, value)
        for key, value in condition.items()
        if key != SOURCE_KEY
    )
    cli_refs = set()
    variable_refs = set()

    match type_:
        case 'all' | 'any':
            for item in cond:
                sub_cli_refs, sub_variable_refs = _options_used_in_condition(item)
                cli_refs |= sub_cli_refs
                variable_refs |= sub_variable_refs

        case 'not':
            cli_refs, variable_refs = _options_used_in_condition(cond)

        case 'equal':
            if 'cli' in cond:
                cli_refs.add(cond['cli'])
            if 'variable' in cond:
                variable_refs.add(cond['variable'])
        case _:
            raise ValueError(f"Unknown condition type: {type_}")

    return cli_refs, variable_refs

# validate if options are defines more than once
# are the parameters correct. 
def validate_cli_options(
    pipeline_conf,
    path='',
    local_cli_options=None,
    local_variables=None,
):
    """
    Validate CLI option and variable references in a pipeline configuration.

    The configuration is checked recursively to ensure that options and
    variables referenced by conditions and processor arguments have been
    defined in the current scope or an enclosing scope. Repeated CLI flag
    definitions must have identical options.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration to validate.
    path : str, optional
        Configuration path used in error messages.
    local_cli_options : Iterable[str], optional
        CLI options defined in an enclosing pipeline scope.
    local_variables : Iterable[str], optional
        Variables defined in an enclosing pipeline scope.

    Raises
    ------
    KeyError
        If an undefined CLI option or variable is referenced.
    ValueError
        If a condition definition is invalid.
    """
    if local_cli_options is None:
        _collect_cli_flags(pipeline_conf)
        local_cli_options = set()
    else:
        local_cli_options = set(local_cli_options)
    local_variables = set() if local_variables is None else set(local_variables)

    # Gather flags defined in this scope.
    cli_conf = pipeline_conf.get('cli', {})
    local_cli_options |= set(cli_conf.get('flags', {}).keys())
    for excl_group in cli_conf.get('exclusive_groups', []):
        local_cli_options |= set(excl_group.get('flags', {}))
    for group in cli_conf.get('groups', []):
        local_cli_options |= set(group.get('flags', {}))
        for excl_group in group.get('exclusive_groups', []):
            local_cli_options |= set(excl_group.get('flags', {}))
    local_cli_options.discard(SOURCE_KEY)

    # force_field variable options
    variable_options = set(pipeline_conf.get("variables", []))

    # add to the variables defined in this scope and its enclosing scopes
    local_variables |= variable_options

    # check for options used in conditions
    if 'condition' in pipeline_conf:
        cond_cli_refs, cond_variable_refs = _options_used_in_condition(
            pipeline_conf['condition']
        )

        if missing := (cond_cli_refs - local_cli_options):
            _path = '.'.join([path, "condition"])
            raise KeyError(
                f"CLI option(s) {missing} in {_path} have not been defined. "
                f"Known CLI options are {local_cli_options}."
            )
        if missing := (cond_variable_refs - local_variables):
            _path = '.'.join([path, "condition"])
            raise KeyError(
                f"Variable(s) {missing} in {_path} have not been defined. "
                f"Known variables are {local_variables}."
            )
    # check for options used in arguments if this is not a pipeline step
    is_pipeline = bool(pipeline_conf.get('steps'))

    if not is_pipeline:
        cli_references = set()
        variable_references = set()

        for value in pipeline_conf.get('args', {}).values():
            if value == SOURCE_KEY: continue
            if 'cli' in value:
                cli_references.add(value['cli'])

            if 'variable' in value:
                variable_references.add(value['variable'])

        if missing := (cli_references - local_cli_options):
            _path = '.'.join([path, "args"])
            raise KeyError(
                f"CLI option(s) {missing} in {_path} have not been defined. "
                f"Known CLI options are {local_cli_options}."
            )

        if missing := (variable_references - local_variables):
            _path = '.'.join([path, "args"])
            raise KeyError(
                f"Variable(s) {missing} in {_path} have not been defined. "
                f"Known variables are {local_variables}."
            )
    else:
        for idx, (name, step) in enumerate(pipeline_conf["steps"].items()):
            _path = '.'.join([path, f'steps[{idx}]', name])
            validate_cli_options(
                step,
                _path,
                local_cli_options,
                local_variables,   
            )

def find_pipeline_yaml(name, pipeline_dirs):
    """
    Locate a pipeline YAML file.

    The function first checks whether ``name`` is an existing path, then
    searches the user-provided pipeline directories, and finally searches the
    default Vermouth pipeline directory.

    Parameters
    ----------
    name : str or pathlib.Path
        Pipeline name or path.
    pipeline_dirs : Iterable[pathlib.Path]
        Additional directories to search.

    Returns
    -------
    pathlib.Path
        Path to the pipeline YAML file.

    Raises
    ------
    FileNotFoundError
        If the pipeline YAML file cannot be found.
    """
    force_field_directory = getattr(name, "directory", None)
    if force_field_directory is not None:
        candidate = Path(force_field_directory) / "pipeline.yaml"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"Force field {name.name!r} does not provide a pipeline YAML "
            f"at '{candidate}'."
        )

    path = Path(name)

    # User specified path
    if path.exists():
        return path

    # search in the user-specified directories
    for directory in pipeline_dirs:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.exists():
            return candidate

    # Standard pipeline location
    candidate = vermouth.DATA_PATH / "pipelines" / f"{name}.yaml"
    if candidate.exists():
        return candidate

    # A force field provides its pipeline alongside its definitions.
    candidate = vermouth.DATA_PATH / "force_fields" / str(name) / "pipeline.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find pipeline YAML '{name}'.")

def translate_cli_opts(opts, type_map=None):
    """Translate string-defined CLI types with the supplied type mapping."""
    opts = dict(opts)
    if "type" in opts and isinstance(opts["type"], str):
        try:
            opts["type"] = (type_map or DEFAULT_CLI_TYPE_MAP)[opts["type"]]
        except KeyError as error:
            raise ValueError(f"Unknown CLI type: {opts['type']}") from error
    return opts


def add_cli_flag(base_group, flag, opts, prefix="-", type_map=None):
    opts = dict(opts)
    cli_name = opts.pop("cli", flag)
    aliases = opts.pop("aliases", [])
    opts = translate_cli_opts(opts, type_map)
    base_group.add_argument(
        f"{prefix}{cli_name}",
        *aliases,
        dest=flag,
        **opts,
    )

# build the CLI based on the pipeline configuration.
def build_cli(
    name,
    pipeline_conf,
    prefix,
    parser=None,
    added_flags=None,
    type_map=None,
    **kwargs,
):
    """
    Build a command-line parser from a pipeline configuration.

    CLI flags and mutually exclusive groups are added recursively from the
    pipeline configuration. Flags that have already been added are skipped.

    Parameters
    ----------
    name : str
    pipeline_conf : dict
        Pipeline configuration containing CLI definitions.
    prefix : str
        Prefix used for command-line options.
    parser : argparse.ArgumentParser, optional
        Existing parser to extend. A new parser is created when omitted.
    added_flags : set[str], optional
        CLI flags that have already been added.
    **kwargs
        Additional arguments passed to ``argparse.ArgumentParser``.

    Returns
    -------
    argparse.ArgumentParser
        Parser containing the configured CLI options.
    """
    # make parser if not given, otherwise use the given one.
    parser = parser or argparse.ArgumentParser(allow_abbrev=False, **kwargs)
    # make an empty set of the added_flags. or use the given one.
    added_flags = set() if added_flags is None else added_flags
    # loop through the cli flags defined in the pipeline config. and don't add the same flag twice.
    cli_conf = pipeline_conf.get('cli', {})
    for flag, opts in cli_conf.get('flags', {}).items():
        if flag == SOURCE_KEY:
            continue
        if flag in added_flags:
            continue
        # make a options dict from the options defined in the yaml. and translate the type from a string to a real python type.
        add_cli_flag(parser, flag, opts, prefix, type_map)
        added_flags.add(flag)

    for excl_group in cli_conf.get('exclusive_groups', []):
        group = parser.add_mutually_exclusive_group(**{k: v for k, v in excl_group.items() if k != 'flags'})
        for flag, opts in excl_group.get('flags', {}).items():
            if flag == SOURCE_KEY:
                continue
            if flag in added_flags:
                continue
            add_cli_flag(group, flag, opts, prefix, type_map)
            added_flags.add(flag)

    for grp in cli_conf.get('groups', []):
        group_flags = {
            flag
            for flag in grp.get('flags', {})
            if flag != SOURCE_KEY
        }
        for excl_group in grp.get('exclusive_groups', []):
            group_flags.update(
                flag for flag in excl_group.get('flags', {})
                if flag != SOURCE_KEY
            )
        if group_flags and group_flags <= added_flags:
            continue
        group = parser.add_argument_group(**{k: v for k, v in grp.items() if k not in ('flags', 'exclusive_groups')})
        for flag, opts in grp.get('flags', {}).items():
            if flag == SOURCE_KEY:
                continue
            if flag in added_flags:
                continue
            add_cli_flag(group, flag, opts, prefix, type_map)
            added_flags.add(flag)
        for excl_group in grp.get('exclusive_groups', []):
            exclusive_group = group.add_mutually_exclusive_group(**{k: v for k, v in excl_group.items() if k != 'flags'})
            for flag, opts in excl_group.get('flags', {}).items():
                if flag == SOURCE_KEY:
                    continue
                if flag in added_flags:
                    continue
                add_cli_flag(exclusive_group, flag, opts, prefix, type_map)
                added_flags.add(flag)

    # recursion for steps in the pipeline
    if pipeline_conf.get('steps'):
        for name, step in pipeline_conf["steps"].items():
            build_cli(
                name,
                step,
                prefix,
                parser=parser,
                added_flags=added_flags,
                type_map=type_map,
            )

    return parser

# evaluate the condition with the cli values
def eval_condition(condition, cli_args, variables):
    """
    Evaluate a pipeline condition.

    Supported conditions are ``any``, ``all``, ``not``, ``equal``, and
    ``has_variable``.

    Parameters
    ----------
    condition : dict
        Condition definition to evaluate.
    cli_args : dict
        Parsed command-line argument values.
    variables : dict
        Runtime variables available to the pipeline.

    Returns
    -------
    bool
        Result of the condition.

    Raises
    ------
    ValueError
        If the condition has an invalid or unknown condition type.
    """
    condition_entries = [
        (key, value)
        for key, value in condition.items()
        if key != SOURCE_KEY
    ]
    if len(condition_entries) != 1:
        raise ValueError(
            f"Condition must contain exactly one condition type, got {condition}."
        )
    
    # get the type and arguments of the condition
    type_, args = condition_entries[0]
    # what type of condition is it and what to do with it 
    match type_:
        case 'any':
            verdict = any(eval_condition(c, cli_args, variables) for c in args)
        case 'all':
            verdict = all(eval_condition(c, cli_args, variables) for c in args)
        case 'not':
            verdict = not eval_condition(args, cli_args, variables)
        case 'equal':
            items = []
            if 'cli' in args:
                items.append(cli_args[args['cli']])
            if 'variable' in args:
                items.append(variables[args['variable']])
            if 'value' in args:
                items.append(args['value'])
            if len(items) != 2:
                raise ValueError("equal condition needs exactly 2 arguments.")
            verdict = items[0] == items[1]
        case _:
            raise ValueError(f"Unknown condition type: {type_}")

    return verdict

def resolve_config_paths(value):
    """Resolve declarative YAML path values recursively."""
    if isinstance(value, Mapping):
        path_directives = [
            directive
            for directive in ("$path", "$data")
            if any(
                _is_directive_key(key, directive)
                for key in value
            )
        ]
        if path_directives:
            if len(path_directives) != 1:
                raise ValueError(
                    "A path value must contain exactly one path directive."
                )
            directive = path_directives[0]
            reference = Path(value[directive])
            if directive == "$path":
                source = value.get(SOURCE_KEY)
                if source is None:
                    raise ValueError("$path requires YAML source metadata.")
                return (Path(str(source).split(":", 1)[0]).parent / reference).resolve()

            data_path = Path(vermouth.DATA_PATH).resolve()
            resolved = (data_path / reference).resolve()
            if reference.is_absolute() or not resolved.is_relative_to(data_path):
                raise ValueError("$data paths must remain under Vermouth's data directory.")
            return resolved

        return {
            key: resolve_config_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        return value.__class__(resolve_config_paths(item) for item in value)
    return value


# set the values from the CLI into the pipeline config
def set_values(pipeline_conf, cli_args, variables):
    """
    Resolve values and conditions in a pipeline configuration.

    Processor arguments are resolved from fixed values, command-line options,
    or runtime variables. Conditions are evaluated and processor classes are
    imported recursively.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration to resolve.
    cli_args : dict
        Parsed command-line arguments.
    variables : dict
        Runtime variables available to the pipeline.

    Raises
    ------
    KeyError
        If a processor argument does not specify a value source.
    """
    # check if there is a condition 
    if 'condition' in pipeline_conf:
        pipeline_conf['condition'] = eval_condition(pipeline_conf['condition'], cli_args, variables)
    else:
        pipeline_conf['condition'] = True
    # check if the current processor has argumnents 
    if 'args' in pipeline_conf:
        # make the args dict with the real values from the CLI
        args = {}
        # loop through the arguments defined in the pipeline config
        for arg_name, value in pipeline_conf['args'].items():
            if arg_name == SOURCE_KEY:
                continue
            if "value" in value:
                # check if its a fixed value
                args[arg_name] = resolve_config_paths(value['value'])
            elif "cli" in value:
                # if not, use the CLI value 
                args[arg_name] = cli_args[value['cli']]
                # set the arg value good 
            elif 'variable' in value:
                args[arg_name] = variables[value['variable']]
            else: 
                raise KeyError(f"{arg_name} must have a value, cli, or variable")
        pipeline_conf['args'] = args
    # if its recursive pipeline, do the same for the steps in the pipeline
    for name, step in pipeline_conf.get("steps", {}).items():
        if not step.get('steps'):
            # go from text to actual processor object
            processor_path = step.get('processor', name)
            step['processor'] = import_processor(processor_path)
        # call itself 
        set_values(step, cli_args, variables)

# import the processor 
def import_processor(processor_name):
    """
    Import a processor from its fully qualified name.

    Parameters
    ----------
    processor_name : str
        Processor name in ``module.ClassName`` format.

    Returns
    -------
    type
        Imported processor class.
    """
    # split the processor name into module and name. 
    module, name = processor_name.rsplit('.', 1)
    # import to python module 
    module = importlib.import_module(module)
    # get the processor class from the module 
    proc = getattr(module, name)
    return proc

def rename_variables(obj, renames):
    """
    Rename variables declared and referenced in a configuration object.

    The configuration is traversed recursively. Values associated with a
    ``variable`` key and names declared in a ``variables`` list are rewritten
    according to ``renames``.

    Parameters
    ----------
    obj : object
        Configuration object to process.
    renames : Mapping[str, str]
        Mapping from names used by the included fragment to names in the
        including scope.

    Returns
    -------
    object
        The configuration object with namespaced variable references.
    """
    available_names = _variable_names(obj)
    unknown_names = set(renames) - available_names
    if unknown_names:
        raise KeyError(
            f"Cannot rename undefined variable(s) {sorted(unknown_names)!r}."
        )

    return _rename_variables(obj, renames)


def _variable_names(value):
    """Collect variable declarations and references from a configuration tree."""
    names = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "variable" and isinstance(item, str):
                names.add(item)
            elif key == "variables" and isinstance(item, Collection):
                names.update(
                    variable for variable in item if isinstance(variable, str)
                )
            else:
                names.update(_variable_names(item))
    elif isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        for item in value:
            names.update(_variable_names(item))
    return names


def _rename_variables(obj, renames):
    """Apply validated variable renames recursively."""
    if isinstance(obj, MutableMapping):
        for key, value in obj.items():
            if key == "variable" and isinstance(value, str):
                obj[key] = renames.get(value, value)
            elif key == "variables" and isinstance(value, MutableSequence):
                obj[key] = [renames.get(variable, variable) for variable in value]
            else:
                _rename_variables(value, renames)
    elif isinstance(obj, Collection) and not isinstance(obj, _STRING_LIKE):
        for item in obj:
            _rename_variables(item, renames)
    return obj


def _validate_raw_step_names(pipeline_conf, path="martinize2", source=None):
    """
    Validate that step keys are unique within each pipeline.

    Step keys identify a step independently of its processor import path.
    The same processor may therefore be used by multiple uniquely named
    steps.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration to validate.
    path : str, optional
        Structural path used in error messages.
    source : pathlib.Path or str, optional
        YAML file from which the steps were loaded.

    Raises
    ------
    ValueError
        If a pipeline contains the same step key more than once.
    """
    seen_names = {}
    for index, (name, step) in enumerate(pipeline_conf.get("steps", [])):
        if step == '$remove': continue
        step_path = f"{path}.steps[{index}].{name}"
        if name in seen_names:
            provenance = f" in {source}" if source is not None else ""
            raise ValueError(
                f"Duplicate step key {name!r}: {seen_names[name]} and "
                f"{step_path}{provenance}."
            )
        seen_names[name] = step_path

        if step.get("steps"):
            _validate_raw_step_names(step, step_path, source)


def validate_step_names(pipeline_conf, path="martinize2"):
    """Validate the structure of post-conversion ordered step mappings."""
    for index, (name, step) in enumerate(
        pipeline_conf.get("steps", OrderedDict()).items()
    ):
        step_path = f"{path}.steps[{index}].{name}"
        if step.get("steps"):
            validate_step_names(step, step_path)


@lru_cache(maxsize=32)
def load_yaml_file(path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the YAML file.

    Returns
    -------
    object
        Parsed contents of the YAML file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return _escape_literal_dollars(yaml.safe_load(file))


def find_pipeline_configs(pipeline_names, pipeline_dirs):
    for name in pipeline_names:
        path = find_pipeline_yaml(name, pipeline_dirs)
        yield path


def _schema_instance(value):
    """
    Convert ordered YAML mapping tuples to JSON Schema-compatible arrays.

    PyYAML represents ``!!omap`` entries as tuples, while the JSON Schema
    array type accepts lists only. Runtime configurations instead represent
    ``steps`` as ``OrderedDict`` objects. Convert both representations to the
    raw ordered-map array expected by the schema without changing the
    configuration being validated.
    """
    if isinstance(value, Mapping):
        return {
            key: (
                [
                    [_schema_instance(step_name), _schema_instance(step)]
                    for step_name, step in item.items()
                ]
                if key == "steps" and isinstance(item, OrderedDict)
                else _schema_instance(item)
            )
            for key, item in value.items()
            if key != SOURCE_KEY
        }
    if isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        return [_schema_instance(item) for item in value]
    return value


def _annotate_sources(value, source, structural_path=""):
    """Attach internal YAML provenance to every mapping in a configuration."""
    if not isinstance(value, MutableMapping):
        return value

    value[SOURCE_KEY] = (
        str(source)
        if not structural_path
        else f"{source}:{structural_path}"
    )
    for key, child in value.items():
        if key == SOURCE_KEY:
            continue
        child_path = (
            str(key)
            if not structural_path
            else f"{structural_path}.{key}"
        )
        if key == "steps" and isinstance(child, MutableSequence):
            step_items = child
        elif key == "steps" and isinstance(child, Mapping):
            step_items = child.items()
        else:
            step_items = None
        if step_items is not None:
            for name, step in step_items:
                if name == SOURCE_KEY:
                    continue
                _annotate_sources(
                    step,
                    source,
                    f"{child_path}.{name}",
                )
        elif isinstance(child, MutableMapping):
            _annotate_sources(child, source, child_path)
        elif isinstance(child, Collection) and not isinstance(child, _STRING_LIKE):
            for item in child:
                _annotate_sources(item, source, child_path)
    return value


def _parse_include_entry(include):
    """Return an include reference and its fragment-local variable renames."""
    if isinstance(include, Mapping):
        renames = {
            name: replacement
            for name, replacement in include.get("rename_variables", {}).items()
            if name != SOURCE_KEY
        }
        return include["path"], renames
    return include, {}


def _parse_include_reference(reference, including_path, pipeline_dirs):
    """
    Resolve an include reference to a file path and optional fragment path.

    The schema validates that ``reference`` is a string. A fragment path
    follows the first colon and uses dot-separated mapping keys with optional
    ``[index]`` or ``[ordered-map-key]`` selectors.
    """
    filename, separator, fragment_path = reference.partition(":")
    include_path = Path(filename)
    relative_path = Path(including_path).parent / include_path
    if not include_path.is_absolute() and relative_path.exists():
        resolved_path = relative_path
    else:
        resolved_path = find_pipeline_yaml(
            filename,
            [Path(including_path).parent, *pipeline_dirs],
        )

    return resolved_path, fragment_path if separator else None


def select_include_fragment(config, fragment_path):
    """
    Select a mapping or ordered-map fragment from a loaded YAML document.

    Ordered mappings support numeric indexes in brackets and step keys as
    dotted path components. For example, ``martinize2.steps[0].args`` and
    ``martinize2.steps.read_input.args`` select the same argument mapping.
    """
    value = config
    if not fragment_path:
        return value

    for component in fragment_path.split("."):
        key, separator, selector = component.partition("[")
        if key:
            if isinstance(value, Mapping):
                value = value[key]
            else:
                for entry_key, entry_value in value:
                    if entry_key == key:
                        value = entry_value
                        break
                else:
                    raise KeyError(
                        f"Ordered-map key {key!r} was not found in "
                        f"{component!r}."
                    )
        if separator:
            if not selector.endswith("]"):
                raise KeyError(
                    f"Invalid include fragment selector {component!r}."
                )
            selector = selector[:-1]
            if not selector.isdecimal():
                raise KeyError(
                    f"Ordered-map selector {selector!r} must be a numeric "
                    f"index in {component!r}."
                )
            index = int(selector)
            if isinstance(value, Mapping):
                value = list(value.values())[index]
            else:
                value = value[index]
            if (
                isinstance(value, Collection)
                and not isinstance(value, _STRING_LIKE)
                and len(value) == 2
                and isinstance(value[1], Mapping)
            ):
                value = value[1]

    return value


def compose_pipeline_file(path, pipeline_dirs=(), inclusion_chain=()):
    """Load and compose a pipeline file, expanding includes depth-first."""
    path = find_pipeline_yaml(path, pipeline_dirs).resolve()
    config = deepcopy(load_yaml_file(path))
    return compose_pipeline_config(
        config,
        path,
        pipeline_dirs,
        inclusion_chain,
    )


def compose_pipeline_config(config, path, pipeline_dirs=(), inclusion_chain=()):
    """Compose an in-memory pipeline document with ordinary include semantics."""
    path = Path(path).resolve()
    if path in inclusion_chain:
        chain = " -> ".join(map(str, (*inclusion_chain, path)))
        raise ValueError(f"Include cycle detected: {chain}")

    _annotate_sources(config, path)
    root = config["martinize2"]
    if isinstance(root.get("steps"), MutableSequence):
        _validate_raw_step_names(root, source=path)
        _convert_step_mappings(root)
    _compose_includes(
        config,
        path,
        pipeline_dirs,
        (*inclusion_chain, path),
        "",
    )
    _validate_pipeline_config(config, path)
    return config


def _compose_includes(
    value,
    path,
    pipeline_dirs,
    inclusion_chain,
    structural_path,
):
    """Expand include directives in a mapping, with local keys taking priority."""
    if not isinstance(value, MutableMapping):
        # FIXME: Can't include in lists? I guess steps are ordereddicts.
        return value

    # Remove the to be consumed include keys
    includes = _pop_directive(value, INCLUDE_KEY, [])
    if not includes:
        # Nothing to merge in, so rebuilding would only evaluate positional
        # directives ($insert_after, ...) against an empty target. Leave them
        # for the parent merge, where the included siblings exist.
        for key, child in value.items():
            child_path = str(key) if not structural_path else f"{structural_path}.{key}"
            if isinstance(child, MutableMapping):
                _compose_includes(
                    child, path, pipeline_dirs, inclusion_chain, child_path
                )
            elif isinstance(child, Collection) and not isinstance(child, _STRING_LIKE):
                for item in child:
                    if isinstance(item, MutableMapping):
                        _compose_includes(
                            item, path, pipeline_dirs, inclusion_chain, child_path
                        )
        return value
    local = deepcopy(value)
    # Clear the existing pipeline. This way we can *first* merge in the included
    # elements, and then merge in the existing elements, which we helpfully
    # copied over to `local`. This way local elements override included elements
    # The clear /is/ required: leaving `value` full and merging in its own
    # elements will cause all sorts of mayhem for the steps lists.
    value.clear()
    if SOURCE_KEY in local:
        value[SOURCE_KEY] = local[SOURCE_KEY]

    for reference in includes:
        reference, renames = _parse_include_entry(reference)
        include_path, fragment_path = _parse_include_reference(
            reference,
            path,
            pipeline_dirs,
        )
        included = compose_pipeline_file(
            include_path,
            [Path(include_path).parent, *pipeline_dirs],
            inclusion_chain,
        )
        fragment = deepcopy(select_include_fragment(included, fragment_path))
        rename_variables(fragment, renames)
        if not isinstance(fragment, Mapping):
            # FIXME: Stupid limitation
            raise TypeError(
                f"Included fragment {reference!r} must resolve to a mapping."
            )
        source = (
            str(include_path)
            if fragment_path is None
            else f"{include_path}:{fragment_path}"
        )
        merge_pipeline_mapping(
            value,
            fragment,
            source=source,
        )

    for key, child in local.items():
        child_path = (
            str(key)
            if not structural_path
            else f"{structural_path}.{key}"
        )
        if isinstance(child, MutableMapping):
            _compose_includes(
                child,
                path,
                pipeline_dirs,
                inclusion_chain,
                child_path,
            )
        elif isinstance(child, Collection) and not isinstance(child, _STRING_LIKE):
            for item in child:
                if isinstance(item, MutableMapping):
                    _compose_includes(
                        item,
                        path,
                        pipeline_dirs,
                        inclusion_chain,
                        child_path,
                    )
        merge_pipeline_mapping(
            value,
            {key: child},
            source=f"{path}:{structural_path}",
        )

    return value


def _validate_pipeline_config(config, path):
    """
    Validate a raw or composed pipeline configuration.

    Raw YAML ordered maps are duplicate-checked before being converted to
    ``OrderedDict`` objects. Already-converted composed configurations are
    schema-checked through ``_schema_instance`` and must not be passed to the
    raw ordered-map validator again.
    """
    schema_uri = config.get('$schema')
    if schema_uri:
        yaml_dir = Path(path).parent
        schema_path = Path(schema_uri)
        if not schema_path.is_absolute():
            schema_path = yaml_dir / schema_path
        schema = load_yaml_file(schema_path)
        jsonschema.validate(_schema_instance(config), schema)

    root = config.get("martinize2")
    steps = root.get("steps")
    if isinstance(steps, MutableSequence):
        _validate_raw_step_names(root, source=path)
        _convert_step_mappings(root)
    validate_step_names(root)


def _convert_step_mappings(value):
    """Convert validated YAML ordered-map step pairs to OrderedDict objects."""
    if isinstance(value, MutableMapping):
        for key, child in tuple(value.items()):
            if key == "steps" and isinstance(child, MutableSequence):
                steps = OrderedDict()
                for step_name, step_config in child:
                    _convert_step_mappings(step_config)
                    steps[step_name] = step_config
                value[key] = steps
            else:
                _convert_step_mappings(child)
    elif isinstance(value, Collection) and not isinstance(value, _STRING_LIKE):
        for child in value:
            _convert_step_mappings(child)


def load_pipeline_configs(pipeline_paths, pipeline_dirs=()):
    """
    Load multiple pipeline YAML configs.

    Parameters
    ----------
    pipeline_paths : Iterable[pathlib.Path]
        Paths to YAML pipeline fragments.
    pipeline_dirs : list[pathlib.Path]
        Extra directories to search in.

    Returns
    -------
    list[tuple[str, dict]]
        List of (namespace, config) pairs.
    """
    configs = []

    for pipeline_path in pipeline_paths:
        path = find_pipeline_yaml(pipeline_path, pipeline_dirs)
        conf = compose_pipeline_file(path, pipeline_dirs)
        namespace = Path(path).stem
        configs.append((namespace, conf))

    return configs


def iter_cli_flags(pipeline_conf):
    """
    Iterate over all CLI flags in a pipeline configuration.

    CLI flags from normal flag definitions, mutually exclusive groups, and
    nested pipeline steps are yielded recursively.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration to inspect.

    Yields
    ------
    tuple[str, dict]
        CLI flag name and its configuration.
    """
    # gather cli_flags defined in cli_flags
    cli_conf = pipeline_conf.get('cli', {})
    yield from (
        (flag, options)
        for flag, options in cli_conf.get("flags", {}).items()
        if flag != SOURCE_KEY
    )
    for excl_group in cli_conf.get('exclusive_groups', []):
        yield from (
            (flag, options)
            for flag, options in excl_group.get('flags', {}).items()
            if flag != SOURCE_KEY
        )
    for group_conf in cli_conf.get('groups', []):
        for excl_group in group_conf.get('exclusive_groups', []):
            yield from (
                (flag, options)
                for flag, options in excl_group.get('flags', {}).items()
                if flag != SOURCE_KEY
            )
        yield from (
            (flag, options)
            for flag, options in group_conf.get('flags', {}).items()
            if flag != SOURCE_KEY
        )

    # recursion for steps in the pipeline
    if pipeline_conf.get("steps"):
        for name, step in pipeline_conf["steps"].items():
            yield from iter_cli_flags(step)


def _collect_cli_flags(pipeline_conf):
    """Collect CLI flags and check that repeated definitions are consistent."""
    flags = {}
    for flag, options in iter_cli_flags(pipeline_conf):
        normalized_options = _strip_source_metadata(deepcopy(options))
        if flag in flags and flags[flag] != normalized_options:
            raise ValueError(
                f"CLI flag {flag!r} is defined multiple times with "
                "different options."
            )
        flags[flag] = normalized_options
    return flags


REMOVE_VALUE = "$remove"
STRATEGY_KEY = "$strategy"
INCLUDE_KEY = "$include"
VALID_STRATEGIES = {"merge", "replace"}


def _insertion_anchors(value):
    """Remove and return insertion directives from a mapping value."""
    if not isinstance(value, MutableMapping):
        return None, None
    return (
        _pop_directive(value, "$insert_before", None),
        _pop_directive(value, "$insert_after", None),
    )


def _validate_existing_insertion_position(
    target,
    key,
    insert_before,
    insert_after,
    source,
):
    """Ensure an existing sibling satisfies its requested local anchors."""
    if insert_before is None and insert_after is None:
        return

    names = list(target)
    anchors = [
        anchor
        for anchor in (insert_before, insert_after)
        if anchor is not None
    ]
    missing = [anchor for anchor in anchors if anchor not in target]
    if missing:
        raise KeyError(
            f"Cannot position {key!r} from {source or 'an unknown source'}: "
            f"local anchor {missing[0]!r} was not found."
        )

    key_index = names.index(key)
    if (
        insert_after is not None
        and key_index != names.index(insert_after) + 1
    ):
        raise ValueError(
            f"Cannot position {key!r} from {source or 'an unknown source'}: "
            f"it is not immediately after local anchor {insert_after!r}."
        )
    if (
        insert_before is not None
        and key_index != names.index(insert_before) - 1
    ):
        raise ValueError(
            f"Cannot position {key!r} from {source or 'an unknown source'}: "
            f"it is not immediately before local anchor {insert_before!r}."
        )


def merge_pipeline_mapping(target, incoming, source=None):
    """
    Compose an incoming pipeline mapping into a target mapping.

    Existing mapping keys are merged recursively and scalar values are
    replaced. New keys are appended unless their mapping value declares
    ``$insert_before`` or ``$insert_after``. ``source`` identifies the YAML
    fragment that contributed ``incoming`` in composition errors.
    """
    if not isinstance(target, MutableMapping):
        raise TypeError(
            f"Composition target must be a mapping, not "
            f"{type(target).__name__}."
        )
    if not isinstance(incoming, Mapping):
        raise TypeError(
            f"Composed value must be a mapping, not "
            f"{type(incoming).__name__}."
        )

    strategy = next(
        (
            value
            for key, value in incoming.items()
            if _is_directive_key(key, STRATEGY_KEY)
        ),
        "merge",
    )
    if strategy == "replace":
        target.clear()
    elif strategy != "merge":
        raise ValueError(
            f"Unknown composition strategy {strategy!r}; expected 'merge' "
            "or 'replace'."
        )

    for key, value in incoming.items():
        if key == SOURCE_KEY:
            continue
        if _is_directive_key(key, STRATEGY_KEY):
            continue
        if _has_escaped_key_collision(target, key):
            raise ValueError(
                f"Escaped mapping key {key!r} collides with an existing key."
            )
        if (
            isinstance(value, str)
            and not isinstance(value, _LiteralDollarString)
            and value == REMOVE_VALUE
        ):
            target.pop(key, None)
            continue
        new_value = deepcopy(value)
        insert_before, insert_after = _insertion_anchors(new_value)
        if key in target and isinstance(target[key], MutableMapping) and isinstance(new_value, Mapping):
            merge_pipeline_mapping(target[key], new_value, source)
            _validate_existing_insertion_position(
                target,
                key,
                insert_before,
                insert_after,
                source,
            )
        elif key in target:
            target[key] = new_value
            _validate_existing_insertion_position(
                target,
                key,
                insert_before,
                insert_after,
                source,
            )
        else:
            if insert_before is None and insert_after is None:
                target[key] = new_value
            else:
                anchors = [
                    anchor
                    for anchor in (insert_before, insert_after)
                    if anchor is not None
                ]
                missing = [anchor for anchor in anchors if anchor not in target]
                if missing:
                    raise KeyError(
                        f"Cannot insert {key!r} from {source or 'an unknown source'}: "
                        f"local anchor {missing[0]!r} was not found."
                    )
                items = list(target.items())
                names = list(target)
                if insert_before is not None and insert_after is not None:
                    before_index = names.index(insert_before)
                    after_index = names.index(insert_after)
                    if before_index != after_index + 1:
                        raise ValueError(
                            f"Cannot insert {key!r} from "
                            f"{source or 'an unknown source'}: "
                            f"{insert_after!r} and {insert_before!r} do not "
                            "define one local insertion slot."
                        )
                    index = before_index
                elif insert_before is not None:
                    index = names.index(insert_before)
                else:
                    index = names.index(insert_after) + 1
                items.insert(index, (key, new_value))
                target.clear()
                target.update(items)


class CLIBuilder:
    """
    Build a command-line interface from a pipeline configuration.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration containing the CLI definitions.
    prefix : str, optional
        Prefix used for generated command-line options.
    """
    def __init__(self, name, pipeline_conf, prefix="-", type_map=None):
        self.name = name
        self.pipeline_conf = _strip_source_metadata(deepcopy(pipeline_conf))
        self.prefix = prefix
        self.type_map = type_map
        self._argparser = None

    @property
    def argparser(self):
        """
        Return the command-line argument parser.

        The parser is built when it is accessed for the first time.

        Returns
        -------
        argparse.ArgumentParser
            Generated command-line parser.
        """
        if self._argparser is None:
            self.build_argparser()
        return self._argparser

    def build_argparser(self, **kwargs):
        """
        Build and store the command-line argument parser.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to ``build_cli``.
        """
        self._argparser = build_cli(
            self.name,
            self.pipeline_conf,
            self.prefix,
            type_map=self.type_map,
            **kwargs,
        )

    def parse_cli_args(self, args=None):
        """
        Parse command-line arguments.

        Parameters
        ----------
        args : Sequence[str], optional
            Arguments to parse. If omitted, arguments are read from
            ``sys.argv``.

        Returns
        -------
        dict
            Parsed command-line arguments.
        """
        return vars(self.argparser.parse_args(args))

class PipelineBuilder:
    """
    Build an executable pipeline from a pipeline configuration.

    Parameters
    ----------
    pipeline_conf : dict
        Pipeline configuration used to construct the pipeline.
    """
    def __init__(self, pipeline_conf, name="pipeline"):
        self.pipeline_conf = pipeline_conf
        self.name = name

    def build_pipeline(self, cli_args, variables):
        """
        Resolve configuration values and build an executable pipeline.

        Parameters
        ----------
        cli_args : dict
            Parsed command-line arguments.
        variables : dict
            Runtime variables available to the pipeline.

        Returns
        -------
        Pipeline
            Executable Vermouth pipeline.
        """
        pipeline_conf = resolve_literal_dollars(deepcopy(self.pipeline_conf))
        _convert_step_mappings(pipeline_conf)
        set_values(pipeline_conf, cli_args, variables)
        _strip_source_metadata(pipeline_conf)

        return Pipeline.from_dict(
            pipeline_conf,
            self.name,
        )
    
