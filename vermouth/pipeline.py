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
    type_, cond = next(iter(condition.items()))
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
            elif 'variable' in cond:
                variable_refs.add(cond['variable'])
            else:
                raise ValueError(
                    "equal condition needs 'cli' or 'variable'"
                )
            
        case 'has_variable':
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
    defined.

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
    local_cli_options = set() if local_cli_options is None else set(local_cli_options)
    local_variables = set() if local_variables is None else set(local_variables)

    # gather flags defined in cli_flags
    cli_conf = pipeline_conf.get('cli', {})
    local_cli_options |= set(cli_conf.get('flags', {}).keys())
    for excl_group in cli_conf.get('exclusive_groups', []):
        local_cli_options |= set(excl_group.get('flags', {}).keys())
    for name, group in cli_conf.get('groups', {}).items():
        local_cli_options |= set(group.get('flags', {}).keys())
        for excl_group in group.get('exclusive_groups', []):
            local_cli_options |= set(excl_group.get('flags', {}).keys())

    # force_field variable options
    variable_options = set(pipeline_conf.get("variables", []))

    # add to the sets of options defined in this scope and globally
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

def _cys_argument(value):
    """
    Parse a cysteine bridge command-line argument.

    Parameters
    ----------
    value : str
        Value supplied on the command line. Accepted values are ``auto``,
        ``none``, or a floating-point number.

    Returns
    -------
    str or float
        ``auto``, ``none``, or the parsed floating-point value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be parsed.
    """
    try:
        return float(value)
    except ValueError:
        match value.lower():
            case "auto" | "none" as v:
                return v
            case _:
                raise argparse.ArgumentTypeError(
                    'Value must be "auto", "none", or a float.'
                )
def water_bias(value):
    """
    Parse a water-bias command-line argument.

    Parameters
    ----------
    value : str
        A letter and epsilon value separated by a colon.

    Returns
    -------
    tuple[str, float]
        The letter and corresponding epsilon value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value does not have the expected format.
    """
    try:
        letter, epsilon = value.split(":")
        return letter, float(epsilon)
    except Exception:
        raise argparse.ArgumentTypeError(
                'value must be a letter and a float separated by a colon'
    )
def ignore_resname(value):
    """
    Parse a comma-separated list of residue names.

    Parameters
    ----------
    value : str
        Comma-separated residue names.

    Returns
    -------
    list[str]
        Residue names with whitespace removed.
    """
    return [item.strip() for item in value.split(",") if item.strip()]

def translate_cli_opts(opts):
    """
    Translate YAML CLI options to values accepted by argparse.

    String type definitions are replaced by their corresponding Python
    callables using ``TYPE_MAP``.

    Parameters
    ----------
    opts : dict
        CLI option configuration.

    Returns
    -------
    dict
        Translated CLI option configuration.

    Raises
    ------
    ValueError
        If an unknown CLI type is specified.
    """
    opts = dict(opts)

    if 'type' in opts and isinstance(opts['type'], str):
        type_name = opts['type']
        if type_name not in TYPE_MAP:
            raise ValueError(f"Unknown CLI type: {type_name}")
        opts['type'] = TYPE_MAP[type_name]

    return opts

def maxwarn(value):
    """
    Given a maxwarn specification, split it in a warning type, and the number
    to ignore.

    >>> maxwarn('3')
    (None, 3)
    >>> maxwarn('general:15')
    ('general', 15)
    >>> maxwarn('inconsistent-data')
    ('inconsistent-data, None)

    Parameters
    ----------
    value: str
        A warning type and a count, separated by a colon.

    Returns
    -------
    tuple[str, int]
        A warning type and the associated count to ignore. Either element can be
        None if not specified.

    Raises
    ------
    argparse.ArgumentTypeError
    """
    msg = (
        "Values for the -maxwarn option must be the name of a "
        "warning type, a number, or following the format "
        "'<warning-type>:<count>' where <warning-type> is the name "
        "of the warning type to ignore, and <count> is the number of "
        "warning of that type to ignore. "
        "'{value}' is not a valid value.".format(value=value)
    )
    splitted = value.split(":")
    if len(splitted) == 1:
        try:
            count = int(value)
        except ValueError:
            # The value is not an int, so a warning type to ignore an
            # an unspecified number of
            return (value, None)
        else:
            return (None, count)
    elif len(splitted) == 2:
        try:
            count = int(splitted[1])
        except ValueError:
            pass  # The exception will be raised at the end of the function
        else:
            return (splitted[0], count)
    raise argparse.ArgumentTypeError(msg)


# translation table 
TYPE_MAP = {
    'str': str,
    'int': int,
    'float': float,
    'path': Path,
    'cys_argument': _cys_argument,
    'water_bias': water_bias,
    'ignore_resname': ignore_resname,
    'maxwarn': maxwarn,
}

#building a mini parser with the pipelines that we want because we need to know what forcefield to use. 
def build_mini_parser():
    """
    Build the preliminary Martinize2 command-line parser.

    The mini parser handles options required before the full pipeline
    configuration and dynamic CLI are constructed.

    Returns
    -------
    argparse.ArgumentParser
        Parser containing the preliminary command-line options.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)

    parser.add_argument(
        "-pipeline",
        nargs="+",
        default=["charmm", "martini3001"],
        help="Pipeline YAML fragments to combine in order.",
    )

    parser.add_argument(
        "-pipeline-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory to search pipeline YAML files in.",
    )

    parser.add_argument(
        "-extra_ff_dir",
        dest="extra_ff_dir",
        action="append",
        default=[],
        type=Path,
    )

    parser.add_argument(
        "-extra_map_dir",
        dest="extra_map_dir",
        action="append",
        default=[],
        type=Path,
    )

    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        help="Enable debug logging output. Can be given multiple times.",
        default=0,
    )

    parser.add_argument(
        "-maxwarn",
        dest="maxwarn",
        type=maxwarn,
        action="append",
        nargs="+",
        default=[],
        help="The maximum number of allowed warnings. If "
        "more warnings are encountered no output files are"
        " written.",
    )

    parser.add_argument("-list_ff", action="store_true")

    return parser

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
    path = Path(name)

    # User specified path
    if path.exists():
        return path

    # search in the user-specified directories
    for directory in pipeline_dirs:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.exists():
            return candidate

    # Standard location
    candidate = vermouth.DATA_PATH / "pipelines" / f"{name}.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find pipeline YAML '{name}'.")

def add_cli_flag(base_group, flag, opts, prefix='-'):
    opts = dict(opts)
    cli_name = opts.pop("cli", flag)
    opts = translate_cli_opts(opts)
    base_group.add_argument(
        f"{prefix}{cli_name}",
        f"{prefix}{flag}",
        dest=flag,
        **opts,
    )

# build the CLI based on the pipeline configuration.
def build_cli(name, pipeline_conf, prefix, parser=None, added_flags=None, **kwargs):
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
        if flag in added_flags:
            continue
        # make a options dict from the options defined in the yaml. and translate the type from a string to a real python type.
        add_cli_flag(parser, flag, opts, prefix)
        added_flags.add(flag)

    for excl_group in cli_conf.get('exclusive_groups', []):
        group = parser.add_mutually_exclusive_group(**{k: v for k, v in excl_group.items() if k != 'flags'})
        for flag, opts in excl_group.get('flags', {}).items():
            if flag in added_flags:
                continue
            add_cli_flag(group, flag, opts, prefix)
            added_flags.add(flag)

    for grp in cli_conf.get('groups', []):
        group = parser.add_argument_group(**{k: v for k, v in grp.items() if k not in ('flags', 'exclusive_groups')})
        for flag, opts in grp.get('flags', {}).items():
            if flag in added_flags:
                continue
            add_cli_flag(group, flag, opts, prefix)
            added_flags.add(flag)
        for excl_group in grp.get('exclusive_groups', []):
            exclusive_group = group.add_mutually_exclusive_group(**{k: v for k, v in excl_group.items() if k != 'flags'})
            for flag, opts in excl_group.get('flags', {}).items():
                if flag in added_flags:
                    continue
                add_cli_flag(exclusive_group, flag, opts, prefix)
                added_flags.add(flag)

    # recursion for steps in the pipeline
    if pipeline_conf.get('steps'):
        for name, step in pipeline_conf["steps"].items():
            build_cli(name, step, prefix, parser=parser, added_flags=added_flags)

    return parser


# evaluete the condition with the cli values 
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
    # every condition can only have 1 key 
    if len(condition) != 1:
        raise ValueError(
            f"Condition must contain exactly one condition type, got {condition}."
        )
    
    # get the type and arguments of the condition
    type_, args = next(iter(condition.items()))
    # what type of condition is it and what to do with it 
    match type_:
        case 'any':
            verdict = any(eval_condition(c, cli_args, variables) for c in args)
        case 'all':
            verdict = all(eval_condition(c, cli_args, variables) for c in args)
        case 'not':
            verdict = not eval_condition(args, cli_args, variables)
        case 'equal':
            if 'cli' in args:
                verdict = cli_args[args['cli']] == args['value']
            elif 'variable' in args:
                verdict = variables[args['variable']] == args['value']
            else:
                raise ValueError("equal condition needs 'cli' or 'variable'")
        case "has_variable":
            obj = variables[args['variable']]
            verdict = args["key"] in obj.variables 
        case _:
            raise ValueError(f"Unknown condition type: {type_}")

    return verdict

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
            if "value" in value:
                # check if its a fixed value
                args[arg_name] = value['value']
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
            for name, step in child:
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
    if path in inclusion_chain:
        chain = " -> ".join(map(str, (*inclusion_chain, path)))
        raise ValueError(f"Include cycle detected: {chain}")

    config = deepcopy(load_yaml_file(path))
    _annotate_sources(config, path)
    _validate_raw_step_names(config["martinize2"], source=path)
    _convert_step_mappings(config["martinize2"])
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
        return value

    includes = _pop_directive(value, INCLUDE_KEY, [])
    local = deepcopy(value)
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
    yield from  cli_conf.get("flags", {}).items()
    for excl_group in cli_conf.get('exclusive_groups', []):
        yield from excl_group.get('flags', {}).items()
    for group_conf in cli_conf.get('groups', []):
        for excl_group in group_conf.get('exclusive_groups', []):
            yield from excl_group.get('flags', {}).items()
        yield from group_conf.get('flags', {}).items()

    # recursion for steps in the pipeline
    if pipeline_conf.get("steps"):
        for name, step in pipeline_conf["steps"].items():
            yield from iter_cli_flags(step)

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

    return target
    
def combine_pipeline_configs(configs):
    """
    Combine multiple pipeline YAML configs into one pipeline config.

    Duplicate CLI flags are allowed only if their definitions are exactly equal.
    Variables retain their declared names; callers must explicitly rename
    conflicting variables at an include site. Steps are composed by mapping
    key in the order given by the user.
    """
    # TODO: Check whether $schema is the same for all, and use that to validate the final pipeline?
    combined = {
        "cli": {},
        "variables": [],
        "steps": OrderedDict(),
    }

    seen_cli_flags = {}

    for _, conf in configs:
        root = conf["martinize2"]

        # Included fragments retain their lexical variable names. Callers that
        # combine otherwise-conflicting roots must rename variables explicitly.
        for variable in root.get("variables", []):
            if variable not in combined["variables"]:
                combined["variables"].append(variable)

        # merge normal CLI flags
        cli_conf = root.get("cli", {})
        all_cli_flags = dict(iter_cli_flags(root))
        for flag, opts in all_cli_flags.items():
            if flag in seen_cli_flags:
                if seen_cli_flags[flag] != opts:
                    raise ValueError(
                        f"CLI flag {flag!r} is defined multiple times "
                        "with different options."
                    )
            else:
                seen_cli_flags[flag] = opts

        combined['cli'] = merge_dictionaries(combined["cli"], cli_conf)
        merge_pipeline_mapping(
            combined,
            {
                "steps": root.get("steps", OrderedDict()),
            },
        )

    return combined


def merge_dictionaries(dict1, dict2):
    """
    Recursively merge dictionaries with support for lists and scalar values.

    When keys overlap, values from ``dict2`` take precedence, except for
    nested dictionaries and lists, which are merged recursively.
    """
    if not isinstance(dict1, MutableMapping) or not isinstance(dict2, Mapping):
        raise TypeError("merge_dictionaries expects two mappings.")

    def _compatible_types(value1, value2):
        if isinstance(value1, Mapping) and isinstance(value2, Mapping):
            return True
        if (
            isinstance(value1, MutableSequence)
            and isinstance(value2, MutableSequence)
        ):
            return True
        return (
            isinstance(value1, value2.__class__)
            or isinstance(value2, value1.__class__)
        )

    def _merge_values(value1, value2, path):
        if isinstance(value1, MutableMapping) and isinstance(value2, Mapping):
            return _merge_dicts(value1, value2, path)

        if (
            isinstance(value1, MutableSequence)
            and isinstance(value2, MutableSequence)
        ):
            merged = deepcopy(value1)
            for index, item2 in enumerate(value2):
                if index < len(merged):
                    item1 = merged[index]
                    if (
                        isinstance(item1, (Mapping, MutableSequence))
                        or isinstance(item2, (Mapping, MutableSequence))
                    ):
                        if not _compatible_types(item1, item2):
                            raise TypeError(
                                f"Type mismatch at {path}[{index}]: "
                                f"{item1.__class__.__name__} vs "
                                f"{item2.__class__.__name__}."
                            )
                        merged[index] = _merge_values(item1, item2, f"{path}[{index}]")
                    elif _compatible_types(item1, item2):
                        merged[index] = deepcopy(item2)
                    else:
                        raise TypeError(
                            f"Type mismatch at {path}[{index}]: "
                            f"{item1.__class__.__name__} vs "
                            f"{item2.__class__.__name__}."
                        )
                else:
                    merged.append(deepcopy(item2))
            return merged

        if not _compatible_types(value1, value2):
            raise TypeError(
                f"Type mismatch at {path}: "
                f"{value1.__class__.__name__} vs "
                f"{value2.__class__.__name__}."
            )

        return deepcopy(value2)

    def _merge_dicts(left, right, path):
        merged = deepcopy(left)
        for key, value2 in right.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in merged:
                merged[key] = _merge_values(merged[key], value2, child_path)
            else:
                merged[key] = deepcopy(value2)
        return merged

    return _merge_dicts(dict1, dict2, path="")


class PipelineConfigBuilder:
    """
    Build a combined pipeline configuration from YAML files.

    Parameters
    ----------
    pipeline_names : Iterable[str]
        Names or paths of pipeline YAML files.
    pipeline_dirs : Iterable[pathlib.Path], optional
        Additional directories in which pipeline files are searched.
    """
    def __init__(self, pipeline_names, pipeline_dirs=None):
        self.pipeline_names = pipeline_names
        self.pipeline_dirs = pipeline_dirs or []
        self.paths = []

    def build_config(self):
        """
        Load, combine, and validate the pipeline configuration.

        Returns
        -------
        tuple[list[tuple[str, dict]], dict]
            Loaded individual configurations and the combined pipeline
            configuration.
        """
        self.paths = list(find_pipeline_configs(self.pipeline_names, self.pipeline_dirs))
        LOGGER.debug('Building a pipeline from {}', ', '.join(map(str, self.paths)))
        configs = load_pipeline_configs(self.paths, self.pipeline_dirs)
        pipeline_conf = combine_pipeline_configs(configs)
        validate_cli_options(pipeline_conf, path="martinize2")
        return configs, pipeline_conf


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
    def __init__(self, name, pipeline_conf, prefix="-"):
        self.name = name
        self.pipeline_conf = pipeline_conf
        self.prefix = prefix
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
        self._argparser = build_cli(self.name, self.pipeline_conf, self.prefix, **kwargs)

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
    def __init__(self, pipeline_conf):
        self.pipeline_conf = pipeline_conf

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
        _strip_source_metadata(pipeline_conf)
        _convert_step_mappings(pipeline_conf)
        set_values(pipeline_conf, cli_args, variables)

        return Pipeline.from_dict(
            pipeline_conf,
            "martinize2",
        )
    
