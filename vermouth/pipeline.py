from html import parser
from pathlib import Path
from copy import deepcopy
import vermouth 
import argparse
import importlib
import yaml
from vermouth.processors.processor import Pipeline



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
    normal_cli_options = set(pipeline_conf.get('cli_flags', {}).keys())

    # gather flags defined in cli_groups
    group_cli_options = set()
    for group_conf in pipeline_conf.get('cli_groups', []):
        group_cli_options |= set(group_conf.get('flags', {}).keys())
    
    # force_field variable options
    variable_options = set(pipeline_conf.get("variables", []))

    # add to the sets of options defined in this scope and globally
    local_cli_options |= normal_cli_options | group_cli_options
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
        for idx, (name, step) in enumerate(pipeline_conf['steps']):
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
        "-override",
        type=Path,
        default=None,
        help="Pipeline override YAML file.",
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

# build the CLI based on the pipeline configuration.
def build_cli(name, pipeline_conf, prefix, parser=None, added_flags = None, **kwargs):
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
    base_group = parser.add_argument_group(name)
    # make an empty set of the added_flags. or use the given one. 
    added_flags = set() if added_flags is None else added_flags
    # loop through the cli flags defined in the pipeline config. and don't add the same flag twice. 
    for flag, opts in pipeline_conf.get('cli_flags', {}).items():
        if flag in added_flags:
            continue 
        # make a options dict from the options defined in the yaml. and translate the type from a string to a real python type.
        opts = dict(opts)
        cli_name = opts.pop("cli", flag)
        opts = translate_cli_opts(opts)

        base_group.add_argument(
            f"{prefix}{cli_name}",
            f"{prefix}{flag}",
            dest=flag,
            **opts,
        )
        added_flags.add(flag)
    # add CLI Flags from the CLI groups. 
    for group_cli in pipeline_conf.get('cli_groups', []):
        flags_to_add = [
            (flag, opts)
            for flag, opts in group_cli.get('flags', {}).items()
            if flag not in added_flags
        ]

        # If all flags were already added earlier, don't create an empty group.
        if not flags_to_add:
            continue

        group = base_group.add_mutually_exclusive_group()

        for flag, opts in flags_to_add:
            # make the options dict from the options defined in the yaml. and translate the type from a string to a real python type.
            opts = translate_cli_opts(opts)

            group.add_argument(f'{prefix}{flag}', **opts)
            added_flags.add(flag)
    # recursion for steps in the pipeline
    if pipeline_conf.get('steps'):
        for name, step in pipeline_conf['steps']:
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
    for name, step in pipeline_conf.get('steps', []):
        if not step.get('steps'):
            # go from text to actual processor object
            step['processor'] = import_processor(name)
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

def namespace_variables(obj, namespace):
    """
    Add a namespace to variable references in a configuration object.

    The configuration is traversed recursively and values associated with a
    ``variable`` key are prefixed with the supplied namespace.

    Parameters
    ----------
    obj : object
        Configuration object to process.
    namespace : str
        Namespace to prepend to variable references.

    Returns
    -------
    object
        The configuration object with namespaced variable references.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "variable" and isinstance(value, str):
                obj[key] = f"{namespace}.{value}"
            else:
                namespace_variables(value, namespace)
    elif isinstance(obj, list): 
        for item in obj: 
            namespace_variables(item, namespace)
    elif isinstance(obj, tuple):
        for item in obj:
            namespace_variables(item, namespace)
    return obj


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
        return yaml.safe_load(file)


def load_pipeline_configs(pipeline_names, pipeline_dirs):
    """
    Load multiple pipeline YAML configs.

    Parameters
    ----------
    pipeline_names : list[str]
        Names or paths of YAML pipeline fragments.
    pipeline_dirs : list[pathlib.Path]
        Extra directories to search in.

    Returns
    -------
    list[tuple[str, dict]]
        List of (namespace, config) pairs.
    """
    configs = []

    for name in pipeline_names:
        path = find_pipeline_yaml(name, pipeline_dirs)
        conf = load_yaml_file(path)

        namespace = Path(name).stem
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
    for flag, opts in pipeline_conf.get("cli_flags", {}).items():
        # using yield so that it saves time and memory by not creating a big list of all the flags, but instead giving them one by one.
        yield flag, opts

    # gather cli_flags defined in cli_groups
    for group_conf in pipeline_conf.get("cli_groups", []):
        for flag, opts in group_conf.get("flags", {}).items():
            yield flag, opts

    # recursion for steps in the pipeline
    if pipeline_conf.get("steps"):
        for name, step in pipeline_conf["steps"]:
            yield from iter_cli_flags(step)

def find_step_by_id(config, target_id, raise_if_missing=True):
    """
    Find a pipeline step by its ID or processor name.

    If a processor does not define an explicit ID, its processor name is used
    as its effective ID.

    Parameters
    ----------
    config : object
        Pipeline configuration to search.
    target_id : str
        ID or processor name to find.
    raise_if_missing : bool, optional
        Raise an error when no matching step is found.

    Returns
    -------
    dict or None
        Matching processor configuration, or ``None`` when no match exists and
        ``raise_if_missing`` is false.

    Raises
    ------
    KeyError
        If no matching processor is found and ``raise_if_missing`` is true.
    ValueError
        If more than one processor matches the requested ID.
    """
    matches = []

    def search(value):
        """
        Recursively search the configuration for matching pipeline steps.
        """
        if isinstance(value, dict):
            for child in value.values():
                search(child)

        elif isinstance(value, list):
            for child in value:
                if isinstance(child, tuple) and len(child) == 2:
                    processor_name, step_conf = child

                    if not isinstance(step_conf, dict):
                        continue

                    # Explicit id, otherwise processor name as default id
                    effective_id = step_conf.get("id", processor_name)

                    if effective_id == target_id:
                        matches.append(step_conf)

                    search(step_conf)
                else:
                    search(child)

    search(config)

    if not matches:
        if raise_if_missing:
            raise KeyError(
                f"No pipeline step found with id or processor name "
                f"{target_id!r}."
            )
        return None

    if len(matches) > 1:
        raise ValueError(
            f"Pipeline step {target_id!r} is not unique. "
            "Add explicit unique ids to these processors."
        )

    return matches[0]

ARG_SOURCE_KEYS = {"cli", "value", "variable"}
REMOVE_VALUE = "$remove"
STRATEGY_KEY = "$strategy"
VALID_STRATEGIES = {"merge", "replace"}


def merge_override(target, override):
    """
    Apply an override dictionary to a target dictionary.

    Dictionaries are merged recursively by default.

    If an override dictionary contains 'cli', 'value', or 'variable',
    its default strategy is 'replace', because these keys describe
    alternative argument value sources.

    The default can be changed explicitly with '$strategy'.
    Lists and ordinary values are replaced.
    '$remove' removes a key.
    """
    if not isinstance(target, dict):
        raise TypeError(
            f"Override target must be a dictionary, "
            f"not {type(target).__name__}."
        )

    if not isinstance(override, dict):
        raise TypeError(
            f"Override must be a dictionary, "
            f"not {type(override).__name__}."
        )

    override_values = {
        key: value
        for key, value in override.items()
        if key != STRATEGY_KEY
    }

    default_strategy = (
        "replace"
        if ARG_SOURCE_KEYS.intersection(override_values)
        else "merge"
    )

    strategy = override.get(STRATEGY_KEY, default_strategy)

    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown override strategy {strategy!r}. "
            f"Expected 'merge' or 'replace'."
        )

    if strategy == "replace":
        target.clear()

    for key, override_value in override_values.items():
        if override_value == REMOVE_VALUE:
            target.pop(key, None)
            continue

        target_value = target.get(key)

        if (
            isinstance(target_value, dict)
            and isinstance(override_value, dict)
        ):
            merge_override(target_value, override_value)
        else:
            target[key] = override_value

    return target


def insert_pipeline_step(pipeline_config, step_definition):
    """
    Insert a new processor step into a pipeline configuration.

    Parameters
    ----------
    pipeline_config : dict
        Pipeline configuration in which the new step is inserted.
    step_definition : dict
        Definition of the new processor step. It must contain ``id`` and
        ``processor``, and either ``$insert_before`` or ``$insert_after``.

    Raises
    ------
    ValueError
        If both or neither insertion directives are specified, or if the
        target step is not unique.
    KeyError
        If the target step cannot be found.
    """
    new_step = deepcopy(step_definition)
    new_step_id = new_step.pop("id")

    insert_before = new_step.pop("$insert_before", None)
    insert_after = new_step.pop("$insert_after", None)

    if insert_before is not None and insert_after is not None:
        raise ValueError(
            f"New step {new_step_id!r} cannot use both "
            "'$insert_before' and '$insert_after'."
        )

    anchor_id = insert_before or insert_after

    if anchor_id is None:
        raise ValueError(
            f"New step {new_step_id!r} must use "
            "'$insert_before' or '$insert_after'."
        )

    matches = []

    def search(value):
        if isinstance(value, dict):
            for child in value.values():
                search(child)

        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, tuple) and len(child) == 2:
                    processor_name, step_config = child

                    if not isinstance(step_config, dict):
                        continue

                    effective_id = step_config.get("id", processor_name)

                    if effective_id == anchor_id:
                        matches.append((value, index))

                    search(step_config)
                else:
                    search(child)

    search(pipeline_config)

    if not matches:
        raise KeyError(
            f"No pipeline step found with id or processor name "
            f"{anchor_id!r}."
        )

    if len(matches) > 1:
        raise ValueError(
            f"Pipeline step {anchor_id!r} is not unique. "
            "Add explicit unique ids to these processors."
        )

    step_list, anchor_index = matches[0]

    insert_index = (
        anchor_index
        if insert_before is not None
        else anchor_index + 1
    )

    processor_name = new_step.pop("processor")
    new_step["id"] = new_step_id

    step_list.insert(
        insert_index,
        (processor_name, new_step),
    )
    
def combine_pipeline_configs(configs):
    """
    Combine multiple pipeline YAML configs into one pipeline config.

    Duplicate CLI flags are allowed only if their definitions are exactly equal.
    Variables are namespaced per YAML fragment.
    Steps are appended in the order given by the user.
    """
    combined = {
        "cli_flags": {},
        "cli_groups": [],
        "variables": [],
        "steps": [],
    }

    seen_cli_flags = {}

    for namespace, conf in configs:
        root = conf["martinize2"]

        # namespace all variable references inside this YAML
        namespace_variables(root, namespace)

        # collect namespaced variables
        for variable in root.get("variables", []):
            namespaced_variable = f"{namespace}.{variable}"

            if namespaced_variable not in combined["variables"]:
                combined["variables"].append(namespaced_variable)

        # merge normal CLI flags
        for flag, opts in root.get("cli_flags", {}).items():
            if flag in seen_cli_flags:
                if seen_cli_flags[flag] != opts:
                    raise ValueError(
                        f"CLI flag {flag!r} is defined multiple times "
                        "with different options."
                    )
            else:
                seen_cli_flags[flag] = opts
                combined["cli_flags"][flag] = opts

        # merge CLI groups
        combined["cli_groups"].extend(root.get("cli_groups", []))

        # append pipeline steps in order
        combined["steps"].extend(root.get("steps", []))

    return combined


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

    def build_config(self):
        """
        Load, combine, and validate the pipeline configuration.

        Returns
        -------
        tuple[list[tuple[str, dict]], dict]
            Loaded individual configurations and the combined pipeline
            configuration.
        """
        configs = load_pipeline_configs(self.pipeline_names, self.pipeline_dirs)
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
        pipeline_conf = deepcopy(self.pipeline_conf)
        set_values(pipeline_conf, cli_args, variables)

        return Pipeline.from_dict(
            pipeline_conf,
            "martinize2",
        )
    
