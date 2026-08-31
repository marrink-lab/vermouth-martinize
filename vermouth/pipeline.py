from pathlib import Path
from copy import deepcopy
import argparse
import importlib
import yaml
from vermouth.processors.processor import Pipeline



# validate conditions  
def _options_used_in_condition(condition):
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
    try:
        letter, epsilon = value.split(":")
        return letter, float(epsilon)
    except Exception:
        raise argparse.ArgumentTypeError(
                'value must be a letter and a float separated by a colon'
    )
def ignore_resname(value):
    return [item.strip() for item in value.split(",") if item.strip()]

def translate_cli_opts(opts):
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

    parser.add_argument("-extra_ff_dir", action="append", default=[], type=Path)
    parser.add_argument("-extra_map_dir", action="append", default=[], type=Path)
    parser.add_argument("-list_ff", action="store_true")

    return parser

def find_pipeline_yaml(name, pipeline_dirs):
    path = Path(name)

    # User gaf een volledig pad op
    if path.exists():
        return path

    # Zoek in opgegeven directories
    for directory in pipeline_dirs:
        candidate = Path(directory) / f"{name}.yaml"
        if candidate.exists():
            return candidate

    # Standaard locatie
    candidate = Path(__file__).parent / "data" / "pipelines" / f"{name}.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Could not find pipeline YAML '{name}'.")

# build the CLI based on the pipeline configuration.
def build_cli(pipeline_conf, prefix, parser=None, added_flags = None, **kwargs):
    # make parser if not given, otherwise use the given one.
    parser = parser or argparse.ArgumentParser(allow_abbrev=False, **kwargs)
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

        parser.add_argument(
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

        group = parser.add_mutually_exclusive_group()

        for flag, opts in flags_to_add:
            # make the options dict from the options defined in the yaml. and translate the type from a string to a real python type.
            opts = translate_cli_opts(opts)

            group.add_argument(f'{prefix}{flag}', **opts)
            added_flags.add(flag)
    # recursion for steps in the pipeline
    if pipeline_conf.get('steps'):
        for name, step in pipeline_conf['steps']:
            build_cli(step, prefix, parser=parser, added_flags=added_flags)

    return parser


# evaluete the condition with the cli values 
def eval_condition(condition, cli_args, variables):
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
    # split the processor name into module and name. 
    module, name = processor_name.rsplit('.', 1)
    # import to python module 
    module = importlib.import_module(module)
    # get the processor class from the module 
    proc = getattr(module, name)
    return proc

def namespace_variables(obj, namespace):
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

# load in the yaml files
def load_yaml_file(path):
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
    matches = []

    def search(value):
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

    step_definition must contain:
    - 'id': the ID of the new step;
    - 'processor': the processor to insert;
    - either '$insert_before' or '$insert_after'.
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
    """Build a pipeline configuration from YAML files."""
    def __init__(self, pipeline_names, pipeline_dirs=None):
        self.pipeline_names = pipeline_names
        self.pipeline_dirs = pipeline_dirs or []

    def build_config(self):
        configs = load_pipeline_configs(self.pipeline_names, self.pipeline_dirs)
        pipeline_conf = combine_pipeline_configs(configs)
        validate_cli_options(pipeline_conf, path="martinize2")
        return configs, pipeline_conf


class CLIBuilder:
    """Build a command-line interface from a pipeline configuration."""
    def __init__(self, pipeline_conf, prefix="-"):
        self.pipeline_conf = pipeline_conf
        self.prefix = prefix
        self._argparser = None

    @property
    def argparser(self):
        if self._argparser is None:
            self.build_argparser()
        return self._argparser

    def build_argparser(self, **kwargs):
        self._argparser = build_cli(self.pipeline_conf, self.prefix, **kwargs)

    def parse_cli_args(self, args=None):
        return vars(self.argparser.parse_args(args))

class PipelineBuilder:
    """Build an executable pipeline from a pipeline configuration."""
    def __init__(self, pipeline_conf):
        self.pipeline_conf = pipeline_conf

    def build_pipeline(self, cli_args, variables):
        pipeline_conf = deepcopy(self.pipeline_conf)
        set_values(self.pipeline_conf, cli_args, variables)

        return Pipeline.from_dict(
            self.pipeline_conf,
            "martinize2",
        )
    
