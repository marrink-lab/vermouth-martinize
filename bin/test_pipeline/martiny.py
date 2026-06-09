from pprint import pprint
from pathlib import Path
import vermouth
import vermouth.forcefield
import yaml
from vermouth.processors.processor import Pipeline
import argparse
import importlib
import sys
import logging
from vermouth.log_helpers import CountingHandler, ignore_warnings_and_count
from vermouth.file_writer import DeferredFileWriter
from vermouth import DATA_PATH
from vermouth.map_input import (
    read_mapping_directory,
    generate_all_self_mappings,
    combine_mappings,
)


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("vermouth")
COUNTER = CountingHandler()
COUNTER.setLevel(logging.WARNING)
LOGGER.addHandler(COUNTER)

sys.path.insert(0, r"C:\Users\roord\Documents\Stage_git\vermouth-martinize\bin")

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
            cli_refs.add(cond['cli'])

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
# function for the variable options, this will set the variables in the yaml to the values from the CLI.
def variable_options(pipeline_conf, args, namespace, **variables):
    # loop through all the variables defined in the pipeline. 
    for variable in pipeline_conf.get("variables", []):
        # make a namespaced variable name, so like charmm + the variable ff --> charmm.ff. 
        namespaced_variables = f"{namespace}.{variable}" 
        # connect the variable name to the value from the CLI. so charmm.ff will get the value of ff from the CLI.
        args[namespaced_variables] = variables[variable]



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

#building a mini parser to get the to_ff and from_ff values so that the yaml knows what they are. because the yaml depends on the cli. 
def build_mini_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-from_ff", default="charmm")
    parser.add_argument("-to_ff", default="martini3001")
    parser.add_argument("-extra_ff_dir", action="append", default=[], type=Path)
    parser.add_argument("-extra_map_dir", action="append", default=[], type=Path)
    parser.add_argument("-list_ff", action="store_true")
    return parser


# build the CLI based on the pipeline configuration.
def build_cli(pipeline_conf, prefix, parser=None, added_flags = None, **kwargs):
    # make parser if not given, otherwise use the given one.
    parser = parser or argparse.ArgumentParser(**kwargs)
    # make an empty set of the added_flags. or use the given one. 
    added_flags = set() if added_flags is None else added_flags
    # loop through the cli flags defined in the pipeline config. and don't add the same flag twice. 
    for flag, opts in pipeline_conf.get('cli_flags', {}).items():
        if flag in added_flags:
            continue 
        # make a options dict from the options defined in the yaml. and translate the type from a string to a real python type.
        opts = translate_cli_opts(opts)
        parser.add_argument(f'{prefix}{flag}', **opts)
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
def eval_condition(condition, cli_values):
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
            verdict = any(eval_condition(c, cli_values) for c in args)
        case 'all':
            verdict = all(eval_condition(c, cli_values) for c in args)
        case 'not':
            verdict = not eval_condition(args, cli_values)
        case 'equal':
            if 'cli' in args:
                verdict = cli_values[args['cli']] == args['value']
            elif 'variable' in args:
                verdict = cli_values[args['variable']] == args['value']
            else:
                raise ValueError("equal condition needs 'cli' or 'variable'")
        case "has_variable":
            obj = cli_values[args['variable']]
            verdict = args["key"] in obj.variables 
        case _:
            raise ValueError(f"Unknown condition type: {type_}")

    return verdict

# set the values from the CLI into the pipeline config 
def set_values_from_cli(pipeline_conf, cli_values):
    # check if there is a condition 
    if 'condition' in pipeline_conf:
        pipeline_conf['condition'] = eval_condition(pipeline_conf['condition'], cli_values)
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
                args[arg_name] = cli_values[value['cli']]
                # set the arg value good 
            elif 'variable' in value:
                args[arg_name] = cli_values[value['variable']]
            else: 
                raise KeyError(f"{arg_name} must have a value, cli, or variable")
        pipeline_conf['args'] = args
    # if its recursive pipeline, do the same for the steps in the pipeline
    for name, step in pipeline_conf.get('steps', []):
        if not step.get('steps'):
            # go from text to actual processor object
            step['processor'] = import_processor(name)
        # call itself 
        set_values_from_cli(step, cli_values)

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


# load in the pipeline halves from the yaml files.
def load_pipeline_halves(script_dir, from_ff, to_ff):
    # source yaml file. 
    from_path = script_dir / "pipelines" / "from" / f"{from_ff}.yaml"
    # target yaml file.
    to_path = script_dir / "pipelines" / "to" / f"{to_ff}.yaml"

    # load in the yaml as dicts
    from_file = load_yaml_file(from_path)
    to_file = load_yaml_file(to_path)

    # only take the part of the yaml that is relevant for martinize2. 
    from_conf = from_file.get("martinize2", from_file)
    to_conf = to_file.get("martinize2", to_file)

    from_conf = namespace_variables(from_conf, from_ff)
    to_conf = namespace_variables(to_conf, to_ff)

    return from_conf, to_conf

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

# combine the actuall pipeline halves into one pipeline config.
def combine_pipeline_halves(from_conf, to_conf, from_ff, to_ff):
    # gather all cli flags 
    seen_cli_flags = {}
    # check for duplicates and combine the cli flags from both halves.
    for conf in (from_conf, to_conf):
        for flag, opts in iter_cli_flags(conf):
            if flag in seen_cli_flags:
                if seen_cli_flags[flag] != opts:
                    raise KeyError(
                        f"CLI flag '{flag}' is defined multiple times "
                        f"with different definitions."
                    )
                continue

            seen_cli_flags[flag] = opts

    return { # give back one big pipeline config with all the info.
        "variables": (
            # namespace the variables with the forcefield they come from. 
            # so for example ff will become charmm.ff. because from_ff = charmm and var is the variable name, like ff. 
            [f"{from_ff}.{var}" for var in from_conf.get("variables", [])] +
            [f"{to_ff}.{var}" for var in to_conf.get("variables", [])]
        ),
        "cli_flags": seen_cli_flags,
        # return the steps from both halves. 
        "steps": from_conf.get("steps", []) + to_conf.get("steps", []),
    }

# path to yamls.
script_dir = Path(__file__).resolve().parent
# which prefix to use
CLI_PREFIX = "-"
#name of program
name = "martinize2"

# makes the mini parser with only the flags defined above.
mini_parser = build_mini_parser()
# parse the known args. and _ the unknown args. 
mini_args, _ = mini_parser.parse_known_args()

# choose the pipeline halves based on the from_ff and to_ff values given in the CLI.
from_conf, to_conf = load_pipeline_halves(
    script_dir,
    mini_args.from_ff,
    mini_args.to_ff,
)
# make one pipelineconfig containing variables, cli_flags and steps. 
pipeline_conf = combine_pipeline_halves(from_conf, to_conf, mini_args.from_ff, mini_args.to_ff)
# validate the cli options 
validate_cli_options(pipeline_conf, path=name)

# build real parser. by calling the build_cli function. 
parser = build_cli(pipeline_conf, CLI_PREFIX, prog=name)

# parser the args with the real parser. this will give us all the values from the CLI.
args = parser.parse_args()


def force_fields(args, parser):
    known_force_fields = vermouth.forcefield.find_force_fields(
        Path(DATA_PATH) / "force_fields"
    )
    known_mappings = read_mapping_directory(
        Path(DATA_PATH) / "mappings", known_force_fields
    )

    for directory in args["extra_ff_dir"]:
        try:
            vermouth.forcefield.find_force_fields(directory, known_force_fields)
        except FileNotFoundError as error:
            msg = '"{}" given to the -ff-dir option should be a directory.'
            raise ValueError(msg.format(directory)) from error

    for directory in args["extra_map_dir"]:
        try:
            partial_mapping = read_mapping_directory(directory, known_force_fields)
        except NotADirectoryError as error:
            msg = '"{}" given to the -map-dir option should be a directory.'
            raise ValueError(msg.format(directory)) from error
        combine_mappings(known_mappings, partial_mapping)

    if args["list_ff"]:
        print("The following force fields are known:")
        for idx, ff_name in enumerate(reversed(list(known_force_fields)), 1):
            print("{:3d}. {}".format(idx, ff_name))
        parser.exit()

    partial_mapping = generate_all_self_mappings(known_force_fields.values())
    combine_mappings(known_mappings, partial_mapping)

    if args["to_ff"] not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(args["to_ff"]))
    if args["from_ff"] not in known_force_fields:
        raise ValueError('Unknown force field "{}".'.format(args["from_ff"]))
    return (
    known_force_fields[args["from_ff"]],
    known_force_fields[args["to_ff"]],
    known_mappings,
)


# make the args into a dict 
args = vars(args)

# if you give noscfix, scfix is faslse otherwise true.
args["scfix"] = not args["noscfix"]
args["deduplicate"] = not args["keep_duplicate_itp"]

source_ff, target_ff, mappings = force_fields(args, parser)

variable_options(from_conf, args, args["from_ff"], ff=source_ff)
variable_options(to_conf, args, args["to_ff"], ff=target_ff, mappings=mappings)

# check for conditions, yaml and cli and variables will be python values, load processor objects
set_values_from_cli(pipeline_conf, args)

# make the pipeline object from the pipeline config
pipeline = Pipeline.from_dict(pipeline_conf, name)

#print the pipeline 
print("Pipeline object:")
print(pipeline)
print()
print(args.keys())

# make an empty vermouth system. 
system = vermouth.System(force_field=source_ff)


print("FROM YAML:", mini_args.from_ff)
print("TO YAML:", mini_args.to_ff)
print("CLI FLAGS:", pipeline_conf["cli_flags"].keys())

# test the pipeline with None 
pipeline.run_system(system)
leftover_warnings = ignore_warnings_and_count(COUNTER, args["maxwarn"])

if leftover_warnings:
    LOGGER.error(
        "{} warnings were encountered after accounting for the "
        "-maxwarn flag. No output files will be "
        "written. Consider fixing the warnings, or if you are sure "
        "they are harmless, use the -maxwarn flag.",
        leftover_warnings,
    )
    sys.exit(2)
else:
    DeferredFileWriter().write()
    vermouth.Quoter().run_system(system)
print(system.meta.get("header"))

