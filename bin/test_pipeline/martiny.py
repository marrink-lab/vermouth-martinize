from pprint import pprint
from pathlib import Path
import vermouth
import vermouth.forcefield
import yaml
from pipeline_class import Pipeline
import argparse
import importlib
import sys
import logging

logging.basicConfig(level=logging.INFO)
#imports 


sys.path.insert(0, r"C:\Users\roord\Documents\Stage_git\vermouth-martinize\bin")

# validate conditions  
def _cli_options_used_in_condition(condition):
    type_, cond = next(iter(condition.items()))
    result = set()

    match type_:
        case 'all' | 'any':
            for item in cond:
                result = result | _cli_options_used_in_condition(item)
        case 'not':
            result = result | _cli_options_used_in_condition(cond)
        case _:
            result = result | {cond['cli']}

    return result

# validate if options are defines more than once
# are the parameters correct. 
def validate_cli_options(
    pipeline_conf,
    path='',
    reference='cli',
    definition='cli_flags',
    local_cli_options=None,
    global_cli_options=None,
):  # make local and globals if set to empty set if None is given.
    local_cli_options = set() if local_cli_options is None else local_cli_options
    global_cli_options = set() if global_cli_options is None else global_cli_options
    
    # is there a definition of CLI options in this pipeline step?
    if definition in pipeline_conf:
        local_cli_options = local_cli_options | set(pipeline_conf[definition].keys())
        # add new local cli options and dbubbel check 
        if overlap := set(pipeline_conf[definition].keys()).intersection(global_cli_options):
            _path = '.'.join([path, definition])
            raise KeyError(
                f'CLI options {overlap} have already been defined before '
                f'but were found again in {_path}'
            )
        # adding cli options to global set 
        global_cli_options |= set(pipeline_conf[definition])
        

    # is condition beging used in the current pipeline step? 
    if 'condition' in pipeline_conf:
        #find all cli options used in the condition
        cond_options = _cli_options_used_in_condition(pipeline_conf['condition'])
        # are options missing, error path
        if missing := (cond_options - local_cli_options):
            _path = '.'.join([path, "condition"])
            raise KeyError(
                f'CLI option(s) {missing} in {_path} have not been defined. '
                f'Only {local_cli_options} are known in this scope.'
            )
    # is it a pipeline step or a processor step?
    is_pipeline = bool(pipeline_conf.get('steps'))
    if not is_pipeline:
        references = {
            # check if the cli options are used in args and gathers them 
            v[reference]
            for v in pipeline_conf.get('args', {}).values()
            if reference in v
        }   # if missing, error path 
        if missing := (references - local_cli_options):
            _path = '.'.join([path, "args"])
            raise KeyError(
                f'CLI option(s) {missing} in {_path} have not been defined. '
                f'Only {local_cli_options} are known in this scope.'
            )
    else:   # is it a pipline with recursive steps?
        for idx, (name, step) in enumerate(pipeline_conf['steps']):
            _path = '.'.join([path, f'steps[{idx}]', name])
            # the function is called recursively to check the steps in the pipeline.
            validate_cli_options(
                step,
                _path,
                reference,
                definition,
                local_cli_options,
                global_cli_options,
            )

# translation table 
TYPE_MAP = {
    'str': str,
    'int': int,
    'float': float,
    'path': Path,
}
# build the CLI based on the pipeline configuration.
def build_cli(pipeline_conf, prefix, parser=None, **kwargs):
    # make parser if not given, otherwise use the given one.
    parser = parser or argparse.ArgumentParser(**kwargs)
    # add arguments for the current pipeline step
    for flag, opts in pipeline_conf.get('cli_flags', {}).items():
        # make copy of dict 
        opts = dict(opts)
        # translate type from string to actual type if needed.
        if 'type' in opts and isinstance(opts['type'], str):
            type_name = opts['type']
            if type_name not in TYPE_MAP:
                raise ValueError(f"Unknown CLI type: {type_name}")
            opts['type'] = TYPE_MAP[type_name]
        # actually add the argument to the parser
        parser.add_argument(f'{prefix}{flag}', **opts)
    # recursion for steps in the pipeline
    if pipeline_conf.get('steps'):
        for name, step in pipeline_conf['steps']:
            build_cli(step, prefix, parser=parser)

    return parser
# evaluete the condition with the cli values 
def eval_condition(condition, cli_values):
    # every condition can only have 1 key 
    assert len(condition) == 1, condition
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
            verdict = cli_values[args['cli']] == args['value']
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
            if 'value' in value:
                # check if its a fixed value
                args[arg_name] = value['value']
            else:
                # if not, use the CLI value 
                args[arg_name] = cli_values[value['cli']]
                # set the arg value good 
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



# path to yaml file 
script_dir = Path(__file__).resolve().parent
yaml_path = script_dir / "pipeline_test.yaml"

# load in the yaml file and read it 
with open(yaml_path, "r", encoding="utf-8") as file:
    parsed_file = yaml.safe_load(file)

# remove unecessary things from the yaml file and keept the prefix 
parsed_file.pop("$schema", None)
cli_prefix = parsed_file.pop("cli_prefix")
# get the name of the pipeline 
name, pipeline = parsed_file.popitem()

# validate the cli options 
validate_cli_options(pipeline, path=name)

# build the cli with the cli_flags 
parser = build_cli(pipeline, cli_prefix, prog=name)



# fake CLI test
args = parser.parse_args()
# make the args into a dict 
args = vars(args)

# if you give noscfix, scfix is faslse otherwise true.
args["scfix"] = not args["noscfix"]

# keep count of how many secondary structure options are given, if its more then 1, give error.
ss_count = [
    args.get("dssp") is not None,
    args.get("ss") is not None,
    args.get("collagen") is True,
]
if sum(ss_count) > 1:
    raise ValueError("Only one of the secondary structure options can be used at the same time.")   


# check for conditions, fill in args, load processor objects
set_values_from_cli(pipeline, args)

# make the pipeline object from the pipeline config
pipeline = Pipeline.from_json_conf(pipeline, name)

#print the pipeline 
print("Pipeline object:")
print(pipeline)
print()


# load in the forcefield because pdb to universal expects a system with a forcefield.
ff = vermouth.forcefield.get_native_force_field('charmm')
system = vermouth.System(force_field=ff)
# test the pipeline with None 
pipeline.run_system(system)
print(system.meta.get("header"))