
"""
Arguments parser and config file loader.

When adding new commandline parameters, please make sure to provide a clear and descriptive help message and put it in under a related hierarchy.
"""

import re
import os
from secrets import choice
import sys
import yaml
import time
import argparse
from collections import defaultdict


class ConfigHandlerAlg:

    def __init__(self):
        self.config_file_hierarchies = {
            # Given a hierarchy for each commandline option. This hierarchy is used in yaml config.
            # For example: "batch_size": ["solver", "propagation", "batch_size"] will be an element in this dictionary.
            # The entries will be created in add_argument() method.
        }
        # Stores all arguments according to their hierarchy.
        self.all_args = {}
        # Parses all arguments with their defaults.
        self.defaults_parser = argparse.ArgumentParser()
        # Parses the specified arguments only. Not specified arguments will be ignored.
        self.no_defaults_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
        # Help message for each configuration entry.
        self.help_messages = defaultdict(str)
        # Add all common arguments.
        self.add_common_options()

    def add_common_options(self):
        """
        Add all parameters that are shared by different front-ends.
        """

        # We must set how each parameter will be presented in the config file, via the "hierarchy" parameter.
        # Global Configurations, not specific for a particular algorithm.

        # The "--config" option does not exist in our parameter dictionary.
        self.add_argument('--alg_config', type=str, help='Path to YAML format config file.', hierarchy=None)

        h = ["general"]
        self.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"],
                          help='Select device to run verifier, cpu or cuda (GPU).', hierarchy=h + ["device"])
        self.add_argument("--seed", type=int, default=100, help='Random seed.', hierarchy=h + ["seed"])
        self.add_argument("--train_method", type=str, default="verification", choices=["verification", "fine-tuning"],
                          help='Select the training framework for the barrier function.', hierarchy=h+["train_method"])

        h = ["dynamics_fcn"]
        self.add_argument("--train_nn_dynamics", action="store_true", help="Train the NN dynamics.", hierarchy=h+["train_nn_dynamics"])

        h = ["barrier_fcn"]
        self.add_argument("--barrier_output_dim", type=int, default=5, help="Output dimension of the NN barrier function.",
                          hierarchy=h+["barrier_output_dim"])

        h = ["barrier_fcn", "dataset"]
        self.add_argument("--train_data_set_path", type=str, default='data/B_train_dataset.p', hierarchy=h+["train_data_set_path"])
        self.add_argument("--collect_samples", action='store_true',
                          help="Collect samples to train the barrier function.", hierarchy=h+["collect_samples"])
        self.add_argument("--num_samples_x0", type=int, default=5000, hierarchy=h+["num_samples_x0"])
        self.add_argument("--num_samples_xu", type=int, default=5000, hierarchy=h+["num_samples_xu"])
        self.add_argument("--num_samples_x", type=int, default=5000, hierarchy=h+["num_samples_x"])

        h = ["barrier_fcn", "train_options"]
        self.add_argument("--num_epochs", type=int, default=50, hierarchy=h+["num_epochs"])
        self.add_argument("--l1_lambda", type=float, default=0.0, hierarchy=h+["l1_lambda"])
        self.add_argument("--early_stopping_tol", type=float, default=1e-10, hierarchy=h+["early_stopping_tol"])
        self.add_argument("--update_A_freq", type=int, default=1, hierarchy=h+["update_A_freq"])

        h = ["ce_sampling"]
        self.add_argument("--num_ce_samples", type=int, default=20, hierarchy=h+["num_ce_samples"])
        self.add_argument("--opt_iter", type=int, default=100, hierarchy=h+["opt_iter"])
        self.add_argument("--radius", type=float, default=0.1, hierarchy=h+["radius"])

        h = ["ACCPM"]
        self.add_argument("--max_iter", type=int, default=30, hierarchy=h+["max_iter"])

        h = ["bab"]
        self.add_argument("--bab_yaml_path", type=str, default='ab_crown_quad.yaml', hierarchy=h+["bab_yaml_path"])

    def add_argument(self, *args, **kwargs):
        """Add a single parameter to the parser. We will check the 'hierarchy' specified and then pass the remaining arguments to argparse."""
        if 'hierarchy' not in kwargs:
            raise ValueError("please specify the 'hierarchy' parameter when using this function.")
        hierarchy = kwargs.pop('hierarchy')
        help = kwargs.get('help', '')
        private_option = kwargs.pop('private', False)
        # Make sure valid help is given
        if not private_option:
            if len(help.strip()) < 10:
                # raise ValueError(
                #     f'Help message must not be empty, and must be detailed enough. "{help}" is not good enough.')
                pass
            elif (not help[0].isupper()) or help[-1] != '.':
                raise ValueError(
                    f'Help message must start with an upper case letter and end with a dot (.); your message "{help}" is invalid.')
            elif help.count('%') != help.count('%%') * 2:
                raise ValueError(
                    f'Please escape "%" in help message with "%%"; your message "{help}" is invalid.')
        self.defaults_parser.add_argument(*args, **kwargs)
        # Build another parser without any defaults.
        if 'default' in kwargs:
            kwargs.pop('default')
        self.no_defaults_parser.add_argument(*args, **kwargs)
        # Determine the variable that will be used to save the argument by argparse.
        if 'dest' in kwargs:
            dest = kwargs['dest']
        else:
            dest = re.sub('^-*', '', args[-1]).replace('-', '_')
        # Also register this parameter to the hierarchy dictionary.
        self.config_file_hierarchies[dest] = hierarchy
        if hierarchy is not None and not private_option:
            self.help_messages[','.join(hierarchy)] = help

    def set_dict_by_hierarchy(self, args_dict, h, value, nonexist_ok=True):
        """Insert an argument into the dictionary of all parameters. The level in this dictionary is determined by list 'h'."""
        # Create all the levels if they do not exist.
        current_level = self.all_args
        assert len(h) != 0
        for config_name in h:
            if config_name not in current_level:
                if nonexist_ok:
                    current_level[config_name] = {}
                else:
                    raise ValueError(f"Config key {h} not found!")
            last_level = current_level
            current_level = current_level[config_name]
        # Add config value to leaf node.
        last_level[config_name] = value

    def construct_config_dict(self, args_dict, nonexist_ok=True):
        """Based on all arguments from argparse, construct the dictionary of all parameters in self.all_args."""
        for arg_name, arg_val in args_dict.items():
            h = self.config_file_hierarchies[arg_name]  # Get levels for this argument.
            if h is not None:
                assert len(h) != 0
                self.set_dict_by_hierarchy(self.all_args, h, arg_val, nonexist_ok=nonexist_ok)

    def update_config_dict(self, old_args_dict, new_args_dict, levels=None):
        """Recursively update the dictionary of all parameters based on the dict read from config file."""
        if levels is None:
            levels = []
        if isinstance(new_args_dict, dict):
            # Go to the next dict level.
            for k in new_args_dict:
                self.update_config_dict(old_args_dict, new_args_dict[k], levels=levels + [k])
        else:
            # Reached the leaf level. Set the corresponding key.
            self.set_dict_by_hierarchy(old_args_dict, levels, new_args_dict, nonexist_ok=False)

    def dump_config(self, args_dict, level=[], out_to_doc=False, show_help=False):
        """Generate a config file based on args_dict with help information."""
        ret_string = ''
        for key, val in args_dict.items():
            if isinstance(val, dict):
                ret = self.dump_config(val, level + [key], out_to_doc, show_help)
                if len(ret) > 0:
                    # Next level is not empty, print it.
                    ret_string += ' ' * (len(level) * 2) + f'{key}:\n' + ret
            else:
                if show_help:
                    h = self.help_messages[','.join(level + [key])]
                    if 'debug' in key or 'not use' in h or 'not be use' in h or 'debug' in h or len(h) == 0:
                        # Skip some debugging options.
                        continue
                    h = f'  # {h}'
                else:
                    h = ''
                yaml_line = yaml.safe_dump({key: val}, default_flow_style=None).strip().replace('{', '').replace('}',
                                                                                                                 '')
                ret_string += ' ' * (len(level) * 2) + f'{yaml_line}{h}\n'
        if len(level) > 0:
            return ret_string
        else:
            # Top level, output to file.
            if out_to_doc:
                output_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'docs',
                                           os.path.splitext(os.path.basename(sys.argv[0]))[0] + '_all_params.yaml')
                with open(output_name, 'w') as f:
                    f.write(ret_string)
            return ret_string

    def parse_config(self):
        """
        Main function to parse parameter configurations. The commandline arguments have the highest priority;
        then the parameters specified in yaml config file. If a parameter does not exist in either commandline
        or the yaml config file, we use the defaults defined in add_common_options() defined above.
        """
        # Parse an empty commandline to get all default arguments.
        default_args = vars(self.defaults_parser.parse_args([]))
        # Create the dictionary of all parameters, all set to their default values.
        self.construct_config_dict(default_args)
        # Update documents.
        # self.dump_config(self.all_args, out_to_doc=True, show_help=True)
        # These are arguments specified in command line.
        specified_args = vars(self.no_defaults_parser.parse_args())
        # Read the yaml config files.
        if 'alg_config' in specified_args:
            with open(specified_args['alg_config'], 'r') as config_file:
                loaded_args = yaml.safe_load(config_file)
                # Update the defaults with the parameters in the config file.
                self.update_config_dict(self.all_args, loaded_args)
        # Finally, override the parameters based on commandline arguments.
        self.construct_config_dict(specified_args, nonexist_ok=False)
        # For compatibility, we still return all the arguments from argparser.
        parsed_args = self.defaults_parser.parse_args()
        # Print all configuration.
        print('Configurations:\n')
        print(self.dump_config(self.all_args))
        return parsed_args

    def keys(self):
        return self.all_args.keys()

    def items(self):
        return self.all_args.items()

    def __getitem__(self, key):
        """Read an item from the dictionary of parameters."""
        return self.all_args[key]

    def __setitem__(self, key, value):
        """Set an item from the dictionary of parameters."""
        self.all_args[key] = value


class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("You must register a global parameter in arguments.py.")

    def __setitem__(self, key, value):
        if key not in self:
            raise RuntimeError("You must register a global parameter in arguments.py.")
        else:
            super().__setitem__(key, value)

    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


# Global configuration variable
Config = ConfigHandlerAlg()
# Global variables
# Globals = ReadOnlyDict({"starting_timestamp": int(time.time()), "example_idx": -1, "lp_perturbation_eps": None})
