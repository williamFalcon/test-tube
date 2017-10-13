from argparse import ArgumentParser
import argparse
import numpy as np
import re
from copy import deepcopy
from .hyper_opt_utils import strategies
import json
import math
import os
from time import sleep


class HyperOptArgumentParser(ArgumentParser):
    """
    Subclass of argparse ArgumentParser which adds optional calls to sample from lists or ranges
    Also enables running optimizations across parallel processes
    """

    def __init__(self, strategy='grid_search', **kwargs):
        """

        :param strategy: 'grid_search', 'random_search'
        :param enabled:
        :param experiment:
        :param kwargs:
        """
        ArgumentParser.__init__(self, **kwargs)

        self.strategy = strategy
        self.trials = []
        self.parsed_args = None
        self.opt_args = {}
        self.json_config_arg_name = None

    def add_argument(self, *args, **kwargs):
        super(HyperOptArgumentParser, self).add_argument(*args, **kwargs)

    def add_opt_argument_list(self, *args, options=None, tunnable=False, **kwargs):
        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(obj_id=arg_name,
                                         opt_values=options,
                                         tunnable=tunnable)

    def add_opt_argument_range(self, *args, start=None, end=None, nb_samples=10, tunnable=False, **kwargs):
        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(obj_id=arg_name,
                                         opt_values=[start, end],
                                         nb_samples=nb_samples,
                                         tunnable=tunnable)

    def add_json_config_argument(self, *args, **kwargs):
        self.add_argument(*args, **kwargs)
        self.json_config_arg_name = re.sub('-', '', args[-1])

    def parse_args(self, args=None, namespace=None):
        # call superclass arg first
        results = super(HyperOptArgumentParser, self).parse_args(args=args, namespace=namespace)

        # extract vals
        old_args = vars(results)

        # override with json args if given
        if self.json_config_arg_name and old_args[self.json_config_arg_name]:
            for arg, v in self.__read_json_config(old_args[self.json_config_arg_name]).items():
                old_args[arg] = v

        # track args
        self.parsed_args = deepcopy(old_args)

        # attach optimization fx
        old_args['trials'] = self.opt_trials
        old_args['optimize_parallel'] = self.optimize_parallel

        return argparse.Namespace(**old_args)

    def __read_json_config(self, file_path):
        with open(file_path) as json_data:
            json_args = json.load(json_data)
            return json_args

    def opt_trials(self, num):
        self.trials = strategies.generate_trials(strategy=self.strategy,
                                                 flat_params=self.__flatten_params(self.opt_args),
                                                 nb_trials=num)
        for trial in self.trials:
            ns = self.__namespace_from_trial(trial)
            yield ns

    def optimize_parallel(self, train_function, nb_trials, nb_parallel=4):
        self.trials = strategies.generate_trials(strategy=self.strategy,
                                                 flat_params=self.__flatten_params(self.opt_args),
                                                 nb_trials=nb_trials)

        # nb of runs through all parallel systems
        nb_fork_batches = int(math.ceil(len(self.trials) / nb_parallel))
        fork_batches = [self.trials[i: i + nb_parallel] for i in range(0, len(self.trials), nb_parallel)]
        for fork_batch in fork_batches:
            children = []

            # run n parallel forks
            for parallel_nb, trial in enumerate(fork_batch):

                # q up the trial and convert to a namespace
                ns = self.__namespace_from_trial(trial)

                # split new fork
                pid = os.fork()

                # when the process is a parent
                if pid:
                    children.append(pid)

                # when process is a child
                else:
                    # slight delay to make sure we don't overwrite over test tube log versions
                    sleep(parallel_nb * 0.5)
                    train_function(ns, parallel_nb)
                    os._exit(0)

            for i, child in enumerate(children):
                os.waitpid(child, 0)


    def __namespace_from_trial(self, trial):
        trial_dict = {d['name']: d['val'] for d in trial}
        for k, v in self.parsed_args.items():
            if k not in trial_dict:
                trial_dict[k] = v

        return argparse.Namespace(**trial_dict)


    def __flatten_params(self, params):
        """
        Turns a list of parameters with values into a flat tuple list of lists
        so we can permute
        :param params:
        :return:
        """
        flat_params = []
        for i, (opt_name, opt_arg) in enumerate(params.items()):
            if opt_arg.tunnable:
                clean_name = re.sub('-', '', opt_name)
                param_groups = []
                for val in opt_arg.opt_values:
                    param_groups.append({'idx': i, 'val': val, 'name': clean_name})
                flat_params.append(param_groups)
        return flat_params


class OptArg(object):

    def __init__(self, obj_id, opt_values, nb_samples=None, tunnable=False):
        self.opt_values = opt_values
        self.obj_id = obj_id
        self.nb_samples = nb_samples
        self.tunnable = tunnable

        # convert range to list of values
        if nb_samples:
            self.opt_values = np.linspace(opt_values[0], opt_values[1], num=nb_samples, endpoint=True)



