from argparse import ArgumentParser
import argparse
import numpy as np
import re
from copy import deepcopy
from .hyper_opt_utils import strategies


class HyperOptArgumentParser(ArgumentParser):

    def __init__(self, strategy='grid_search', enabled=True, experiment=None, **kwargs):
        ArgumentParser.__init__(self, **kwargs)

        self.strategy = strategy
        self.enabled = enabled
        self.experiment = experiment
        self.trials = []
        self.parsed_args = None
        self.opt_args = {}

    def add_argument(self, *args, **kwargs):
        super(HyperOptArgumentParser, self).add_argument(*args, **kwargs)

    def add_opt_argument_list(self, *args, opt_values=None, tunnable=False, **kwargs):
        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(obj_id=arg_name,
                                         opt_values=opt_values,
                                         tunnable=tunnable)

    def add_opt_argument_range(self, *args, start=None, end=None, nb_samples=10, tunnable=False, **kwargs):
        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(obj_id=arg_name,
                                         opt_values=[start, end],
                                         nb_samples=nb_samples,
                                         tunnable=tunnable)

    def parse_args(self, args=None, namespace=None):
        # call superclass arg first
        results = super(HyperOptArgumentParser, self).parse_args(args=args, namespace=namespace)

        # extract vals
        old_args = vars(results)
        self.parsed_args = deepcopy(old_args)

        # attach optimization fx
        old_args['trials'] = self.opt_trials

        return argparse.Namespace(**old_args)

    def opt_trials(self, num):
        self.trials = strategies.generate_trials(strategy=self.strategy,
                                                 flat_params=self.__flatten_params(self.opt_args),
                                                 nb_trials=num)
        for trial in self.trials:
            ns = self.__namespace_from_trial(trial)
            yield ns
        pass

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



