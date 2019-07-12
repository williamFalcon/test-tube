import argparse
import json
import math
import os
import random
import re
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from multiprocessing import Pool, Queue
from time import sleep
import numpy as np
from .hyper_opt_utils import strategies
from gettext import gettext as _

# needed to work with pytorch multiprocess
try:
    import torch
    import multiprocessing
    # multiprocessing.set_start_method('spawn', force=True)
except ModuleNotFoundError:
    pass


def optimize_parallel_gpu_private(args):
    trial_params, train_function = args[0], args[1]

    # get set of gpu ids
    gpu_id_set = g_gpu_id_q.get(block=True)

    try:

        # enable the proper gpus
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id_set

        # run training fx on the specific gpus
        results = train_function(trial_params)

        return [trial_params, results]

    except Exception as e:
        print('Caught exception in worker thread', e)

        # This prints the type, value, and stack trace of the
        # current exception being handled.
        traceback.print_exc()
        return [trial_params, None]

    finally:
        g_gpu_id_q.put(gpu_id_set)


def optimize_parallel_cpu_private(args):
    trial_params, train_function = args[0], args[1]

    sleep(random.randint(0, 4))

    # run training fx on the specific gpus
    results = train_function(trial_params)

    # True = completed
    return [trial_params, results]


class HyperOptArgumentParser(ArgumentParser):
    """
    Subclass of argparse ArgumentParser which adds optional calls to sample from lists or ranges
    Also enables running optimizations across parallel processes
    """

    # these are commands injected by test tube from cluster operations
    TRIGGER_CMD = 'test_tube_from_cluster_hopt'
    SLURM_CMD_PATH = 'test_tube_slurm_cmd_path'
    SLURM_EXP_CMD = 'hpc_exp_number'
    SLURM_LOAD_CMD = 'test_tube_do_checkpoint_load'
    CMD_MAP = {
        TRIGGER_CMD: bool,
        SLURM_CMD_PATH: str,
        SLURM_EXP_CMD: int,
        SLURM_LOAD_CMD: bool
    }

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
        self.pool = None

    def add_argument(self, *args, **kwargs):
        super(HyperOptArgumentParser, self).add_argument(*args, **kwargs)

    def opt_list(self, *args, **kwargs):
        options = kwargs.pop("options", None)
        tunable = kwargs.pop("tunable", False)
        self.add_argument(*args, **kwargs)
        for i in range(len(args)):
            arg_name = args[i]
            self.opt_args[arg_name] = OptArg(obj_id=arg_name, opt_values=options, tunable=tunable)

    def opt_range(
            self,
            *args,
            **kwargs
    ):
        low = kwargs.pop("low", None)
        high = kwargs.pop("high", None)
        arg_type = kwargs["type"]
        nb_samples = kwargs.pop("nb_samples", 10)
        tunable = kwargs.pop("tunable", False)
        log_base = kwargs.pop("log_base", None)

        self.add_argument(*args, **kwargs)
        arg_name = args[-1]
        self.opt_args[arg_name] = OptArg(
            obj_id=arg_name,
            opt_values=[low, high],
            arg_type=arg_type,
            nb_samples=nb_samples,
            tunable=tunable,
            log_base=log_base,
        )

    def json_config(self, *args, **kwargs):
        self.add_argument(*args, **kwargs)
        self.json_config_arg_name = re.sub('-', '', args[-1])

    def __parse_args(self, args=None, namespace=None):
        # allow bypassing certain missing params which other parts of test tube may introduce
        args, argv = self.parse_known_args(args, namespace)
        args, argv = self.__whitelist_cluster_commands(args, argv)
        if argv:
            msg = _('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
        return args

    def __whitelist_cluster_commands(self, args, argv):
        parsed = {}

        # build a dict where key = arg, value = value of the arg or None if just a flag
        for i, arg_candidate in enumerate(argv):
            arg = None
            value = None

            # only look at --keys
            if '--' not in arg_candidate:
                continue

            # skip items not on the white list
            if arg_candidate[2:] not in HyperOptArgumentParser.CMD_MAP:
                continue

            arg = arg_candidate[2:]
            # pull out the value of the argument if given
            if i + 1 <= len(argv) - 1:
                if '--' not in argv[i + 1]:
                    value = argv[i + 1]

                if arg is not None:
                    parsed[arg] = value
            else:
                if arg is not None:
                    parsed[arg] = value

        # add the whitelist cmds to the args
        all_values = set()
        for k, v in args.__dict__.items():
            all_values.add(k)
            all_values.add(v)

        for arg, v in parsed.items():
            v_parsed = self.__parse_primitive_arg_val(v)
            all_values.add(v)
            all_values.add(arg)
            args.__setattr__(arg, v_parsed)

        # make list with only the unknown args
        unk_args = []
        for arg in argv:
            arg_candidate = re.sub('--', '', arg)
            is_bool = arg_candidate == 'True' or arg_candidate == 'False'
            if is_bool: continue

            if arg_candidate not in all_values:
                unk_args.append(arg)

        # when no bad args are left, return none to be consistent with super api
        if len(unk_args) == 0:
            unk_args = None

        # add hpc_exp_number if not passed in so we can never get None
        if HyperOptArgumentParser.SLURM_EXP_CMD not in args:
            args.__setattr__(HyperOptArgumentParser.SLURM_EXP_CMD, None)

        return args, unk_args

    def __parse_primitive_arg_val(self, val):
        if val is None:
            return True
        try:
            return int(val)
        except ValueError:
            try:
                return float(val)
            except ValueError:
                return val

    def parse_args(self, args=None, namespace=None):
        # call superclass arg first
        results = self.__parse_args(args, namespace)

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
        old_args['optimize_parallel_gpu'] = self.optimize_parallel_gpu
        old_args['optimize_parallel_cpu'] = self.optimize_parallel_cpu
        old_args['generate_trials'] = self.generate_trials
        old_args['optimize_trials_parallel_gpu'] = self.optimize_trials_parallel_gpu

        return TTNamespace(**old_args)

    def __read_json_config(self, file_path):
        with open(file_path) as json_data:
            json_args = json.load(json_data)
            return json_args

    def opt_trials(self, num):
        self.trials = strategies.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=num,
        )

        for trial in self.trials:
            ns = self.__namespace_from_trial(trial)
            yield ns

    def generate_trials(self, nb_trials):
        trials = strategies.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials,
        )

        trials = [self.__namespace_from_trial(x) for x in trials]
        return trials

    def optimize_parallel_gpu(
            self,
            train_function,
            nb_trials,
            gpu_ids,
            nb_workers=4,
    ):
        """
        Runs optimization across gpus with cuda drivers
        :param train_function:
        :param nb_trials:
        :param gpu_ids: List of strings like: ['0', '1, 3']
        :param nb_workers:
        :return:
        """
        self.trials = strategies.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials,
        )

        self.trials = [(self.__namespace_from_trial(x), train_function) for x in self.trials]

        # build q of gpu ids so we can use them in each process
        # this is thread safe so each process can pull out a gpu id, run its task and put it back when done
        if self.pool is None:
            gpu_q = Queue()
            for gpu_id in gpu_ids:
                gpu_q.put(gpu_id)

            # called by the Pool when a process starts
            def init(local_gpu_q):
                global g_gpu_id_q
                g_gpu_id_q = local_gpu_q

            # init a pool with the nb of worker threads we want
            self.pool = Pool(processes=nb_workers, initializer=init, initargs=(gpu_q,))

        # apply parallelization
        results = self.pool.map(optimize_parallel_gpu_private, self.trials)
        return results

    def optimize_trials_parallel_gpu(
            self,
            train_function,
            nb_trials,
            trials,
            gpu_ids,
            nb_workers=4,
    ):
        """
        Runs optimization across gpus with cuda drivers
        :param train_function:
        :param nb_trials:
        :param gpu_ids: List of strings like: ['0', '1, 3']
        :param nb_workers:
        :return:
        """
        self.trials = trials
        self.trials = [(x, train_function) for x in self.trials]

        # build q of gpu ids so we can use them in each process
        # this is thread safe so each process can pull out a gpu id, run its task and put it back when done
        if self.pool is None:
            gpu_q = Queue()
            for gpu_id in gpu_ids:
                gpu_q.put(gpu_id)

            # called by the Pool when a process starts
            def init(local_gpu_q):
                global g_gpu_id_q
                g_gpu_id_q = local_gpu_q

            # init a pool with the nb of worker threads we want
            self.pool = Pool(processes=nb_workers, initializer=init, initargs=(gpu_q,))

        # apply parallelization
        results = self.pool.map(optimize_parallel_gpu_private, self.trials)
        return results

    def optimize_parallel_cpu(
            self,
            train_function,
            nb_trials,
            nb_workers=4,
    ):
        """
        Runs optimization across n cpus
        :param train_function:
        :param nb_trials:
        :param nb_workers:
        :return:
        """
        self.trials = strategies.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials
        )

        self.trials = [(self.__namespace_from_trial(x), train_function) for x in self.trials]

        # init a pool with the nb of worker threads we want
        if self.pool is None:
            self.pool = Pool(processes=nb_workers)

        # apply parallelization
        results = self.pool.map(optimize_parallel_cpu_private, self.trials)
        return results

    def optimize_parallel(
            self,
            train_function,
            nb_trials,
            nb_parallel=4,
    ):
        self.trials = strategies.generate_trials(
            strategy=self.strategy,
            flat_params=self.__flatten_params(self.opt_args),
            nb_trials=nb_trials
        )

        # nb of runs through all parallel systems
        fork_batches = [
            self.trials[i:i + nb_parallel] for i in range(0, len(self.trials), nb_parallel)
        ]

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

        return TTNamespace(**trial_dict)

    def __flatten_params(self, params):
        """
        Turns a list of parameters with values into a flat tuple list of lists
        so we can permute
        :param params:
        :return:
        """
        flat_params = []
        for i, (opt_name, opt_arg) in enumerate(params.items()):
            if opt_arg.tunable:
                clean_name = opt_name.strip('-')
                clean_name = re.sub('-', '_', clean_name)
                param_groups = []
                for val in opt_arg.opt_values:
                    param_groups.append({'idx': i, 'val': val, 'name': clean_name})
                flat_params.append(param_groups)
        return flat_params


class TTNamespace(argparse.Namespace):

    def __str__(self):
        result = '-' * 100 + '\nHyperparameters:\n'
        for k, v in self.__dict__.items():
            result += '{0:20}: {1}\n'.format(k, v)
        return result

    def __getstate__(self):
        # capture what is normally pickled
        state = self.__dict__.copy()

        # remove all functions from the namespace
        clean_state = {}
        for k, v in state.items():
            if not hasattr(v, '__call__'):
                clean_state[k] = v

        # what we return here will be stored in the pickle
        return clean_state

    def __setstate__(self, newstate):
        # re-instate our __dict__ state from the pickled state
        self.__dict__.update(newstate)


class OptArg(object):
    def __init__(
            self,
            obj_id,
            opt_values,
            arg_type=None,
            nb_samples=None,
            tunable=False,
            log_base=None,
    ):
        self.opt_values = opt_values
        self.obj_id = obj_id
        self.tunable = tunable

        # convert range to list of values
        if nb_samples:
            low, high = opt_values

            if log_base is None:
                # random search on uniform scale
                if arg_type is int:
                    self.opt_values = np.random.choice(np.arange(low, high), nb_samples, replace=False)
                elif arg_type is float:
                    self.opt_values = np.random.uniform(low, high, nb_samples)
            else:
                # random search on log scale with specified base
                assert high >= low > 0, "`opt_values` must be positive to do log-scale search."

                log_low, log_high = math.log(low, log_base), math.log(high, log_base)

                self.opt_values = log_base ** np.random.uniform(log_low, log_high, nb_samples)

