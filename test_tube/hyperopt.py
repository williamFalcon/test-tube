import itertools
import random
import json


class HyperParamOptimizer(object):
    method = None
    enabled = None

    # the params to use at each trial
    trials = None

    # total iterations we're doing
    nb_iterations = None

    # details about each param
    params = []

    current_iteration = 0

    seen_params = {}

    def __init__(self, method='grid_search', enabled=True):
        """
        :param method: 'grid_search', 'random_search'
        :param enabled:
        """
        self.method = method
        self.enabled = enabled

    # -----------------------------
    # PARAMETER CHOICES
    # -----------------------------
    def tune_uniform(self, low, high, samples, default, name):
        # how this fx samples for the data
        def gen_samples():
            vals = [random.uniform(low, high) for i in range(samples)]
            return vals

        return self.__resolve_param(gen_samples, default, name)

    def tune_choice(self, options, default, name):
        def gen_samples():
            return options

        return self.__resolve_param(gen_samples, default, name)

    def __resolve_param(self, gen_fx, default, name):
        # case when no action was requested
        if not self.enabled:
            return default

        # create the param when it's new
        # return the first value in this case
        if name not in self.seen_params:
            vals = gen_fx()
            param = {'vals': vals, 'name': name}
            self.seen_params[name] = {'idx': len(self.params)}
            self.params.append(param)
            return vals[0]

        # not the first iteration so return the ith element
        # in the possible values
        iteration_params = self.trials[self.current_iteration]
        param_i = self.seen_params[name]['idx']
        param = iteration_params[param_i]
        return param['val']
    # -----------------------------
    # RUN OPTIMIZATION
    # -----------------------------
    def optimize(self, fx, nb_iterations=None):
        """
        Primary entry point into the optimization
        :param fx:
        :param nb_iterations:
        :return:
        """
        self.nb_iterations = nb_iterations

        results = []

        # run first iteration
        result = fx(self)
        results.append(result)
        self.current_iteration += 1

        # generate the rest of the training seq
        # we couldn't do this before because we don't know
        # how many params the user needed
        self.__generate_trials()

        # run trials for the rest of the iterations
        # we either know the iterations or they're
        # calculated from the strategy used
        for i in range(1, len(self.trials)):
            result = fx(self)
            results.append(result)
            self.current_iteration += 1

    # -----------------------------
    # TRIALS HELPER
    # -----------------------------
    def __generate_trials(self):
        """
        Generates the parameter combinations for each requested trial
        :return:
        """
        flat_params = self.__flatten_params(self.params)

        # permute for grid search
        if self.method == 'grid_search':
            self.trials = list(itertools.product(*flat_params))

            if self.nb_iterations is not None:
                self.trials = self.trials[0: self.nb_iterations]

        if self.method == 'random_search':
            self.trials = self.__generate_random_search_trials(flat_params)

    def __flatten_params(self, params):
        """
        Turns a list of parameters with values into a flat tuple list of lists
        so we can permute
        :param params:
        :return:
        """
        flat_params = []
        for i, param in enumerate(params):
            param_groups = []
            for val in param['vals']:
                param_groups.append({'idx': i, 'val': val})
            flat_params.append(param_groups)
        return flat_params

    def __generate_random_search_trials(self, params):
        results = []

        # ensures we have unique results
        seen_trials = set()

        # shuffle each param list
        potential_trials = 1
        for p in params:
            random.shuffle(p)
            potential_trials *= len(p)

        # we can't sample more trials than are possible
        max_iters = min(potential_trials, self.nb_iterations)

        # then for the nb of trials requested, create a new param tuple
        # by picking a random integer at each param level
        while len(results) < max_iters:
            trial = []
            for param in params:
                p = random.sample(param, 1)[0]
                trial.append(p)

            # verify this is a unique trial so we
            # don't duplicate work
            trial_str = json.dumps(trial)
            if trial_str not in seen_trials:
                seen_trials.add(trial_str)
                results.append(trial)

        return results
