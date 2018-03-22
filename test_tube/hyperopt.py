import itertools
import random
import json

class HyperParamOptimizer(object):

    def __init__(self, method='grid_search', enabled=True, experiment=None):
        """
        :param method: 'grid_search', 'random_search'
        :param enabled:
        """
        self.method = method
        self.enabled = enabled
        self.experiment = experiment
        self.seen_params = {}
        self.current_iteration = 0

        # the params to use at each trial
        self.trials = None

        # total iterations we're doing
        self.nb_iterations = None

        # details about each param
        self.params = []

    # -----------------------------
    # PARAMETER CHOICES
    # -----------------------------
    def tune_uniform(self, low, high, samples, default, name):
        # how this fx samples for the data
        def gen_samples():
            vals = [random.uniform(low, high) for i in range(samples)]
            return vals

        return self.__resolve_param(gen_samples, default, name)

    def tune_odds(self, low, high, default, name):
        start = low if low %2 != 0 else low + 1
        def gen_samples():
            return range(start, high+1, 2)

        return self.__resolve_param(gen_samples, default, name)

    def tune_evens(self, low, high, default, name):
        start = low if low %2 == 0 else low + 1
        def gen_samples():
            return range(start, high+1, 2)

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
    # OPTIMIZATION
    # -----------------------------
    def optimize(self, fx, nb_iterations=None):
        """
        Primary entry point into the optimization
        :param fx:
        :param nb_iterations:
        :return:
        """
        self.nb_iterations = nb_iterations

        # run first iteration
        result = fx(self)

        # log if requested
        if self.experiment is not None:
            result['hypo_iter_nb'] = self.current_iteration
            self.experiment.log(result)

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
            result['hypo_iter_nb'] = self.current_iteration

            # log if requested
            if self.experiment is not None:
                self.experiment.log(result)

            self.current_iteration += 1

    # -----------------------------
    # INTERFACE WITH LOGGER
    # -----------------------------
    def get_current_trial_meta(self):
        meta_results = []

        # when we have trials, means we've already done 1 run
        # we can just get the params that are about to be run
        # otherwise we need to infer params from the current param list
        # this assumes the user feeds the opt into the experiment after
        # they're done setting up the params
        is_first_trial = self.trials is not None and len(self.trials) > 0
        if is_first_trial:
            trial_params = self.trials[self.current_iteration]
            for trial_param in trial_params:
                root_param = self.params[trial_param['idx']]
                meta_results.append({'hypo_' + root_param['name']: trial_param['val']})

        # if we haven't done a pass through the data yet,
        # we need to infer from the params in the list
        else:
            for param in self.params:
                meta_results.append({'hypo_' + param['name']: param['vals'][0]})

        # add shared meta
        meta_results.append({'hypo_iter_nb': self.current_iteration})
        return meta_results

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
