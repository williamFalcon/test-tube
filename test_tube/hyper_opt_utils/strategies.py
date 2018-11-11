"""Hyperparameter search strategies."""
import itertools
import json
import random


def generate_trials(strategy, flat_params, nb_trials=None):
    r"""Generates the parameter combinations to search.

    Two search strategies are implemented:
    1. `grid_search`: creates a search space that consists of the
        product of all flat_params. If `nb_trials` is specified
        the first `nb_trials` combinations are searched.
    2. `random_search`: Creates random combinations of the
        hyperparameters. Can be used for a more efficient search.
        See (Bergstra and Bengio, 2012) for more details.

    :param strategy: The hyperparameter search to strategy. Can be
        one of: {`grid_search`, `random`}.
    :param flat_params: The hyperparameter arguments to iterate over.
    :param nb_trials: The number of hyperparameter combinations to try.
    Generates the parameter combinations for each requested trial
    :param strategy:
    :param flat_params:
    :param nb_trials: The number of trials to un.
    :return:
    """
    if strategy == 'grid_search':
        trials = generate_grid_search_trials(flat_params, nb_trials)
        return trials
    elif strategy == 'random_search':
        trials = generate_random_search_trials(flat_params, nb_trials)
        return trials
    else:
        raise ValueError(
            ('Unknown strategy "{}". Must be one of '
             '{{grid_search, random_search}}').format(strategy))


def generate_grid_search_trials(flat_params, nb_trials):
    """
    Standard grid search. Takes the product of `flat_params`
    to generate the search space.

    :param params: The hyperparameters options to search.
    :param nb_trials: Returns the first `nb_trials` from the
        combinations space. If this is None, all combinations
        are returned.
    :return: A dict containing the hyperparameters.
    """
    trials = list(itertools.product(*flat_params))
    if nb_trials:
        trials = trials[0:nb_trials]
    return trials


def generate_random_search_trials(params, nb_trials):
    """
    Generates random combination of hyperparameters to try.
    See (Bergstra and Bengio, 2012) for more details.

    :param params: The hyperparameters options to search.
    :param nb_trials: The number of trials to run.
    :return: A dict containing the hyperparameters.
    """
    if nb_trials is None:
        raise TypeError(
            '`random_search` strategy requires nb_trails to be an int.')
    results = []

    # ensures we have unique results
    seen_trials = set()

    # shuffle each param list
    potential_trials = 1
    for param in params:
        random.shuffle(param)
        potential_trials *= len(param)

    # we can't sample more trials than are possible
    max_iters = min(potential_trials, nb_trials)

    # then for the nb of trials requested, create a new param tuple
    # by picking a random integer at each param level
    while len(results) < max_iters:
        trial = []
        for param in params:
            sampled_param = random.sample(param, 1)[0]
            trial.append(sampled_param)

        # verify this is a unique trial so we
        # don't duplicate work
        trial_str = json.dumps(trial)
        if trial_str not in seen_trials:
            seen_trials.add(trial_str)
            results.append(trial)

    return results
