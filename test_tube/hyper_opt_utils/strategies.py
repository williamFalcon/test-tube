import itertools
import json
import random


def generate_trials(strategy, flat_params, nb_trials=None):
    """
    Generates the parameter combinations for each requested trial
    :return:
    """
    # permute for grid search
    if strategy == 'grid_search':
        trials = generate_grid_search_trials(flat_params, nb_trials)
        return trials

    # generate random search
    if strategy == 'random_search':
        trials = generate_random_search_trials(flat_params, nb_trials)
        return trials


def generate_grid_search_trials(flat_params, nb_trials):
    trials = list(itertools.product(*flat_params))
    if nb_trials:
        trials = trials[0:nb_trials]
    return trials


def generate_random_search_trials(params, nb_trials):
    results = []

    # ensures we have unique results
    seen_trials = set()

    # shuffle each param list
    potential_trials = 1
    for p in params:
        random.shuffle(p)
        potential_trials *= len(p)

    # we can't sample more trials than are possible
    max_iters = min(potential_trials, nb_trials)

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
