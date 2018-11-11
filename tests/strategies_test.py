import pytest

from test_tube.hyper_opt_utils import strategies

GRID_SEARCH = 'grid_search'
RANDOM_SEARCH = 'random_search'

FLAT_PARAMS = [
    [
        {'idx': 0, 'val': 0.0001, 'name': 'learning_rate'},
        {'idx': 1, 'val': 0.001, 'name': 'learning_rate'},
        {'idx': 2, 'val': 0.01, 'name': 'learning_rate'},
        {'idx': 3, 'val': 0.1, 'name': 'learning_rate'}
    ],
    [
        {'idx': 4, 'val': 0.99, 'name': 'decay'},
        {'idx': 5, 'val': 0.999, 'name': 'decay'},
    ]
]
def test_unknown_strategy():
    with pytest.raises(ValueError):
        strategies.generate_trials(
            'unknown_strategy', FLAT_PARAMS, nb_trials=None)

def test_grid_search_no_limit():
    trials = strategies.generate_trials(
        GRID_SEARCH, FLAT_PARAMS, nb_trials=None)
    assert len(trials) == len(FLAT_PARAMS[0]) * len(FLAT_PARAMS[1])

def test_grid_search_limit():
    trials = strategies.generate_trials(
        GRID_SEARCH, FLAT_PARAMS, nb_trials=5)
    assert len(trials) == 5


def test_random_search():
    trials = strategies.generate_trials(
        RANDOM_SEARCH, FLAT_PARAMS, nb_trials=5)
    assert len(trials) == 5

def test_random_search_unbounded_error():
    with pytest.raises(TypeError):
        trials = strategies.generate_trials(
            RANDOM_SEARCH, FLAT_PARAMS, nb_trials=None)

