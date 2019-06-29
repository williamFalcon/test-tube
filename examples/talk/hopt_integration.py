"""
Example on using the Experiment API as a stand-alone logger.
"""
from test_tube import Experiment
import numpy as np
from argparse import ArgumentParser
from test_tube import HyperOptArgumentParser

def fit(hparams):
    # fake model
    n = 10
    in_features, out_features = 20, 1
    W = np.random.randn(in_features, out_features)
    b = np.random.randn(1, out_features)

    # linear + sigmoid
    model = lambda x, y: ((1 / (1 + np.exp(-np.matmul(x, W) + b)) - y)**2).sum()

    # fake some training
    for step_i in range(100):
        x, y = np.random.randn(n, in_features), np.random.randn(n)

        # log training error and an "image"
        tng_err = model(x, y)
        img = np.random.rand(36, 36)
        print(tng_err)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--hat_color', default='red', options=['red', 'orange', 'blue'])
    parser.opt_list('--prop', default='cup', options=['cup', 'shoe'])
    parser.opt_list('--mustache', default=True, options=[True, False], action='store_true', tunable=False)
    parser.opt_list('--hat', default=True, options=[True, False], action='store_true', tunable=False)

    args = parser.parse_args()

    # will run whatever is in the defaults
    fit(args)

    # will distribute 1 permutation on 1 gpu
    args.optimize_parallel_gpu(
        fit,
        gpu_ids=['0', '1', '2', '3'],
        nb_trials=24,
        nb_workers=4
    )