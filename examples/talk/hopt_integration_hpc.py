"""
Example on using the Experiment API as a stand-alone logger.
"""
import numpy as np
from test_tube import HyperOptArgumentParser
from test_tube import SlurmCluster

def fit(hparams):
    # ...
    pass


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