import torch
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster


"""
Example script to show how to run a hyperparameter search on a cluster managed by SLURM

A single trial gets allocated on a single GPU until all trials have completed.   
This means for 10 trials and 4 GPUs, we'll run 4 in parallel twice and the last 2 trials in parallel.   
"""


# main training function (very simple)
def train(hparams):
    # init exp and track all the parameters from the HyperOptArgumentParser
    exp = Experiment(
        name=hparams.test_tube_exp_name,
        save_dir=hparams.log_path,
        autosave=False,
    )
    exp.argparse(hparams)

    # pretend to train
    x = torch.rand((1, 5))
    for train_step in range(0, 100):
        y = torch.rand((5, 1))
        out = x.mm(y)
        exp.log({'fake_err': out.item()})

    # save exp when we're done
    exp.save()


# set up our argparser and make the y_val tunable
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--test_tube_exp_name', default='my_test')
parser.add_argument('--log_path', default='/Users/waf/Desktop/test')
parser.opt_list('--y_val', default=12, options=[1, 2, 3, 4, 5, 6], tunable=True)
parser.opt_list('--x_val', default=12, options=[20, 12, 30, 45], tunable=True)
hyperparams = parser.parse_args()

# enable cluster training
cluster = SlurmCluster(hyperparam_optimizer=hyperparams, log_path=hyperparams.log_path, test_tube_exp_name=hyperparams.test_tube_exp_name)

# email results if your hpc supports it
cluster.notify_job_status(email='waf251@nyu.edu', on_done=True, on_fail=True)

# any modules for code to run in env
cluster.load_modules(['python-3'])
cluster.add_command('source activate organics')

# set the environment variables
cluster.per_experiment_nb_gpus = 4
cluster.per_experiment_nb_nodes = 3
cluster.gpu_type = '1080ti'

# optimize on 4 gpus at the same time
cluster.optimize_parallel_cluster(train, nb_trials=24, job_name='test_job')
