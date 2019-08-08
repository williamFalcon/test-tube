"""Example launcher for a hyperparameter search on SLURM.

This example shows how to use gpus on SLURM with PyTorch.
"""
import torch
from test_tube import Experiment, HyperOptArgumentParser, SlurmCluster

def train(hparams, *args):
    """Train your awesome model.

    :param hparams: The arguments to run the model with.
    """
    # Initialize experiments and track all the hyperparameters
    exp = Experiment(
        name=hparams.test_tube_exp_name,
        # Location to save the metrics.
        save_dir=hparams.log_path,
        autosave=False,
    )
    exp.argparse(hparams)

    # Pretend to train.
    x = torch.rand((1, hparams.x_val))
    for train_step in range(0, 100):
        y = torch.rand((hparams.x_val, 1))
        out = x.mm(y)
        exp.log({'fake_err': out.item()})

    # Save exp when .
    exp.save()


if __name__ == '__main__':
    # Set up our argparser and make the y_val tunable.
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--test_tube_exp_name', default='my_test')
    parser.add_argument('--log_path', default='/some/path/to/log')
    parser.opt_list('--y_val',
        default=12, options=[1, 2, 3, 4, 5, 6], tunable=True)
    parser.opt_list('--x_val',
        default=12, options=[20, 12, 30, 45], tunable=True)
    hyperparams = parser.parse_args()

    # Enable cluster training.
    cluster = SlurmCluster(
        hyperparam_optimizer=hyperparams,
        log_path=hyperparams.log_path,
        python_cmd='python3',
        test_tube_exp_name=hyperparams.test_tube_exp_name
    )

    # Email results if your hpc supports it.
    cluster.notify_job_status(
        email='some@email.com', on_done=True, on_fail=True)

    # SLURM Module to load.
    cluster.load_modules([
        'python-3',
        'anaconda3'
    ])

    # Add commands to the non-SLURM portion.
    cluster.add_command('source activate myCondaEnv')

    # Add custom SLURM commands which show up as:
    # #comment
    # #SBATCH --cmd=value
    # ############
    # cluster.add_slurm_cmd(
    #    cmd='cpus-per-task', value='1', comment='CPUS per task.')

    # Set job compute details (this will apply PER set of hyperparameters.)
    cluster.per_experiment_nb_gpus = 4
    cluster.per_experiment_nb_nodes = 2
    cluster.gpu_type = '1080ti'

    # Each hyperparameter combination will use 8 gpus.
    cluster.optimize_parallel_cluster_gpu(
        # Function to execute:
        train,
        # Number of hyperparameter combinations to search:
        nb_trials=24,
        # This is what will display in the slurm queue:
        job_name='first_tt_job')
