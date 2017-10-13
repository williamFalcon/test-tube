import tensorflow as tf
import os
from test_tube import Experiment, HyperOptArgumentParser

"""
This script demonstrates how to do a hyperparameter search over 1 parameter in tensorflow
on 4 simultaneous GPUs. Each trial will also save its own experiment logs
"""


# main training function (very simple)
def train(hparams):
    # init exp and track all the parameters from the HyperOptArgumentParser
    exp = Experiment(name='dense_model', save_dir='../Desktop/test_tube', autosave=False)
    exp.add_argparse_meta(hparams)

    # define tensorflow graph
    x = tf.placeholder(dtype=tf.int32, name='x')
    y = tf.placeholder(dtype=tf.int32, name='y')
    out = x * y

    sess = tf.Session()

    # Run the tf op
    for train_step in range(0, 100):
        print(sess.run(out, feed_dict={x: 2, y: hparams.y_val}))

    # save exp when we're done
    exp.save()


# build a wrapper around a tng function so we can use the correct gpu
# the optimizer passes in the hyperparams and the job index as arguments
# to the function to optimize
def parallelize_on_gpus(trial_params, job_index_nb):
    gpu_nb = str(job_index_nb)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_nb
    train(trial_params)


# set up our argparser and make the y_val tunnable
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--path', default='some/path')
parser.add_opt_argument_list('--y_val', default=12, options=[1, 2, 3, 4], tunnable=True)

hyperparams = parser.parse_args()

# optimize on 4 gpus at the same time
# each gpu will get 1 experiment with a set of hyperparams
hyperparams.optimize_parallel(parallelize_on_gpus, nb_trials=4, nb_parallel=4)
