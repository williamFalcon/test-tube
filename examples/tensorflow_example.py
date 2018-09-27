import tensorflow as tf
from test_tube import Experiment, HyperOptArgumentParser


"""
This script demonstrates how to do a hyperparameter search over 2 parameters in tensorflow
on 4 simultaneous GPUs. Each trial will also save its own experiment logs.   

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

    # define tensorflow graph
    x = tf.placeholder(dtype=tf.int32, name='x')
    y = tf.placeholder(dtype=tf.int32, name='y')
    out = x * y

    sess = tf.Session()

    # Run the tf op
    for train_step in range(0, 100):
        output = sess.run(out, feed_dict={x: hparams.x_val, y: hparams.y_val})
        exp.log({'fake_err': output})

    # save exp when we're done
    exp.save()


# set up our argparser and make the y_val tunable
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--test_tube_exp_name', default='my_test')
parser.add_argument('--log_path', default='/Users/waf/Desktop/test')
parser.opt_list('--y_val', default=12, options=[1, 2, 3, 4], tunable=True)
parser.opt_list('--x_val', default=12, options=[20, 12, 30, 45], tunable=True)
hyperparams = parser.parse_args()


# optimize on 4 gpus at the same time
# each gpu will get 1 experiment with a set of hyperparams
hyperparams.optimize_parallel_gpu(train, gpu_ids=['1', '0', '3', '2'], nb_trials=4, nb_workers=4)
