<p align="center">
  <a href="https://williamfalcon.github.io/test-tube/">
    <img alt="react-router" src="https://raw.githubusercontent.com/williamfalcon/test-tube/master/imgs/test_tube_logo.png" width="50">
  </a>
</p>
<h3 align="center">
  Test Tube
</h3>
<p align="center">
  Log, organize and parallelize hyperparameter search for Deep Learning experiments
</p>
<p align="center">
  <a href="https://badge.fury.io/py/test-tube"><img src="https://badge.fury.io/py/test-tube.svg" alt="PyPI version" height="18"></a>
  <a href="https://travis-ci.org/williamFalcon/test-tube"><img src="https://travis-ci.org/williamFalcon/test-tube.svg?branch=master"></a>
  <a href="https://williamfalcon.github.io/test-tube/"><img src="https://readthedocs.org/projects/test-tube/badge/?version=latest"></a>
  <a href="https://github.com/williamFalcon/test-tube/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>   

## Docs

**[View the docs here](https://williamfalcon.github.io/test-tube/)**

---   

Test tube is a python library to track and parallelize hyperparameter
search for Deep Learning and ML experiments. It's framework agnostic and
built on top of the python argparse API for ease of use.

``` {.bash}
pip install test_tube
```

---   

### Main test-tube uses

-   [Parallelize hyperparameter
    optimization](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/)
    (across multiple gpus or cpus).
-   [Parallelize hyperparameter
    optimization](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/)
    across HPC cluster using SLURM.   
-   Log experiment hyperparameters and experiment data.   
    [Experiments](https://williamfalcon.github.io/test-tube/experiment_tracking/experiment/)
    across models.
-   Visualize with [tensorboardx](https://github.com/lanpa/tensorboardX)

Compatible with Python any Python ML library like Tensorflow, Keras, Pytorch, Caffe, Caffe2, Chainer, MXNet, Theano, Scikit-learn   

---   
### Examples   
The Experiment object is a subclass of tensorboardx.SummaryWriter.   

**Log and visualize with TensorboardX**     

```{.python}
from test-tube import Experiment
import torch

exp = Experiment('/some/path')
exp.tag({'learning_rate': 0.02, 'layers': 4})    

# exp is superclass of SummaryWriter
features = torch.Tensor(100, 784)
writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

# simulate training
for n_iter in range(2000):
    e.log({'testtt': n_iter * np.sin(n_iter)})

# save and close
exp.save()
exp.close()
```

```{.bash}
pip install tensorflow   

tensorboard --logdir /some/path
``` 
    
**Run grid search on SLURM GPU cluster**    

``` {.python}   
from test_tube.hpc import SlurmCluster

# hyperparameters is a test-tube hyper params object
hyperparams = args.parse()

# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/path/to/log/results/to',
    python_cmd='python3'
)

# let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1

# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(train, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')   

# we just ran 20 different hyperparameters on 20 GPUs in the HPC cluster!!    
```    

**Optimize hyperparameters across GPUs**

``` {.python}
from test_tube import HyperOptArgumentParser

# subclass of argparse
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')

# let's enable optimizing over the number of layers in the network
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])

# and tune the number of units in each layer
parser.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=10)

# compile (because it's argparse underneath)
hparams = parser.parse_args()

# optimize across 4 gpus
# use 2 gpus together and the other two separately
hparams.optimize_parallel_gpu(MyModel.fit, gpu_ids=['1', '2,3', '0'], nb_trials=192, nb_workers=4)
```

Or... across CPUs

``` {.python}
hparams.optimize_parallel_cpu(MyModel.fit, nb_trials=192, nb_workers=12)
```

You can also optimize on a *log* scale to allow better search over
magnitudes of hyperparameter values, with a chosen base (disabled by
default). Keep in mind that the range you search over must be strictly
positive.

``` {.python}
from test_tube import HyperOptArgumentParser

# subclass of argparse
parser = HyperOptArgumentParser(strategy='random_search')

# Randomly searches over the (log-transformed) range [100,800).

parser.opt_range('--neurons', default=50, type=int, tunable=True, low=100, high=800, nb_samples=10, log_base=10)


# compile (because it's argparse underneath)
hparams = parser.parse_args()

# run 20 trials of random search over the hyperparams
for hparam_trial in hparams.trials(20):
    train_network(hparam_trial)
```

### Convert your argparse params into searchable params by changing 1 line

``` {.python}
import argparse
from test_tube import HyperOptArgumentParser

# these lines are equivalent
parser = argparse.ArgumentParser(description='Process some integers.')
parser = HyperOptArgumentParser(description='Process some integers.', strategy='grid_search')

# do normal argparse stuff
...
```

### Log images inline with metrics

``` {.python}
# name must have either jpg, png or jpeg in it
img = np.imread('a.jpg')
exp.log('test_jpg': img, 'val_err': 0.2)

# saves image to ../exp/version/media/test_0.jpg
# csv has file path to that image in that cell
```

## Demos

-   [Hyperparameter optimization for PyTorch across 20 cluster GPUs](https://github.com/williamFalcon/test-tube/blob/master/examples/pytorch_hpc_example.py)   
-   [Hyperparameter optimization across 20 cluster CPUs](https://github.com/williamFalcon/test-tube/blob/master/examples/hpc_cpu_example.py)   
-   [Experiments and hyperparameter optimization for tensorflow across 4 GPUs simultaneously](https://github.com/williamFalcon/test-tube/blob/master/examples/tensorflow_example.py)

## How to contribute

Feel free to fix bugs and make improvements! 1. Check out the [current
bugs here](https://github.com/williamFalcon/test-tube/issues) or
[feature
requests](https://github.com/williamFalcon/test-tube/projects/1). 2. To
work on a bug or feature, head over to our [project
page](https://github.com/williamFalcon/test-tube/projects/1) and assign
yourself the bug. 3. We'll add contributor names periodically as people
improve the library!

## Bibtex

To cite the framework use:

    @misc{Falcon2017,
      author = {Falcon, W.A.},
      title = {Test Tube},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/williamfalcon/test-tube}}
    }    
    
 ## License    
 In addition to the terms outlined in the license, this software is U.S. Patent Pending.    
