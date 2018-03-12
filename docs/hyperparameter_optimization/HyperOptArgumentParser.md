# HyperOptArgumentParser class API

The HyperOptArgumentParser is a subclass of python's [argparse](https://docs.python.org/3/library/argparse.html), with added finctionality to change parameters on the fly as determined by a sampling strategy.

You can instantiate an `HyperOptArgumentParser` via:

```python
from test_tube import HyperOptArgumentParser

# subclass of argparse
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')

# let's enable optimizing over the number of layers in the network
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])

# and tune the number of units in each layer
parser.opt_range('--neurons', default=50, type=int, tunable=True, start=100, end=800, nb_samples=10)

# compile (because it's argparse underneath)
hparams = parser.parse_args()

# run 20 trials of random search over the hyperparams
for hparam_trial in hparams.trials(20):
    train_network(hparam_trial)
```
---
## init options

### strategy
Use either [random search](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) or [grid search](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) for tuning:
```python
parser = HyperOptArgumentParser(strategy='grid_search')
```

---
## Methods
All the functionality from argparse works but we've added the following functionality:

All the functionality from argparse works but we've added the following
functionality:

### `opt_list`

``` {.python}
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])
```
Enables searching over a list of values for this parameter. The tunable values ONLY replace the argparse values when running a hyperparameter optimization search. This is on purpose so your code doesn't have to change when you want to tune it.


**Example**

``` {.python}
parser.opt_list('--nb_layers', default=2, type=int, tunable=True, options=[2, 4, 8])
hparams = parser.parse_args()
# hparams.nb_layers = 2

for trial in hparams.trials(2):
    # trial.nb_layers is now a value in [2, 4, 8]
    # but hparams.nb_layers is still 2
```

### `opt_range`

``` {.python}
parser.opt_range('--neurons', default=50, type=int, tunable=True, start=100, end=800, nb_samples=8)
```

Enables searching over a range of values chosen linearly using the
`nb_samples` given. The tunable values ONLY replace the argparse values
when running a hyperparameter optimization search. This is on purpose so
your code doesn't have to change when you want to tune it.

**Example**

``` {.python}
parser.opt_range('--neurons', default=50, type=int, tunable=True, start=100, end=800, nb_samples=8)
hparams = parser.parse_args()
# hparams.neurons = 50

for trial in hparams.trials(2):
    # trial.nb_layers is now a value in [100, 200, 300, 400, 500, 600 700, 800]
    # but hparams.neurons is still 50
```

### `json_config`

``` {.python}
parser.json_config('--config', default='example.json')
```
Replaces default values in the parser with those read from the json file

**Example**
*example.json*
```json
{
    "learning_rate": 200
}
```

```python
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')
parser.json_config('--config', default='example.json')
hparams = parser.parse_args()

# hparams.learning_rate = 200
```

### trials
```python
trial_generator = hparams.trials(2)
```
Computes the trials needed for these experiments and serves them via a generator

**Example**

```python
hparams = parser.parse_args()
for trial_hparams in hparams.trials(2):
    # trial_hparams now has values sampled from the training routine
```

### optimize_parallel
`DEPRECATED... see optimize_parallel_gpu / _cpu`
```python
hparams = parser.parse_args()
hparams.optimize_parallel(function_to_optimize, nb_trials=20, nb_parallel=2)
```
Parallelize the trials across nb_parallel processes.
Arguments passed into the `function_to_optimize` are the `trial_params` and index of process it's in.

**Example**

```python
# parallelize tuning on 2 gpus
# this will place each trial in n into a given gpu
def opt_function(trial_params, process_index):
    GPUs = ['0', '1']
    os.environ["CUDA_VISIBLE_DEVICES"] = GPUs[process_index]
    train_main(trial_params)

hparams = parser.parse_args()
hparams.optimize_parallel(opt_function, nb_trials=20, nb_parallel=2)

# at the end of the optimize_parallel function, all 20 trials will be completed
# in this case by running 10 sets of 2 trials in parallel
```

### `optimize_parallel_gpu`
hparams = parser.parse_args()
hparams.optimize_parallel_gpu(function_to_optimize, gpu_ids=['1', '0, 2'], nb_trials=20, nb_workers=2)
```
Parallelize the trials across nb_workers processes. Auto assign the correct gpus.
Argument passed into the `function_to_optimize` is the `trial_params` argument.

**Example**

```python
# parallelize tuning on 2 gpus
# this will place each trial in n into a given gpu
def train_main(trial_params):
    # train your model, etc here...

hparams = parser.parse_args()
hparams.optimize_parallel_gpu(train_main, gpu_ids=['1', '0, 2'], nb_trials=20, nb_workers=2)

# at the end of the optimize_parallel function, all 20 trials will be completed
# in this case by running 10 sets of 2 trials in parallel
```

### optimize_parallel_cpu
```python
hparams = parser.parse_args()
hparams.optimize_parallel_cpu(function_to_optimize, nb_trials=20, nb_workers=2)
```
Parallelize the trials across nb_workers cpus.
Argument passed into the `function_to_optimize` is the `trial_params` argument.

**Example**

```python
# parallelize tuning on 2 cpus
# this will place each trial in n into a given gpu
def train_main(trial_params):
    # train your model, etc here...

hparams = parser.parse_args()
hparams.optimize_parallel_cpu(train_main, nb_trials=20, nb_workers=2)

# at the end of the optimize_parallel function, all 20 trials will be completed
# in this case by running 10 sets of 2 trials in parallel
```

