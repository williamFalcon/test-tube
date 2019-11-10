# Test Tube: Easily log and tune Deep Learning experiments

Test Tube allows you to easily log metadata and track your machine
learning experiments.

Use Test Tube if you need to:

-   Track many [Experiments](experiment_tracking/experiment.md) across
    models.
-   Visualize and compare different
    experiments without uploading anywhere.
-   [Optimize your
    hyperparameters](hyperparameter_optimization/HyperOptArgumentParser/)
    using grid search or random search.
-   Automatically track ALL parameters for a particular training run.

Test Tube is compatible with: Python 2 and 3

## Getting started

------------------------------------------------------------------------

### Create an [Experiment](experiment_tracking/experiment.md)

``` {.python}
from test_tube import Experiment

exp = Experiment(name='dense_model',
                 debug=False,
                 save_dir='/Desktop/test_tube')

exp.tag({'learning_rate': 0.002, 'nb_layers': 2})

for step in training_steps:
    tng_err = model.eval(tng_x, tng_y)

    exp.log('tng_err': tng_err)

# training complete!
# all your logs and data are ready to be visualized at testtube.williamfalcon.com
```

------------------------------------------------------------------------

### Optimize your [hyperparameters](hyperparameter_optimization/HyperOptArgumentParser/)

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

# run 20 trials of random search over the hyperparams
for hparam_trial in hparams.trials(20):
    train_network(hparam_trial)
```

------------------------------------------------------------------------

### Visualize

``` {.python}
import pandas as pd
import matplotlib

# each experiment is saved to a metrics.csv file which can be imported anywhere
# images save to exp/version/images
df = pd.read_csv('../some/dir/test_tube_data/dense_model/version_0/metrics.csv')
df.tng_err.plot()
```
