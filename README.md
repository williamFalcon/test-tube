# Test tube
<div style="text-align: center">
<img src="https://raw.githubusercontent.com/williamfalcon/test_tube/master/imgs/test_tube_logo.png">
</div>
<br>    

[![PyPI version](https://badge.fury.io/py/test_tube.svg)](https://badge.fury.io/py/test_tube)    [![Doc status](https://readthedocs.org/projects/pip/badge/?version=latest)](https://readthedocs.org/projects/pip/badge/?version=latest)     [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/williamFalcon/test_tube/blob/master/LICENSE)


Log and optimize Deep Learning experiments with ease. 

```bash
pip install test_tube
```   

[DOCUMENTATION](https://williamfalcon.github.io/test_tube/)    

## Example use:

### Log experiments   

```python
from test_tube import Experiment

exp = Experiment(name='dense_model', save_dir='/Desktop/test_tube')

exp.add_meta_tags({'learning_rate': 0.002, 'nb_layers': 2})

for step in training_steps:
    tng_err = model.eval(tng_x, tng_y)
    
    exp.add_metric_row('tng_err': tng_err)

# training complete!
# all your logs and data are ready to be visualized at testtube.williamfalcon.com

```    

### Optimize your hyperparameters
```python
from test_tube import HyperOptArgumentParser

# subclass of argparse
parser = HyperOptArgumentParser(strategy='random_search')
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')

# let's enable optimizing over the number of layers in the network 
parser.add_opt_argument_list('--nb_layers', default=2, type=int, tunnable=True, options=[2, 4, 8])

# and tune the number of units in each layer
parser.add_opt_argument_range('--neurons', default=50, type=int, tunnable=True, start=100, end=800, nb_samples=10)

# compile (because it's argparse underneath)
hparams = parser.parse_args()   

# run 20 trials of random search over the hyperparams
for hparam_trial in hparams.trials(20):
    train_network(hparam_trial)
```     

[FULL DOCS](https://williamfalcon.github.io/test_tube/)
