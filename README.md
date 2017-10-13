# Test tube
<div style="text-align: center">
<img src="https://raw.githubusercontent.com/williamfalcon/test_tube/master/imgs/test_tube_logo.png">
</div>
<br>    

[![PyPI version](https://badge.fury.io/py/test_tube.svg)](https://badge.fury.io/py/test_tube)    [![Doc status](https://readthedocs.org/projects/pip/badge/?version=latest)](https://readthedocs.org/projects/pip/badge/?version=latest)     [![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/williamFalcon/test_tube/blob/master/LICENSE)


Log and optimize Deep Learning experiments with ease. 

Nothing is uploaded, it's all done on your machine.   

```bash
pip install test_tube
```   

[DOCUMENTATION](https://williamfalcon.github.io/test_tube/)    

![Screenshot](https://github.com/williamFalcon/test_tube/raw/master/imgs/viz_a.png)
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

### Optimize hyperparameters
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

## Visualizing and accessing your data   

[Example test_tube output](https://github.com/williamFalcon/test_tube/tree/master/examples/saved_logs/example_test_tube_data)    
Test tube saves each experiment in a folder structure like:   
```
test_tube_data
    /my_exp_name_A
        /version_0
        /version_1
            meta.experiment
            meta_tags.json
            metrics.csv
            media/
                test_img_0.jpg
                test_img_1.jpg
         /version_2   
         
    /my_exp_name_B
        ...
```    

#### Media files    

Any saved images are saved to `/media`:   

**example**   
```python
img = np.imread('a.jpg')
exp.add_metric_row('test_jpg': img, 'val_err': 0.2)    

# saves image to /media/test_0.jpg
```   
     
     
#### Metrics file
metrics.csv is a standard csv with rows for each `add_metric_row` call and columns for each key across all rows   

**example**   
```python
exp.add_metric_row('val_err': 0.2, 'epoch': 1)    
exp.add_metric_row('test_err': 0.2, 'epoch': 1)    
```    

`metrics.csv`    
*created_at is added automatically*     

|  val_err |  epoch | test_err | created_at |
|---|---|---| --- |
| 0.2  |  1 | - | 2017-10-13 01:34:14 |
| -  | 1  | 0.2| 2017-10-13 01:34:18 |


#### Meta tags file 
`meta_tags.json` contains a json file with the information for your experiment    

**example**   
```python
exp.add_meta_tags({'learning_rate': 0.002, 'nb_layers': 2})
```    

`meta_tags.json`   
```json
{
    "learning_rate": 0.002,
    "nb_layers": 2
}
``` 
