# Test tube
<div style="text-align: center">
<img src="https://raw.githubusercontent.com/williamfalcon/test_tube/master/imgs/test_tube_logo.png">
</div>
<br>    

[![PyPI version](https://badge.fury.io/py/test_tube.svg)](https://badge.fury.io/py/test_tube)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/williamFalcon/test_tube/blob/master/LICENSE)


Log machine learning / data science experiments with ease

```bash
pip install test_tube
```   


## Example use:

```python
from test_tube.log import Experiment

# make new experiment
# each param is optional
# name defaults to 'default'.
# if no version it always increases +1 by default
# if you set the version it will make changes to that version

e = Experiment(name='model a', version=1)

# add a column
e.add_meta_tag('loss', 'mse')

# add a row of metrics
e.add_metric_row({'mse': 23, 'mae': 12, 'epoch': 1})

```
When debugging you can keep it from saving anything
```python
e = Experiment(debug=True)

# add a column
e.add_meta_tag('loss', 'mse')
# no effect...
```

## API
