# Test_tube
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
