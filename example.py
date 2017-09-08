from test_tube.log import Experiment

# make new experiment
e = Experiment()

# add a column
e.add_meta_tag('loss', 'mse')

# add a row of metrics
e.add_metric_row({'mse': 23, 'mae': 12, 'epoch': 1})