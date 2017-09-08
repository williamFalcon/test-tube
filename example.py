from test_tube.log import Experiment

# make new experiment
e = Experiment(autosave=False)

# add a column
e.add_meta_tag('loss', 'mse')
e.add_meta_tags({'a': 2, 'sd': 'as'})

# add a row of metrics
for i in range(1000):
    e.add_metric_row({'mse': i, 'mae': 12, 'epoch': 1})
e.save()
