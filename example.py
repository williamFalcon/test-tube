from test_tube.log import Experiment
import numpy as np
from scipy.misc import imread

# make new experiment
e = Experiment(autosave=False, create_git_tag=True)

img = imread('/Users/waf/Desktop/a.png')

# add a column
e.add_meta_tag('loss', 'mse')
e.add_meta_tags({'a': 2, 'sd': 'as'})

# add a row of metrics
for i in range(1000):
    e.add_metric_row({'mse': i, 'mae': 12, 'epoch': 32, 'png_testwill': img})
e.save()
