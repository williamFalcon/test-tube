from exp_src.exp import Experiment


e = Experiment(debug=False, version=2)
e.add_meta_tag('loss', 'mse')
e.add_metric_row({'a': 23, 'b': 32})
