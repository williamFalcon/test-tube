from log.exp import Experiment


e = Experiment(version=1, save_dir='/Users/waf/Desktop')
e.add_meta_tag('loss', 'msd')
e.add_metric_row({'a': 23, 'b': 442})
