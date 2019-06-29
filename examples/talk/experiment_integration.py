"""
Example on using the Experiment API as a stand-alone logger.
"""
from test_tube import Experiment
import numpy as np

# 2. integrate experiment logger
def fit():
    # -------------------------
    # INIT EXPERIMENT
    exp = Experiment(name='dense_model',
                     debug=False,
                     save_dir='/Users/williamfalcon/Desktop/test_tube')

    # add a tag, such as learning rate
    exp.tag({'learning_rate': 0.002, 'nb_layers': 1})
    # -------------------------

    # fake model
    n = 10
    in_features, out_features = 20, 1
    W = np.random.randn(in_features, out_features)
    b = np.random.randn(1, out_features)

    # linear + sigmoid
    model = lambda x, y: ((1 / (1 + np.exp(-np.matmul(x, W) + b)) - y)**2).sum()

    # fake some training
    for step_i in range(100):
        x, y = np.random.randn(n, in_features), np.random.randn(n)

        # log training error and an "image"
        tng_err = model(x, y)
        img = np.random.rand(36, 36)

        # -------------------------------------
        # LOG (auto-saves)
        exp.log({'tng_err': tng_err, 'example_jpg': img})
        # -------------------------------------


if __name__ == '__main__':
    fit()