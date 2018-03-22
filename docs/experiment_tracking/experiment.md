# Experiment class API

An Experiment holds metadata and the results of the training run, you
can instantiate an `Experiment` via:

``` {.python}
from test_tube import Experiment

exp = Experiment(name='dense_model',
                 debug=False,
                 save_dir='/Desktop/test_tube')

exp.tag({'learning_rate': 0.002, 'nb_layers': 2})

for step in training_steps:
    tng_err = model.eval(tng_x, tng_y)

    exp.log('tng_err': tng_err)

# training complete!
# all your logs and data are ready to be visualized at testtube.williamfalcon.com
```

------------------------------------------------------------------------

## init options

### version

The same Experiment can have multiple versions. Test tube generates
these automatically each time you run your model. To set your own
version use:

``` {.python}
exp = Experiment(name='dense_model',version=1)
```

### debug

If you're debugging and don't want to create a log file turn debug to
True

``` {.python}
exp = Experiment(name='dense_model',debug=True)
```

### autosave

If you only want to save at the end of training, turn autosave off:

``` {.python}
exp = Experiment(name='dense_model', autosave=False)

# run long training...

# first time any logs are saved
exp.save()
```

### `create_git_tag`

Ever wanted a flashback to your code when you ran an experiment?
Snapshot your code for this experiment using git tags:

``` {.python}
exp = Experiment(name='dense_model', create_git_tag=True)
```

------------------------------------------------------------------------

## Methods

### tag

``` {.python}
exp.tag({k: v})
```

Adds an arbitrary dictionary of tags to the experiment

**Example**

``` {.python}
exp.tag({'dataset_name': 'imagenet_1', 'learning_rate': 0.0002})
```

### log

``` {.python}
exp.log({k:v})
```

Adds a row of data to the experiments

**Example**

``` {.python}
exp.log({'val_loss': 0.22, 'epoch_nb': 1, 'batch_nb': 12})

# you can also add other rows that have separate information
exp.log({'tng_loss': 0.01})

# or even a numpy array image
image = np.imread('image.png')
exp.log({'fake_png': image})
```

**Saving images Example**

``` {.python}
# name must have either jpg, png or jpeg in it
img = np.imread('a.jpg')
exp.log('test_jpg': img, 'val_err': 0.2)

# saves image to ../exp/version/media/test_0.jpg
# csv has file path to that image in that cell
```

To save an image, add `jpg`, `png` or `jpeg` to the key corresponding
with the image array. The image must be formatted the same as skimage's
[imsave](http://scikit-image.org/docs/dev/api/skimage.io.html#skimage.io.imsave)
function

### argparse

``` {.python}
exp.argparse(hparams)
```

Transfers hyperparam information from Argparser or
HyperOptArgumentParser

**Example**

``` {.python}
from test_tube import HyperOptArgumentParser

# parse args
parser = HyperOptArgumentParser()
parser.add_argument('--learning_rate', default=0.002, type=float, help='the learning rate')
hparams = parser.parse_args()

# learning_rate is now a meta tag for your experiment
exp.argparse(hparams)
```

### save

``` {.python}
exp.save()
```

Saves the exp to disk (including images)

**Example**

``` {.python}
exp = Experiment(name='dense_model', autosave=False)

# run long training...

# first time any logs are saved
exp.save()
```
