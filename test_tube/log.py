import os
import json
from datetime import datetime
from .lib import img_io

# constants
_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data_path():
    """
    Returns the path to the local package cache
    :param path:
    :return:
    """
    return os.path.join(_ROOT, 'test_tube_data')


def get_media_path(exp_name):
    """
    Returns the path to the local package cache
    :param path:
    :return:
    """
    return os.path.join(get_data_path(), 'media_{}'.format(exp_name))

# -----------------------------
# Experiment object
# -----------------------------


class Experiment(object):
    name = None
    version = None
    tag = None
    debug = False
    autosave = None
    created_at = str(datetime.utcnow())
    description = None
    tags = {}
    metrics = []
    create_git_tag = False
    exp_hash = None

    def __init__(self, name='default', debug=False, version=None, save_dir=None, autosave=True, description=None, create_git_tag=False):
        """
        A new Experiment object defaults to 'default' unless a specific name is provided
        If a known name is already provided, then the file version is changed
        :param name:
        :param debug:
        """
        # change where the save dir is if requested
        if save_dir is not None:
            global _ROOT
            _ROOT = save_dir

        self.name = name
        self.debug = debug
        self.version = version
        self.autosave = autosave
        self.description = description
        self.create_git_tag = create_git_tag
        self.exp_hash = '{}_{}'.format(self.name, version)

        # update version hash if we need to increase version on our own
        # we will increase the previous version, so do it now so the hash
        # is accurate
        if self.version is None:
            old_version = self.__get_last_experiment_version()
            self.exp_hash = '{}_{}'.format(self.name, old_version + 1)

        self.__init_cache_file_if_needed()

        # create a new log file if not in debug mode
        if not debug:

            # when we have a version, load it
            if self.version is not None:

                # when no version and no file, create it
                if not os.path.exists(self.__get_log_name()):
                    self.__create_exp_file(self.version)
                    self.save()
                else:
                    # otherwise load it
                    self.__load()
            else:
                # if no version given, increase the version to a new exp
                # create the file if not exists
                old_version = self.__get_last_experiment_version()
                self.version = old_version
                self.__create_exp_file(self.version + 1)
                self.save()

            # create a git tag if requested
            if self.create_git_tag == True:
                desc = description if description is not None else 'no description'
                tag_msg = 'Test tube exp: {} - {}'.format(self.name, desc)
                cmd = 'git tag -a tt_{} -m "{}"'.format(self.exp_hash, tag_msg)
                os.system(cmd)
                print('Test tube created git tag:', 'tt_{}'.format(self.exp_hash))

    # --------------------------------
    # FILE IO UTILS
    # --------------------------------
    def __init_cache_file_if_needed(self):
        """
        Inits a file that we log historical experiments
        :return:
        """
        exp_cache_file = get_data_path()
        if not os.path.exists(exp_cache_file):
            os.mkdir(exp_cache_file)


    def __create_exp_file(self, version):
        """
        Recreates the old file with this exp and version
        :param version:
        :return:
        """
        exp_cache_file = get_data_path()
        # if no exp, then make it
        path = '{}/{}.experiment'.format(exp_cache_file, self.exp_hash)
        open(path, 'w').close()
        self.version = version

        # make the directory for the experiment media assets name
        os.mkdir(get_media_path(self.exp_hash))

    def __get_last_experiment_version(self):
        try:
            exp_cache_file = get_data_path()
            last_version = -1
            for f in os.listdir(exp_cache_file):
                if '_' in f:
                    name, version = f.split('_')[0:2]
                    if self.name == name:
                        version = int(version.split('.')[0])
                        last_version = max(last_version, version)
            return last_version
        except Exception as e:
            return -1

    def __get_log_name(self):
        exp_cache_file = get_data_path()
        return '{}/{}_{}.experiment'.format(exp_cache_file, self.name, self.version)

    def add_meta_tag(self, key, val):
        """
        Adds a tag to the experiment.
        Tags are metadata for the exp

        >> e.add_meta_tag({"model": "Convnet A"})

        :param key:
        :param val:
        :return:
        """
        if self.debug: return

        self.tags[key] = val
        if self.autosave == True:
            self.save()

    def add_meta_tags(self, tag_dict):
        """
        Adds a tag to the experiment.
        Tags are metadata for the exp

        >> e.add_meta_tag({"model": "Convnet A"})

        :param key:
        :param val:
        :return:
        """
        if self.debug: return

        # parse tags
        for k, v in tag_dict.items():
            self.tags[k] = v

        # save if needed
        if self.autosave == True:
            self.save()

    def add_metric_row(self, metrics_dict):
        """
        Adds a json dict of metrics.

        >> e.add_metrics({"loss": 23, "coeff_a": 0.2})

        :param metrics_dict:
        :return:
        """
        if self.debug: return

        # timestamp
        if 'created_at' not in metrics_dict:
            metrics_dict['created_at'] = str(datetime.utcnow())

        self.__convert_numpy_types(metrics_dict)
        self.metrics.append(metrics_dict)
        if self.autosave == True:
            self.save()

    def __convert_numpy_types(self, metrics_dict):
        for k, v in metrics_dict.items():
            if v.__class__.__name__ == 'float32':
                metrics_dict[k] = float(v)

            if v.__class__.__name__ == 'float64':
                metrics_dict[k] = float(v)

    def save(self):
        """
        Saves current experiment progress
        :return:
        """
        if self.debug: return

        # save images and replace the image array with the
        # file name
        self.__save_images(self.metrics)

        obj = {
            'name': self.name,
            'version': self.version,
            'tags': self.tags,
            'metrics': self.metrics,
            'autosave': self.autosave,
            'description': self.description,
            'created_at': self.created_at,
            'exp_hash': self.exp_hash
        }
        with open(self.__get_log_name(), 'w') as file:
            json.dump(obj, file, ensure_ascii=False)

    def __save_images(self, metrics):
        """
        Save tags that have a png_ prefix (as images)
        and replace the meta tag with the file name
        :param metrics:
        :return:
        """
        # iterate all metrics and find keys with a specific prefix
        for i, metric in enumerate(metrics):
            for k, v in metric.items():
                # if the prefix is a png, save the image and replace the value with the path
                if 'png_' in k:
                    # determine the file name
                    img_name = '_'.join(k.split('_')[1:])
                    save_path = get_media_path(self.exp_hash)
                    save_path = '{}/{}_{}.png'.format(save_path, img_name, i)

                    # save image to disk
                    img_io.save_as_png(metric[k], save_path)

                    # replace the image in the metric with the file path
                    metric[k] = save_path

    def __load(self):
        with open(self.__get_log_name(), 'r') as file:
            data = json.load(file)
            self.name = data['name']
            self.version = data['version']
            self.tags = data['tags']
            self.metrics = data['metrics']
            self.autosave = data['autosave']
            self.created_at = data['created_at']
            self.description = data['description']
            self.exp_hash = data['exp_hash']



    # ----------------------------
    # OVERWRITES
    # ----------------------------

    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

