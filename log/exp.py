import os
import json


# constants
_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data_path():
    """
    Returns the path to the local package cache
    :param path:
    :return:
    """
    return os.path.join(_ROOT, '../data')

# -----------------------------
# Experiment object
# -----------------------------


class Experiment(object):
    name = None
    version = None
    tag = None
    debug = False
    tags = {}
    metrics = []

    def __init__(self, name='default', debug=False, version=None):
        """
        A new Experiment object defaults to 'default' unless a specific name is provided
        If a known name is already provided, then the file version is changed
        :param name:
        :param debug:
        """
        self.__init_cache_file_if_needed()
        self.name = name
        self.debug = debug
        self.version = version

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
        path = '{}/{}_{}.experiment'.format(exp_cache_file, self.name, version)
        open(path, 'w').close()
        self.version = version

    def __get_last_experiment_version(self):
        exp_cache_file = get_data_path()
        last_version = -1
        for f in os.listdir(exp_cache_file):
            if '_' in f:
                name, version = f.split('_')[0:2]
                if self.name == name:
                    version = int(version.split('.')[0])
                    last_version = max(last_version, version)
        return last_version

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
        self.save()

    def add_metric_row(self, metrics_dict):
        """
        Adds a json dict of metrics.

        >> e.add_metrics({"loss": 23, "coeff_a": 0.2})

        :param metrics_dict:
        :return:
        """
        if self.debug: return

        self.metrics.append(metrics_dict)
        self.save()

    def save(self):
        if self.debug: return
        obj = {
            'name': self.name,
            'version': self.version,
            'tags': self.tags,
            'metrics': self.metrics
        }
        with open(self.__get_log_name(), 'w') as file:
            json.dump(obj, file, ensure_ascii=False)

    def __load(self):
        with open(self.__get_log_name(), 'r') as file:
            data = json.load(file)
            self.name = data['name']
            self.version = data['version']
            self.tags = data['tags']
            self.metrics = data['metrics']

    # ----------------------------
    # OVERWRITES
    # ----------------------------

    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

