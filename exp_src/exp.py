import os

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
    internal_data = {}

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
            # when we have a version, overwrite it
            if self.version:
                self.__create_exp_file(version)
            else:
                # if no version given, just increase the version to a new exp
                old_version = self.__get_last_experiment_version()
                self.__create_exp_file(old_version + 1)

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

    def __create_exp_file(self, old_version):
        """
        Recreates the old file with this exp and version
        :param old_version:
        :return:
        """
        exp_cache_file = get_data_path()
        # if no exp, then make it
        path = '{}/{}_{}.experiment'.format(exp_cache_file, self.name, old_version)
        open(path, 'w').close()

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

    # ----------------------------
    # OVERWRITES
    # ----------------------------

    def __str__(self):
        return 'Exp: {}, v: {}'.format(self.name, self.version)

