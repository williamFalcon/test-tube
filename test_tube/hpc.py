import os
import sys
from .argparse_hopt import HyperOptArgumentParser
from subprocess import call
import datetime
import traceback

class AbstractCluster(object):


    def __init__(
            self,
            hyperparam_optimizer: HyperOptArgumentParser,
            log_path: str,
            python_cmd: str ='python3',
            enable_log_err: bool =True,
            enable_log_out: bool =True,
            test_tube_exp_name: str =None
    ):
        self.hyperparam_optimizer = hyperparam_optimizer
        self.log_path = os.path.join(log_path, 'test_tube_data')
        self.enable_log_err = enable_log_err
        self.enable_log_out = enable_log_out
        self.test_tube_exp_name = test_tube_exp_name
        self.slurm_files_log_path = None
        self.err_log_path = None
        self.out_log_path = None
        self.modules = []
        self.script_name = sys.argv[0]
        self.job_time = '15:00'
        self.per_experiment_nb_gpus = 1
        self.hns_gpu = False
        self.per_experiment_nb_nodes = 1
        self.memory_mb_per_node = 4000
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.job_name = None
        self.python_cmd = python_cmd
        self.gpu_type: str = None
        self.commands = []

        # detect when this was called because a slurm object started a hopt.
        # if true, remove the flag so tt logs don't show it
        self.is_from_slurm_object = HyperOptArgumentParser.TRIGGER_CMD in vars(self.hyperparam_optimizer)
        if self.is_from_slurm_object:
            self.hyperparam_optimizer.__delattr__(HyperOptArgumentParser.TRIGGER_CMD)

    def add_command(self, cmd):
        self.commands.append(cmd)

    def load_modules(self, modules):
        self.modules = modules

    def notify_job_status(self, email, on_done, on_fail):
        self.email = email
        self.notify_on_end = on_done
        self.notify_on_fail = on_fail

    def optimize_parallel_cluster(self, train_function, nb_trials, job_name):
        raise NotImplementedError

    def optimize_parallel_slurm(self, job_name, output_file, error_file, job_time, nb_gpus, nb_nodes, memory, notifications_email, gpu_types):
        pass


class SlurmCluster(AbstractCluster):
    def __init__(self, *args, **kwargs):
        super(SlurmCluster, self).__init__(*args, **kwargs)

    def optimize_parallel_cluster(
            self,
            train_function,
            nb_trials: int,
            job_name: str
    ):
        """
        Runs optimization on the attached cluster
        :param train_function:
        :param nb_trials:
        :param job_name:
        :return:
        """
        self.job_name = job_name

        # whenever this script is called by slurm, it's an actual experiment, so start it
        if self.is_from_slurm_object:
            results = self.__run_experiment(train_function)
            return results

        # generate hopt trials
        trials = self.hyperparam_optimizer.generate_trials(nb_trials)

        # layout logging structure
        self.__layout_logging_dir()

        # for each trial, generate a slurm command
        for i, trial_params in enumerate(trials):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
            timestamp = 'trial_{}_{}'.format(i, timestamp)

            # generate command
            slurm_cmd_script_path = os.path.join(self.slurm_files_log_path, '{}_slurm_cmd.sh'.format(timestamp))
            slurm_cmd = self.__build_slurm_command(trial_params, slurm_cmd_script_path)
            self.__save_slurm_cmd(slurm_cmd, slurm_cmd_script_path)

            # run script to launch job
            print('\nlaunching exp...')
            result = call('sh {}'.format(slurm_cmd_script_path), shell=True)
            if result == 0:
                print('launched exp ', slurm_cmd_script_path)
            else:
                print('launch failed...')

    def __run_experiment(self, train_function):
        try:
            results = train_function(self.hyperparam_optimizer)
            return results
        except Exception as e:
            print('Caught exception in worker thread', e)

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            return [self.hyperparam_optimizer, None]

    def __save_slurm_cmd(self, slurm_cmd, slurm_cmd_script_path):
        with open(file=slurm_cmd_script_path, mode='w') as file:
            file.write(slurm_cmd)


    def __layout_logging_dir(self):
        """
        Generates dir structure for logging errors and outputs
        :return:
        """

        # make main folder for slurm output
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # format the logging folder path
        if self.test_tube_exp_name is not None:
            slurm_out_path = os.path.join(self.log_path, self.test_tube_exp_name)
        else:
            slurm_out_path = os.path.join(self.log_path, self.job_name)

        # if we have a test tube name, make the folder and set as the logging destination
        if not os.path.exists(slurm_out_path):
            os.makedirs(slurm_out_path)

        # when err logging is enabled, build add the err logging folder
        if self.enable_log_err:
            err_path = os.path.join(slurm_out_path, 'slurm_err_logs')
            if not os.path.exists(err_path):
                os.makedirs(err_path)
            self.err_log_path = err_path

        # when out logging is enabled, build add the out logging folder
        if self.enable_log_out:
            out_path = os.path.join(slurm_out_path, 'slurm_out_logs')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.out_log_path = out_path

        # place where slurm files log to
        self.slurm_files_log_path = os.path.join(slurm_out_path, 'slurm_scripts')
        if not os.path.exists(self.slurm_files_log_path):
            os.makedirs(self.slurm_files_log_path)

    def __get_hopt_params(self, trial):
        """
        Turns hopt trial into script params
        :param trial:
        :return:
        """

        params = []
        for k in trial.__dict__:
            v = trial.__dict__[k]
            cmd = '--{} {}'.format(k, v)
            params.append(cmd)

        # this arg lets the hyperparameter optimizer do its thing
        params.append('--{}'.format(HyperOptArgumentParser.TRIGGER_CMD))

        full_cmd = ' '.join(params)
        return full_cmd

    def __build_slurm_command(self, trial, slurm_cmd_script_path):
        sub_commands = []

        command =[
            '#!/bin/bash',
            '#',
            '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add job name
        command = [
            '# set a job name',
            '#SBATCH --job-name={}'.format(self.job_name),
            '#################\n',
        ]
        sub_commands.extend(command)

        # add out output
        if self.enable_log_out:
            command = [
                '# a file for job output, you can check job progress',
                '#SBATCH --output=slurm_output.out',
                '#################\n',
            ]
            sub_commands.extend(command)

        # add err output
        if self.enable_log_err:
            command = [
                '# a file for errors',
                '#SBATCH --error=slurm_output.err',
                '#################\n',
            ]
            sub_commands.extend(command)

        # add job time
        command = [
            '# time needed for job',
            '#SBATCH --time={}'.format(self.job_time),
            '#################\n'
        ]
        sub_commands.extend(command)

        # add nb of gpus
        command = [
            '# gpus per cluster',
            '#SBATCH --gres gpu:{}'.format(self.per_experiment_nb_gpus),
            '#################\n'
        ]
        if self.gpu_type is not None:
            command = [
                '# gpus per cluster',
                '#SBATCH --gres gpu:{}:{}'.format(self.gpu_type, self.per_experiment_nb_gpus),
                '#################\n'
            ]

        sub_commands.extend(command)

        # pick gpu partition
        partition_cmd = [
            '# gpu partition',
            '#SBATCH --qos=batch',
            '#################\n'
        ]
        if self.hns_gpu:
            partition_cmd = [
                '# gpu partition',
                '#SBATCH -p hns_gpu',
                '#################\n'
            ]
        sub_commands.extend(partition_cmd)

        # pick nb nodes
        command = [
            '# number of requested nodes',
            '#SBATCH --nodes={}'.format(self.per_experiment_nb_nodes),
            '#################\n'
        ]
        sub_commands.extend(command)

        # pick memory per node
        command = [
            '# memory per node',
            '#SBATCH --mem={}'.format(self.memory_mb_per_node),
            '#################\n'
        ]
        sub_commands.extend(command)

        # Subscribe to email if requested
        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0:
            mail_type_query = [
                '# Have SLURM send you an email when the job ends or fails',
                '#SBATCH --mail-type={}'.format(','.join(mail_type))
            ]
            sub_commands.extend(mail_type_query)

            email_query = [
                '#SBATCH --mail-user={}'.format(self.email),
            ]
            sub_commands.extend(email_query)

        # load modules
        sub_commands.append('\n')
        for module in self.modules:
            cmd = 'module load {}'.format(module)
            sub_commands.append(cmd)

        # remove spaces before the hash
        sub_commands = [x.lstrip() for x in sub_commands]

        # add additional commands
        for cmd in self.commands:
            sub_commands.append(cmd)
            sub_commands.append('\n')

        # add run command
        trial_args = self.__get_hopt_params(trial)
        trial_args = '{} --{} {}'.format(trial_args, HyperOptArgumentParser.SLURM_CMD_PATH, slurm_cmd_script_path)
        cmd = 'srun {} .{} {}'.format(self.python_cmd, self.script_name, trial_args)
        sub_commands.append(cmd)

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command

















