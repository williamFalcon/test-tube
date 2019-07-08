import os
import sys
from .argparse_hopt import HyperOptArgumentParser
from subprocess import call
import datetime
import traceback
import re
from shutil import copyfile
import threading
import time
import signal
import pdb


def exit():
    time.sleep(1)
    os._exit(1)


class AbstractCluster(object):

    RUN_CMD = 'sbatch'
    def __init__(
            self,
            hyperparam_optimizer=None,
            log_path=None,
            python_cmd='python3',
            enable_log_err=True,
            enable_log_out=True,
    ):
        self.hyperparam_optimizer = hyperparam_optimizer
        self.log_path = log_path

        self.enable_log_err = enable_log_err
        self.enable_log_out = enable_log_out
        self.slurm_files_log_path = None
        self.err_log_path = None
        self.out_log_path = None
        self.modules = []
        self.script_name = os.path.realpath(sys.argv[0])
        self.job_time = '15:00'
        self.minutes_to_checkpoint_before_walltime = 5
        self.per_experiment_nb_gpus = 1
        self.per_experiment_nb_cpus = 1
        self.per_experiment_nb_nodes = 1
        self.memory_mb_per_node = 2000
        self.email = None
        self.notify_on_end = False
        self.notify_on_fail = False
        self.job_name = None
        self.python_cmd = python_cmd
        self.gpu_type = None
        self.on_gpu = False
        self.call_load_checkpoint = False
        self.commands = []
        self.slurm_commands = []
        self.hpc_exp_number = 0

        # these are set via getters and setters so we can use a BaseManager which can be shared across processes
        self.checkpoint_save_function = None
        self.checkpoint_load_function = None

        # detect when this was called because a slurm object started a hopt.
        # if true, remove the flag so tt logs don't show it
        if hyperparam_optimizer is not None:

            self.is_from_slurm_object = HyperOptArgumentParser.TRIGGER_CMD in vars(self.hyperparam_optimizer) and vars(self.hyperparam_optimizer)[HyperOptArgumentParser.TRIGGER_CMD] == True
            if self.is_from_slurm_object:
                self.hyperparam_optimizer.__delattr__(HyperOptArgumentParser.TRIGGER_CMD)

            self.call_load_checkpoint = HyperOptArgumentParser.SLURM_LOAD_CMD in vars(self.hyperparam_optimizer)
            if self.call_load_checkpoint:
                self.hyperparam_optimizer.__delattr__(HyperOptArgumentParser.SLURM_LOAD_CMD)

            self.hpc_exp_number = self.hyperparam_optimizer.hpc_exp_number

    def set_checkpoint_save_function(self, fx, kwargs):
        self.checkpoint_save_function = [fx, kwargs]

    def get_checkpoint_save_function(self):
        return self.checkpoint_save_function

    def set_checkpoint_load_function(self, fx, kwargs):
        # if we were passed in the load flag, then we call the load function as soon as it's added
        if self.call_load_checkpoint:
            fx(**kwargs)

        self.checkpoint_load_function = [fx, kwargs]

    def get_checkpoint_load_function(self):
        return self.checkpoint_load_function

    def add_slurm_cmd(self, cmd, value, comment):
        self.slurm_commands.append((cmd, value, comment))

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

    def optimize_parallel_cluster_gpu(
            self,
            train_function,
            nb_trials,
            job_name,
            job_display_name=None
    ):
        if job_display_name is None:
            job_display_name = job_name

        self.__optimize_parallel_cluster_internal(train_function, nb_trials, job_name, job_display_name, on_gpu=True)

    def optimize_parallel_cluster_cpu(
            self,
            train_function,
            nb_trials,
            job_name,
            job_display_name=None
    ):
        if job_display_name is None:
            job_display_name = job_name

        self.__optimize_parallel_cluster_internal(train_function, nb_trials, job_name, job_display_name, on_gpu=False)

    def __optimize_parallel_cluster_internal(
            self,
            train_function,
            nb_trials,
            job_name,
            job_display_name,
            on_gpu
    ):
        """
        Runs optimization on the attached cluster
        :param train_function:
        :param nb_trials:
        :param job_name:
        :return:
        """
        self.job_name = job_name
        self.job_display_name = job_display_name
        self.on_gpu = on_gpu

        # layout logging structure
        self.__layout_logging_dir()

        if self.is_from_slurm_object:
            # Script is called by slurm: it's an actual experiment.
            self.__run_experiment(train_function)
        else:
            # Launcher script. Generate trials and launch jobs.

            # generate hopt trials
            trials = self.hyperparam_optimizer.generate_trials(nb_trials)

            # get the max test tube exp version so far if it's there
            next_test_tube_version = self.__get_max_test_tube_version(self.log_path)

            # for each trial, generate a slurm command
            for i, trial_params in enumerate(trials):
                exp_i = i + next_test_tube_version
                self.schedule_experiment(trial_params, exp_i)

    def schedule_experiment(self, trial_params, exp_i):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        timestamp = 'trial_{}_{}'.format(exp_i, timestamp)

        # generate command
        slurm_cmd_script_path = os.path.join(self.slurm_files_log_path, '{}_slurm_cmd.sh'.format(timestamp))
        slurm_cmd = self.__build_slurm_command(trial_params, slurm_cmd_script_path, timestamp, exp_i, self.on_gpu)
        self.__save_slurm_cmd(slurm_cmd, slurm_cmd_script_path)

        # run script to launch job
        print('\nlaunching exp...')
        result = call('{} {}'.format(AbstractCluster.RUN_CMD, slurm_cmd_script_path), shell=True)
        if result == 0:
            print('launched exp ', slurm_cmd_script_path)
        else:
            print('launch failed...')

    def slurm_time_to_seconds(self, job_time):
        seconds = 0
        time_component = job_time
        if '-' in job_time:
            days, time_component = job_time.split('-')
            seconds += int(days) * 24 * 60 * 60

        time_components = time_component.split(':')
        if len(time_components) == 3:
            hours, minutes, secs = time_components
            time_seconds = int(secs) + (int(minutes) * 60) + (int(hours) * 60 * 60)
            seconds += time_seconds

        elif len(time_components) == 2:
            minutes, secs = time_components
            time_seconds = int(secs) + (int(minutes) * 60)
            seconds += time_seconds

        elif len(time_components) == 1:
            secs = time_components[0]
            seconds += int(secs)

        return seconds

    def call_save(self):
        print('calling save')

        # if save function was passed, call it
        if self.get_checkpoint_save_function() is not None:
            save_fx, kwargs = self.get_checkpoint_save_function()
            save_fx(**kwargs)

            # if we're here, the job didn't finish and we were given a save function
            # if we were given a load function, then schedule the program again and pass in the load function
            if self.get_checkpoint_load_function() is not None:

                # copy the original slurm command into a new file, rename with current time, add load_flag
                # and call
                original_slurm_cmd_script_path = self.hyperparam_optimizer.test_tube_slurm_cmd_path
                exp_i = self.hyperparam_optimizer.hpc_exp_number
                self.__call_old_slurm_cmd(original_slurm_cmd_script_path, exp_i)

        # stop program
        os._exit(0)

    def sig_handler(self, signum, frame):
        print("caught signal", signum)
        self.call_save()

        # sys.exit(-1)

    # ------------------------
    # HANDLE SLURM SIGNALS
    # ------------------------
    def term_handler(self, signum, frame):
        print("bypassing sigterm")

    def __run_experiment(self, train_function):
        print('setting signal')
        signal.signal(signal.SIGUSR1, self.sig_handler)
        signal.signal(signal.SIGTERM, self.term_handler)

        try:
            # run training
            train_function(self.hyperparam_optimizer, self, {})

        except Exception as e:
            print('Caught exception in worker thread', e)

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            raise SystemExit

    def __call_old_slurm_cmd(self, original_slurm_cmd_script_path, exp_i, copy_current=True):
        """
        Copies old slurm script into a new one and adds a load flag in case it wasn't there.
        Then schedules the script again, but this time with the load flag which will signal the program
        to load the model so it can continue training.

        :param original_slurm_cmd_script_path:
        :param exp_i:
        :param copy_current:
        :return:
        """

        # generate command
        script_path = original_slurm_cmd_script_path.split('slurm_scripts')[0] + 'slurm_scripts'
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        timestamp = 'trial_{}_{}'.format(exp_i, timestamp)
        new_slurm_cmd_script_path = os.path.join(script_path, '{}_slurm_cmd.sh'.format(timestamp))

        # copy with new time
        copyfile(original_slurm_cmd_script_path, new_slurm_cmd_script_path)

        # add continue flag if not there
        old_file = open(original_slurm_cmd_script_path)
        lines = old_file.read().split('\n')
        last_line = lines[-1]
        lines = [line + '\n' for line in lines]
        lines[-1] = last_line
        old_file.close()
        if not HyperOptArgumentParser.SLURM_LOAD_CMD in lines[-1]:
            last_line = lines[-1]
            last_line = '{} --{}\n'.format(last_line, HyperOptArgumentParser.SLURM_LOAD_CMD)
            lines[-1] = last_line
            open(new_slurm_cmd_script_path, 'w').writelines(lines)

        # run script to launch job
        print('\nlaunching exp...')
        result = call('{} {}'.format(AbstractCluster.RUN_CMD, new_slurm_cmd_script_path), shell=True)
        if result == 0:
            print('launched exp ', new_slurm_cmd_script_path)
        else:
            print('launch failed...')

    def __save_slurm_cmd(self, slurm_cmd, slurm_cmd_script_path):
        with open(slurm_cmd_script_path, mode='w') as file:
            file.write(slurm_cmd)

    def __get_max_test_tube_version(self, path):
        files = os.listdir(path)
        version_files = [f for f in files if 'version_' in f]
        if len(version_files) > 0:
            # regex out everything except file version for ve
            versions = [int(re.sub('version_', '', f_name)) for f_name in version_files]
            max_version = max(versions)
            return max_version + 1
        else:
            return 0

    def __layout_logging_dir(self):
        """
        Generates dir structure for logging errors and outputs
        :return:
        """

        # format the logging folder path
        slurm_out_path = os.path.join(self.log_path, self.job_name)

        self.log_path = slurm_out_path

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

            # don't add None params
            if v is None or v == False:
                continue

            # put everything in quotes except bools
            if self.__should_escape(v):
                cmd = '--{} \"{}\"'.format(k, v)
            else:
                cmd = '--{} {}'.format(k, v)
            params.append(cmd)

        # this arg lets the hyperparameter optimizer do its thing
        params.append('--{}'.format(HyperOptArgumentParser.TRIGGER_CMD))

        full_cmd = ' '.join(params)
        return full_cmd

    def __should_escape(self, v):
        v = str(v)
        return '[' in v or ';' in v or ' ' in v

    def __build_slurm_command(self, trial, slurm_cmd_script_path, timestamp, exp_i, on_gpu):
        sub_commands = []

        command =[
            '#!/bin/bash',
            '#',
            '# Auto-generated by test-tube (https://github.com/williamFalcon/test-tube)',
            '#################\n'
        ]
        sub_commands.extend(command)

        # add job name
        job_with_version = '{}v{}'.format(self.job_display_name, exp_i)
        command = [
            '# set a job name',
            '#SBATCH --job-name={}'.format(job_with_version),
            '#################\n',
        ]
        sub_commands.extend(command)

        # add out output
        if self.enable_log_out:
            out_path = os.path.join(self.out_log_path, '{}_slurm_output_%j.out'.format(timestamp))
            command = [
                '# a file for job output, you can check job progress',
                '#SBATCH --output={}'.format(out_path),
                '#################\n',
            ]
            sub_commands.extend(command)

        # add err output
        if self.enable_log_err:
            err_path = os.path.join(self.err_log_path, '{}_slurm_output_%j.err'.format(timestamp))
            command = [
                '# a file for errors',
                '#SBATCH --error={}'.format(err_path),
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
        if self.per_experiment_nb_gpus > 0 and on_gpu:
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

        # add nb of cpus if not looking at a gpu job
        if self.per_experiment_nb_cpus > 0:
            command = [
                '# cpus per job',
                '#SBATCH --cpus-per-task={}'.format(self.per_experiment_nb_cpus),
                '#################\n'
            ]
            sub_commands.extend(command)

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

        # add signal command to catch job termination
        command = [
            '# slurm will send a signal this far out before it kills the job',
            f'#SBATCH --signal=USR1@{self.minutes_to_checkpoint_before_walltime * 60}',
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

        # add custom sbatch commands
        sub_commands.append('\n')
        for (cmd, value, comment) in self.slurm_commands:
            comment = '# {}'.format(comment)
            cmd = '#SBATCH --{}={}'.format(cmd, value)
            spaces = '#################\n'
            sub_commands.extend([comment, cmd, spaces])

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
        trial_args = '{} --{} {} --{} {}'.format(trial_args,
                                                 HyperOptArgumentParser.SLURM_CMD_PATH,
                                                 slurm_cmd_script_path,
                                                 HyperOptArgumentParser.SLURM_EXP_CMD,
                                                 exp_i)

        cmd = 'srun {} {} {}'.format(self.python_cmd, self.script_name, trial_args)
        sub_commands.append(cmd)

        # build full command with empty lines in between
        full_command = '\n'.join(sub_commands)
        return full_command
