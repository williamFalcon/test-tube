import os

class AbstractCluster(object):
    def __init__(self, log_path, enable_log_err=True, enable_log_out=True, test_tube_exp_name=None):
        self.log_path = log_path
        self.enable_log_err = enable_log_err
        self.enable_log_out = enable_log_out
        self.test_tube_exp_name = test_tube_exp_name
        self.err_log_path = None
        self.out_log_path = None

    def schedule(self, trials, job_name):
        raise NotImplementedError

    def optimize_parallel_slurm(self, job_name, output_file, error_file, job_time, nb_gpus, nb_nodes, memory, notifications_email, gpu_types):
        pass


class SlurmCluster(AbstractCluster):
    def __init__(self, *args, **kwargs):
        super(SlurmCluster, self).__init__(*args, **kwargs)

    def schedule(self, trials, job_name):
        self.__layout_logging_dir()
        pass

    def __layout_logging_dir(self):
        # make main folder for slurm output
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # if we have a test tube name, make the folder and set as the logging destination
        if self.test_tube_exp_name is not None:
            slurm_out_path = os.path.join(self.log_path, self.test_tube_exp_name)
            if not os.path.exists(slurm_out_path):
                os.makedirs(slurm_out_path)

            # when err logging is enabled, build add the err logging folder
            if self.enable_log_err:
                err_path = os.path.join(slurm_out_path, 'err_logs')
                if not os.path.exists(err_path):
                    os.makedirs(err_path)
                self.err_log_path = err_path

            # when out logging is enabled, build add the out logging folder
            if self.enable_log_out:
                out_path = os.path.join(slurm_out_path, 'out_logs')
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                self.out_log_path = out_path

            self.log_path = slurm_out_path