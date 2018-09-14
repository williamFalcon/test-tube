
class AbstractCluster(object):
    def __init__(self):
        pass


    def optimize_parallel_slurm(self, job_name, output_file, error_file, job_time, nb_gpus, nb_nodes, memory, notifications_email, gpu_types):
        pass


class SlurmCluster(AbstractCluster):
    def __init__(self):
        super(SlurmCluster, self).__init__()
        pass