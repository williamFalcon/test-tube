# SlurmCluster class API

The SlurmCluster class enables hyperparameter search parallelization on a cluster managed via [Slurm workload manager](https://slurm.schedmd.com/).

At a high level, the SlurmCluster creates a submit script for each permutation of hyperparameters requested. If the job hits the walltime but has not completed, the SlurmManager will checkpoint the model and submit a new job to continue training using the saved weights.
    
You can instantiate a `SlurmCluster` via:   

``` {.python}
from test_tube.hpc import SlurmCluster

# hyperparameters is a test-tube hyper params object
# see https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/
hyperparams = args.parse()

# init cluster
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path='/path/to/log/results/to',
    python_cmd='python3'
)

# let the cluster know where to email for a change in job status (ie: complete, fail, etc...)
cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)

# set the job options. In this instance, we'll run 20 different models
# each with its own set of hyperparameters giving each one 1 GPU (ie: taking up 20 GPUs)
cluster.per_experiment_nb_gpus = 1
cluster.per_experiment_nb_nodes = 1

# we'll request 10GB of memory per node
cluster.memory_mb_per_node = 10000

# set a walltime of 10 minues
cluster.job_time = '10:00'

# 1 minute before walltime is up, SlurmCluster will launch a continuation job and kill this job.
# you must provide your own loading and saving function which the cluster object will call
cluster.minutes_to_checkpoint_before_walltime = 1

# run the models on the cluster
cluster.optimize_parallel_cluster_gpu(train, nb_trials=20, job_name='first_tt_batch', job_display_name='my_batch')
```   

------------------------------------------------------------------------

## init options

### `hyperparam_optimizer`

A [HyperOptArgumentParser](https://williamfalcon.github.io/test-tube/hyperparameter_optimization/HyperOptArgumentParser/) object 
which contains all permutations of model hyperparameters to run.   

### `log_path`

Path to save the slurm scripts, error logs and out logs created. Usually this would be the experiments folder path where test tube saves [Experiment](https://williamfalcon.github.io/test-tube/experiment_tracking/experiment/) information.

### `python_cmd`

This is the command that starts the python program. Normally it is:   

``` {.python}
# python 2
python main.py   

# python 3   
python3 main.py
```   

### `enable_log_err`

If true, saves slurm error logs to the path at *log_path*. If anything goes wrong in your job, you'll find the error here.   

### `enable_log_out`

If true, saves slurm output logs to the path at *log_path*. This file contains all outputs that would show up on the console normally.   

### `test_tube_exp_name`

When this is given, it structures the files in a nice format to fit with the folder structure of the [Experiment](https://williamfalcon.github.io/test-tube/experiment_tracking/experiment/) object's output.

## Methods    
Properties   

- `job_time` String. Walltime requested. Examples:    
```{.python}    
# 1 hour and 10 minutes    
cluster.job_time = '1:10:00'

# 1 day and 1 hour and 10 minutes    
cluster.job_time = '1-1:10:00'

# 1 day and 1 hour and 10 minutes    
cluster.job_time = '25:10:00'   

# 10 minutes    
cluster.job_time = '10:00'   

# 10 seconds    
cluster.job_time = '10'   
```   

- `minutes_to_checkpoint_before_walltime` Int. Minutes before walltime when a continuation job will be auto-submitted. 
```{.python}
cluster.job_time = '10:00'   
cluster.minutes_to_checkpoint_before_walltime = 2

# New job will be submited to continue training after 8 minutes of the job running.      
```     

- `per_experiment_nb_gpus` Int. Number of GPUs each job will get.   
```{.python}
# EACH job will get 2 GPUs (ie: if a model runs over two GPUs at the same time).   
cluster.per_experiment_nb_gpus = 2  
```     

- `per_experiment_nb_cpus` Int. Number of CPUs each job will get.   
```{.python}
cluster.per_experiment_nb_cpus = 1 
```     

- `per_experiment_nb_nodes` Int. Number of nodes each job will get.    
```{.python}
cluster.per_experiment_nb_nodes = 1 
```     

- `gpu_type` String. Gpu type requested. Example:   
```{.python}
cluster.gpu_type = '1080ti'   
```     

------------------------------------------------------------------------

## Methods

### `set_checkpoint_save_function`

``` {.python}
cluster.set_checkpoint_save_function(fx, kwargs)    
```

Called if the model isn't finished training *minutes_to_checkpoint_before_walltime* before the walltime. If walltime = '15:00' and minutes_to_checkpoint_before_walltime = '1:00' the SlurmCluster will call your save function after 14 minutes of training.   

- ```fx``` A python function.  
- ```kwargs``` Dictionary where keys are the literal argument names to the function. Dictionary values are the values of the arguments.     

**Example**

``` {.python}
def my_save_function(arg_1, arg_k):  
    # ... save my model here    
    
cluster.set_checkpoint_save_function(my_save_function, kwargs={'arg_1': 'whatever', 'arg_k': 'you_want'})    

```

### `set_checkpoint_load_function`

``` {.python}
cluster.set_checkpoint_load_function(fx, kwargs)    
```

Called internally when a job is auto-submitted by the SlurmCluster to give your program a chance to load the model weights or whatever you need to continue training.  
This will call your load function immediately whenever you call this method AND training is continuing. 

- ```fx``` A python function.  
- ```kwargs``` Dictionary where keys are the literal argument names to the function. Dictionary values are the values of the arguments.   

**Example**

``` {.python}
def my_load_function(arg_1, arg_k):  
    # ... restore my model here    
    
cluster.set_checkpoint_save_function(my_load_function, kwargs={'arg_1': 'whatever', 'arg_k': 'you_want'})    

```

### `add_slurm_cmd`

``` {.python}
cluster.add_slurm_cmd(cmd, value, comment)
```

Adds whatever Slurm command you need manually to the generated script. All possible commands are listed [here](https://slurm.schedmd.com/pdfs/summary.pdf).

- ```cmd``` String with the bash command.   
- ```value``` String value for the command. Numericals need to be in single quotes ```'1'```  
- ```comment``` String with the command comment.  

**Example**

``` {.python}
cluster.add_slurm_cmd(cmd='cpus-per-task', value='1', comment='nb cpus per task')

# the above command will add an entry like this to the slurm script   

# #nb cpus per task
# #SBATCH --cpus-per-task=1
# ############

```    

### `add_command`

``` {.python}
cluster.add_command(cmd)    
```

Adds arbitrary bash commands to the script. Use this to activate conda environments, install packages, whatever else you would think about calling on bash.    

- ```cmd``` String with your bash command.   

**Example**


``` {.python}
# load the anaconda package on the launch node   
cluster.add_command('module load anaconda')   

# activate the environment on the launch node   
cluster.add_command('source activate myCondaEnv')   
```   

### `load_modules`

``` {.python}
cluster.load_modules(modules)  
```

Loads modules needed to run the job. Your Slurm documentation should have a list of available modules. You can also get those by running ```module avail```.   
- ```modules``` Array of module names.    

**Example**


``` {.python}
cluster.load_modules([
    'python-3',
    'anaconda3'
])   
```   

### `notify_job_status`

``` {.python}
cluster.notify_job_status(email, on_done, on_fail)  
```

Loads modules needed to run the job. Your Slurm documentation should have a list of available modules. You can also get those by running ```module avail```.   

- ```email``` String. Email address to get notifications.       
- ```on_done``` Boolean. If true, you'll get an email when the job completes.      
- ```on_fail``` Boolean. If true, you'll get an email if the job fails.    

**Example**


``` {.python}
cluster.notify_job_status(email='some@email.com', on_done=True, on_fail=True)   
```   

### `optimize_parallel_cluster_gpu`

``` {.python}
cluster.optimize_parallel_cluster_gpu(train_function, nb_trials, job_name, job_display_name=None)  
```

Launches the hyperparameter search across the cluster nodes.      
- ```train_function``` The entry point to start your training routine.   
- ```nb_trials``` Number of trials to launch. This is the number of hyperparameter configurations to train over.   
- ```job_name``` Folder name where the slurm scripts will save to. This should be the same as your [Experiment](https://williamfalcon.github.io/test-tube/experiment_tracking/experiment/) name.      
- ```job_display_name``` Visible name when slurm lists running jobs (ie: through ```squeue -u user_name```).  This should be really short (if using a test tube Experiment, it'll put the experiment version at the end).   

**Example**


``` {.python}
def main(hparams, cluster, return_dict):   
    # do your own generic training code here... 
    # init model
    model = model_build(hparams)    
    
    # set the load and save fxs
    cluster.set_checkpoint_save_function(fx, {})
    cluster.set_checkpoint_load_function(fx, {})
    
    # train ...
    

cluster.optimize_parallel_cluster_gpu(main, nb_trials=20, job_name='my_job', job_display_name='mj')    
```    

Now if you get the job information, you'll see this:    
``` {.bash}   
(conda_env) [user@node dir]$ squeue -u my_name
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            104040       all  mjv0   my_name  R      58:22      1 nodeName
            104041       all  mjv1   my_name  R      58:22      1 nodeName
            104042       all  mjv2   my_name  R      58:22      1 nodeName
            104043       all  mjv3   my_name  R      58:22      1 nodeName
```    

### `optimize_parallel_cluster_cpu`

``` {.python}
cluster.optimize_parallel_cluster_cpu(train_function, nb_trials, job_name, job_display_name=None)  
```

Launches the hyperparameter search across the cluster nodes using cpus.      
- ```train_function``` The entry point to start your training routine.   
- ```nb_trials``` Number of trials to launch. This is the number of hyperparameter configurations to train over.   
- ```job_name``` Folder name where the slurm scripts will save to. This should be the same as your [Experiment](https://williamfalcon.github.io/test-tube/experiment_tracking/experiment/) name.      
- ```job_display_name``` Visible name when slurm lists running jobs (ie: through ```squeue -u user_name```).  This should be really short (if using a test tube Experiment, it'll put the experiment version at the end).   

**Example**


``` {.python}
def main(hparams, cluster, return_dict):   
    # do your own generic training code here... 
    # init model
    model = model_build(hparams)    
    
    # set the load and save fxs
    cluster.set_checkpoint_save_function(fx, {})
    cluster.set_checkpoint_load_function(fx, {})
    
    # train ...
    

cluster.optimize_parallel_cluster_cpu(main, nb_trials=20, job_name='my_job', job_display_name='mj')    
```    

Now if you get the job information, you'll see this:    
``` {.bash}   
(conda_env) [user@node dir]$ squeue -u my_name
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
            104040       all  mjv0   my_name  R      58:22      1 nodeName
            104041       all  mjv1   my_name  R      58:22      1 nodeName
            104042       all  mjv2   my_name  R      58:22      1 nodeName
            104043       all  mjv3   my_name  R      58:22      1 nodeName
```    
