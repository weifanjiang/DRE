import pandas as pd
import os


philly_data_dir = "data/philly"

philly_label_dict = {
    "Pass": 0,
    "Killed": 1,
    "Failed": 2
}

philly_start_time = -10
philly_end_time = 3


def get_philly_per_job_trace(jobid):

    cpu_traces = pd.read_csv(os.path.join(philly_data_dir, "cpu_util", "{}.csv".format(jobid)))
    gpu_traces = pd.read_csv(os.path.join(philly_data_dir, "gpu_util", "{}.csv".format(jobid)))
    mem_traces = pd.read_csv(os.path.join(philly_data_dir, "mem_util", "{}.csv".format(jobid)))

    return [cpu_traces, gpu_traces, mem_traces]
