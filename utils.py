from cProfile import label
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


philly_data_dir = "data/philly"
scout_data_dir = "data/scout"

scout_total_size = 312049 * 230

scout_azure_dbfs_dir = "/dbfs/user/weifan/"
scout_azure_device_health_reports = os.path.join(
    scout_azure_dbfs_dir, "device_health_data"
)

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


def get_scout_dataset_characteristics(filename):
    if filename.endswith(".pickle"):
        filename = filename.replace(".pickle", "")
    if "=" in filename:
        filename = filename.split("=")[0]
    return pd.read_csv(os.path.join(scout_data_dir, "dataset_characteristics", "{}.csv".format(filename)), index_col=0)


def parse_scout_filename(filename):
    if filename.endswith(".pickle"):
        filename = filename.replace(".pickle", "")
    tokens = filename.split("=")
    eval_time = int(tokens[1])
    hop_strs = tokens[0].split("&")
    parsed_reductions = list()
    for hop_str in hop_strs:
        sub_tokens = hop_str.split("_")
        reduce_time = int(sub_tokens[-1])
        reduce_method = "_".join(sub_tokens[0:-1])
        parsed_reductions.append((reduce_method, reduce_time))
    return {"reduction": parsed_reductions, "eval_time": eval_time}


def get_scout_hop_count(scout_reduction):
    return len(scout_reduction["reduction"])


def scout_is_prefix(reduction1, reduction2):
    red1_str = "&".join([x[0] for x in reduction1["reduction"]])
    red2_str = "&".join([x[0] for x in reduction2["reduction"]])
    return red2_str.startswith(red1_str)


def get_scout_all_next_hops(curr_reduction, all_reductions):
    next_length = get_scout_hop_count(curr_reduction) + 1
    candidates = [x for x in all_reductions if get_scout_all_next_hops(x) == next_length]
    return [x for x in candidates if scout_is_prefix(curr_reduction, x)]
    
