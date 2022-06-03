from cProfile import label
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


philly_data_dir = "data/philly"


def get_philly_per_job_trace(jobid):

    cpu_traces = pd.read_csv(os.path.join(philly_data_dir, "cpu_util", "{}.csv".format(jobid)))
    gpu_traces = pd.read_csv(os.path.join(philly_data_dir, "gpu_util", "{}.csv".format(jobid)))
    mem_traces = pd.read_csv(os.path.join(philly_data_dir, "mem_util", "{}.csv".format(jobid)))

    return pd.concat([cpu_traces, gpu_traces, mem_traces], ignore_index=True)


def load_philly_data(test_size=0.2):

    status_to_num = {
        'Pass': 0,
        'Failed': 1,
        'Killed': 2
    }

    with open(os.path.join(philly_data_dir, "sampled_jobs.json"), "r") as fin:
        job_infos = json.load(fin)
    train_jobs, test_jobs = train_test_split(job_infos, test_size=test_size, random_state=10)
    pairs = list()

    for job_info in (train_jobs, test_jobs):
        job_dfs = list()
        label_mapping = dict()

        for job in job_info:
            label_mapping[job['jobid']] = status_to_num[job['status']]
            job_df = get_philly_per_job_trace(job['jobid'])
            job_df['jobid'] = job['jobid']
            job_dfs.append(job_df)
        
        pairs.append((pd.concat(job_dfs, ignore_index=True), label_mapping))
    
    metadatas = ['jobid', 'name', 'machine_type', 'trace']
    metrics = [x for x in pairs[0][0].columns if x not in metadatas]

    return pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1], metadatas, metrics
