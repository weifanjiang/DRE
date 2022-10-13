import pandas as pd
import json
import os
import numpy as np
import random
import uuid
import pickle

from sklearn.model_selection import train_test_split


philly_data_dir = "data/philly"
scout_data_dir = "data/scout"
gcut_data_dir = "data/gcut"

scout_total_size = 312049 * 230

scout_azure_dbfs_dir = "/dbfs/user/weifan/"
scout_device_health_dir = "device_health_data"

scout_azure_device_health_dir = os.path.join(scout_azure_dbfs_dir, scout_device_health_dir)
scout_dummy_device_health_dir = os.path.join(scout_data_dir, scout_device_health_dir)

scout_guided_reduction_save_dir = "scout_guided_reduction"
scout_naive_reduction_save_dir = "scout_unstructured_reduction"
gcut_naive_reduction_save_dir = "gcut_unstructured_reduction"
philly_naive_reduction_save_dir = "philly_unstructured_reduction"
philly_guided_reduction_save_dir = "philly_guided_reduction"

scout_dummy_label_path = os.path.join(scout_data_dir, "labels.csv")

scout_automl_evaluate_dir = "scout_automl_evaluate"
scout_dummy_automl_eval_dir = os.path.join(scout_data_dir, scout_automl_evaluate_dir)
scout_dbfs_automl_eval_dir = os.path.join(scout_azure_dbfs_dir, scout_automl_evaluate_dir)

if os.path.isdir(scout_data_dir):
    os.system("mkdir -p {}".format(scout_dummy_automl_eval_dir))

if os.path.isdir(scout_azure_dbfs_dir):
    os.system("mkdir -p {}".format(scout_dbfs_automl_eval_dir))

scout_automl_time = 100
scout_mem_limit = 20000

gcut_automl_time = 120
gcut_mem_limit = 300000

scout_entity_types = ['cluster_switch', 'switch', 'tor_switch', ]
scout_tiers = {
    "cluster_switch": (0, 1),
    "switch": (0, 1, 2, 3),
    "tor_switch": (0, )
}

philly_label_dict = {
    "Pass": 0,
    "Killed": 1,
    "Failed": 2
}

scout_metadata = ['IncidentId', 'EntityType', 'Tier']
gcut_metadata = ["TaskId", "Label", "TimeWindow"]
philly_metadata = ["JobId", "Label", "MachineType", "CPU", "GPU", "TraceType"]

philly_start_time = -10
philly_end_time = 3

reduction_strengths = [0.2, 0.4, 0.6, 0.8]


random.seed(10)
np.random.seed(10)


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


def load_device_health_report_columns():
    with open(os.path.join(scout_data_dir, 'device_healthy_data_cols.json')) as fin:
        cols = json.load(fin)
    return cols


def load_raw_incident_device_health_reports(dummy=False):
    if dummy:
        source_dir = scout_dummy_device_health_dir
    else:
        source_dir = scout_azure_device_health_dir
    
    all_csv_files= [x for x in os.listdir(source_dir) if x.endswith(".csv")]
    all_reports = list()
    for fname in all_csv_files:
        one_report = pd.read_csv(os.path.join(source_dir, fname))
        all_reports.append(one_report)
    
    report_df = pd.concat(all_reports, axis=0)
    dh_metric_cols = [x for x in report_df.columns[5:]]
    report_df['Tier'] = report_df.apply(
        lambda row: extract_tier_from_entity_name(row, dummy),
        axis=1
    )
    report_df = report_df[report_df['Tier'].isin(['t0', 't1', 't2', 't3'])]
    report_df.fillna(0, inplace=True)
    min_metric_val_abs = abs(np.amin(report_df[dh_metric_cols].values))
    report_df[dh_metric_cols] = report_df[dh_metric_cols].values + min_metric_val_abs
    report_df = report_df[
        ["IncidentId", "EntityType", "Tier",] + dh_metric_cols
    ]
    return report_df


def extract_tier_from_entity_name(row, dummy=False):
    if dummy:
        return "t{}".format(np.random.choice(scout_tiers[row["EntityType"]]))
    
    else:
        # sample: dsm06-0102-0130-07t0
        entity_name = row['EntityName']
        return "t" + entity_name.split("-")[-1].split('t')[-1]


def get_str_desc_of_reduction_function(method_str, granularity, **kwargs):

    if granularity is None:
        gran_str = 'None'
    elif type(granularity) == str:
        gran_str = granularity
    else:
        gran_str = "+".join(granularity)

    keys = sorted(list(kwargs.keys()))
    vals_joined = "-".join([str(kwargs[x]) for x in keys])
    return "{}_{}_{}".format(method_str, gran_str, vals_joined)


def if_file_w_prefix_exists(dir, prefix):
    existing_files = os.listdir(dir)
    for e_file in existing_files:
        if e_file.startswith(prefix):
            return True
    return False


def train_test_split_scout_data(raw_df, train_size):

    all_incident_ids = raw_df.IncidentId.unique()
    train_ids, test_ids = train_test_split(
        all_incident_ids,
        train_size=train_size,
        random_state=10
    )

    train_ids, test_ids = set(train_ids), set(test_ids)

    train_df = raw_df[raw_df.IncidentId.isin(train_ids)]
    test_df = raw_df[raw_df.IncidentId.isin(test_ids)]

    return train_df, test_df


def safe_get_subgroup(df_groupby, key):
    if key in df_groupby.groups:
        return df_groupby.get_group(key)
    return None


def scout_load_labels(df_list, dummy=False):

    if dummy:
        label_df = pd.read_csv(scout_dummy_label_path)
    else:
        label_paths = [x for x in os.listdir(scout_azure_dbfs_dir) if x.startswith("sampled_incidents_")]
        label_paths = [x for x in label_paths if x.endswith(".csv")]

        loaded = list()
        for label_path in label_paths:
            loaded.append(pd.read_csv(os.path.join(scout_azure_dbfs_dir, label_path))[['IncidentId', 'Label']])
        label_df = pd.concat(loaded, axis=0).reset_index(drop=True).drop_duplicates(ignore_index=True)
    
    label_dict = dict()
    for _, row in label_df.iterrows():
        label_dict[row["IncidentId"]] = int(row["Label"])
    
    extracted_labels = list()
    for df in df_list:
        extracted_labels.append([label_dict[x] for x in df.IncidentId.values])

    return extracted_labels


def add_missing_cols_to_test(train_df, test_df):
    missing_cols = [x for x in train_df.columns if x not in test_df.columns]
    for missing_col in missing_cols:
        test_df[missing_col] = None
    test_df = test_df[train_df.columns]
    return test_df


def load_google_dataset():

    if os.path.isfile("data/gcut/google/gcut_df.pickle"):
        with open("data/gcut/google/gcut_df.pickle", "rb") as fin:
            processed_df = pickle.load(fin)
    else:
        google_col_names = [
            'CPU_rate',
            'MEM_canonical_usage',
            'MEM_assigned_usage',
            'CACHE_unmapped_page',
            'CACHE_total_page',
            'MEM_maximum_usage',
            'DISK_local_space_usage',
            'CPU_maximum_rate',
            'CPU_sampled_usage'
        ]

        train_data = np.load("data/gcut/google/data_train.npz")
        test_data = np.load("data/gcut/google/data_test.npz")

        # only keep the first 50 measurement cycles
        train_mat = train_data["data_feature.npy"][:, :50, :]
        test_mat = test_data["data_feature.npy"][:, :50, :]

        train_label = np.argmax(train_data["data_attribute.npy"], axis=1)
        test_label = np.argmax(test_data["data_attribute.npy"], axis=1)

        processed_df = list()
        
        for feature_mat, labels in [(train_mat, train_label), (test_mat, test_label)]:

            processed_tasks = list()
            for i in range(feature_mat.shape[0]):
                task_mat = feature_mat[i]
                task_df = pd.DataFrame(task_mat, columns=google_col_names)
                task_df["TaskId"] = str(uuid.uuid1())
                task_df["TimeWindow"] = np.linspace(0, 49, 50).astype(int)
                task_df["Label"] = labels[i]

                processed_tasks.append(task_df)
            
            processed_df.append(pd.concat(processed_tasks, axis=0, ignore_index=True))
        
        with open("data/gcut/google/gcut_df.pickle", "wb") as fout:
            pickle.dump(processed_df, fout)

    return processed_df

def is_valid(row):
    keys = row.keys()
    for key in keys:
        if key not in philly_metadata:
            try:
                float(row[key])
            except:
                return False
            if float(row[key]) < 0:
                return False
    return True


def load_philly_dataset():
    
    with open(os.path.join(philly_data_dir, "sampled_jobs.json"), "r") as fin:
        sampled_jobs = json.load(fin)
    
    cpu_dfs, gpu_dfs, mem_dfs = list(), list(), list()
    for job in sampled_jobs:
        jobid = job["jobid"]
        cpu = pd.read_csv(os.path.join(philly_data_dir, "cpu_util", "{}.csv".format(jobid)))
        gpu = pd.read_csv(os.path.join(philly_data_dir, "gpu_util", "{}.csv".format(jobid)))
        mem = pd.read_csv(os.path.join(philly_data_dir, "mem_util", "{}.csv".format(jobid)))

        for df in [cpu, gpu, mem]:
            df["JobId"] = jobid
            df["Label"] = int(job["status"] == 'Pass')
        
        cpu_dfs.append(cpu)
        gpu_dfs.append(gpu)
        mem_dfs.append(mem)

    cpu_df = pd.concat(cpu_dfs, ignore_index=True, axis=0)
    gpu_df = pd.concat(gpu_dfs, ignore_index=True, axis=0)
    mem_df = pd.concat(mem_dfs, ignore_index=True, axis=0)

    # fix schema
    cpu_df.rename(columns={"name": "CPU", "trace": "TraceType", "machine_type": "MachineType"}, inplace=True)
    cpu_df["GPU"] = None

    gpu_df["CPU"] = gpu_df.apply(lambda row: row["name"].split("_")[0], axis=1)
    gpu_df["GPU"] = gpu_df.apply(lambda row: row["name"].split("_")[1], axis=1)
    gpu_df.drop(columns=["name", ], inplace=True)
    gpu_df.rename(columns={"trace": "TraceType", "machine_type": "MachineType"}, inplace=True)

    mem_df.rename(columns={"name": "CPU", "trace": "TraceType", "machine_type": "MachineType"}, inplace=True)
    mem_df["GPU"] = None

    # join all traces
    philly_df = pd.concat([cpu_df, gpu_df, mem_df], axis=0, ignore_index=True)

    # fix column order
    philly_metrics = sorted([x for x in philly_df.columns if x not in philly_metadata])
    reorder_cols = philly_metadata + philly_metrics

    # remove invalid rows
    valid = philly_df.apply(lambda row: is_valid(row), axis=1)
    philly_df = philly_df[valid]

    # fill na
    philly_df[philly_metrics] = philly_df[philly_metrics].fillna(0)

    return philly_df[reorder_cols]


def train_test_split_philly_data(raw_df, train_size=0.7):

    no_dup_df = raw_df[['JobId', 'Label']].drop_duplicates(ignore_index=True)
    jobids = no_dup_df.JobId.values
    labels = no_dup_df.Label.values
    job_id_train, job_id_test = train_test_split(jobids, stratify=labels, random_state=10, train_size=train_size)

    job_id_train = set(job_id_train)
    job_id_test = set(job_id_test)

    train_df = raw_df[raw_df.JobId.isin(job_id_train)]
    test_df = raw_df[raw_df.JobId.isin(job_id_test)]

    return train_df, test_df


def load_train_test_split_philly_dataset():

    train_path = os.path.join(philly_data_dir, "philly_train.csv")
    test_path = os.path.join(philly_data_dir, "philly_test.csv")
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    else:
        train_df, test_df = train_test_split_philly_data(load_philly_dataset())
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
    
    return train_df, test_df
