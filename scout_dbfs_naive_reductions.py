# %%
dummy = False

# %%
import apricot
import numpy as np
import scipy
import pandas as pd
import json
import os
import random
import uuid
import pickle
import time
import autosklearn.classification

from CSSPy.volume_sampler import k_Volume_Sampling_Sampler
from CSSPy.doublephase_sampler import double_Phase_Sampler
from CSSPy.largest_leveragescores_sampler import largest_leveragescores_Sampler
from CSSPy.dataset_tools import calculate_right_eigenvectors_k_svd
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import margin_sampling, entropy_sampling, uncertainty_sampling
from modAL.expected_error import expected_error_reduction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.naive_bayes import GaussianNB

def submodular_function_optimization(X, Y, **kwargs):
    """
    Submodular function optimization for row/column sampling
    does not require labeled dataset

    Required params:
    - model: fls or fbs
    """
    if kwargs["dir"] == 'col':
        X = X.transpose()
    if X.shape[0] == 1:
        return [0, ]
    tokeep = max(1, int(X.shape[0] * kwargs['keepFrac']))
    if kwargs['model'] == 'fls':
        clf = apricot.FacilityLocationSelection(tokeep).fit(X)
    elif kwargs['model'] == 'fbs':
        clf = apricot.FeatureBasedSelection(tokeep).fit(X)
    else:
        return None
    toret = clf.ranking
    selected = [int(i) for i in toret]
    if len(selected) > tokeep:
        selected = selected[:tokeep]
    return selected


def subset_selection_problem(X, Y, **kwargs):
    """
    Subset selection to optimize for the span of selected rows/columns
    does not require labeled dataset

    Required params:
    - sampler: volume, doublePhase or leverage
    """
    if kwargs["dir"] == 'row':
        X = X.transpose()

    if X.shape[0] <= 2 or X.shape[1] <= 2:
        return [x for x in range(X.shape[1])]

    tokeep = int(X.shape[0] * kwargs['keepFrac'])
    tokeep = min(tokeep, X.shape[0] - 1)
    tokeep = min(tokeep, X.shape[1] - 1)
    tokeep = max(2, tokeep)

    d = np.shape(X)[1]
    N = np.shape(X)[0] - 1
    # _, D, V = np.linalg.svd(X)
    _, D, V = scipy.linalg.svd(X)
    V_k = calculate_right_eigenvectors_k_svd(X, tokeep)

    if kwargs['sampler'] == 'volume':
        NAL = k_Volume_Sampling_Sampler(X, tokeep, D, V, d)
        A_S = NAL.MultiRounds()
    elif kwargs['sampler'] == 'doublePhase':
        NAL = double_Phase_Sampler(X, tokeep, V_k, d, 10*tokeep)
        A_S = NAL.DoublePhase()
    elif kwargs['sampler'] == 'leverage':
        NAL = largest_leveragescores_Sampler(X, tokeep, V, d)
        A_S = NAL.MultiRounds()
    
    selected = [int(x) for x in NAL.selected]
    if len(selected) > tokeep:
        selected = selected[:tokeep]
    return selected


def active_learning(X, Y, **kwargs):
    """
    Semi-supervised learning based sample selection
    """
    selected = list()
    tokeep = int(X.shape[0] * kwargs["keepFrac"])
    model = kwargs["model"]
    if model == "RF":
        d2d = RandomForestClassifier()
    elif model == "MLP":
        d2d = MLPClassifier()
    elif model == "KNN":
        d2d = KNeighborsClassifier(n_neighbors=5)

    # sample initial points
    n_initial = max(int(kwargs["initFrac"] * tokeep), 1)
    initial_idx = np.random.choice(range(X.shape[0]), size=n_initial, replace=False)
    selected.extend(initial_idx.tolist())
    X_initial = X[initial_idx]
    Y_initial = Y[initial_idx]
    X_pool = np.delete(X, initial_idx, axis=0)
    Y_pool = np.delete(Y, initial_idx, axis=0)

    # construct learner
    strat = kwargs["strat"]
    if strat == "margin":
        samp = margin_sampling
    elif strat == "entropy":
        samp = entropy_sampling
    elif strat == "uncertain":
        samp = uncertainty_batch_sampling
    elif strat == "expected":
        samp = expected_error_reduction
    learner = ActiveLearner(estimator=d2d, query_strategy=samp, X_training=X_initial, y_training=Y_initial)

    n_queries = 10
    n_instances = int((tokeep  - n_initial)/n_queries) + 1

    for _ in range(n_queries):
        query_idx, _ = learner.query(X_pool, n_instances=n_instances)
        learner.teach(X=X_pool[query_idx], y=Y_pool[query_idx], only_new=True)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        Y_pool = np.delete(Y_pool, query_idx, axis=0)
        selected.extend(query_idx)
    
    return [int(x) for x in selected][:tokeep]


def sampling_based_reduction(X, Y, **kwargs):
    """
    Returns list of selected col/row index
    """
    mapper = {
        "smf": submodular_function_optimization,
        "ssp": subset_selection_problem,
        "al": active_learning
    }

    return mapper[kwargs["method"]](X, Y, **kwargs)


# helper function for percentiles used in pandas
# https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
def get_percentile_fun(pct):

    def percentile_(x):
        return np.percentile(x, pct)
    percentile_.__name__ = "percentile_{}".format(pct)

    return percentile_


def aggregation_based_reduction(X, **kwargs):
    """
    Returns aggregated matrix X
    """

    # options for aggregation
    # 1: only common aggregation functions
    # 2: in addition to 1, add 4 percentiles
    # 3: in addition to 1 & 2, add 2 percentiles
    aggregation_options = {
        # function set 1: common
        1: [np.amax, np.amin, np.median, np.average, np.std],

        # function set 2: percentiles
        2: [10, 25, 75, 90],

        # function set 3: more detailed percentiles
        3: [1, 99]
    }

    aggregation_option_labels = {
        1: ["max", "min", "median", "avg", "std"],
        2: ["pct10", "pct25", "pct75", "pct90"],
        3: ["pct1", "pct99"]
    }

    if kwargs["dir"] == "col":
        processed, processed_labels = list(), list()
        for agg_option in range(1, kwargs["option"] + 1):
            funcs, func_labels = aggregation_options[agg_option], aggregation_option_labels[agg_option]
            if agg_option == 1:
                for func, func_label in zip(funcs, func_labels):
                    agg_stat = func(X.values, axis=1)
                    processed.append(agg_stat)
                    processed_labels.append(func_label)
            else:
                agg_stat = np.percentile(a=X.values, q=funcs, axis=1).transpose()
                processed.append(agg_stat)
                processed_labels.extend(func_labels)
        output_values = np.vstack(processed).transpose()
        
        if kwargs["prefix"] is not None:
            processed_labels = [kwargs["prefix"] + "_" + x for x in processed_labels]
        
        return pd.DataFrame(data=output_values, columns=processed_labels)
    
    else:  # row aggregation
        grb_criteria = kwargs["grb"]
        grb = X.groupby(by=grb_criteria)
        functions_to_apply = list()
        for agg_option in range(1, kwargs["option"] + 1):
            if agg_option == 1:
                functions_to_apply.extend(aggregation_options[agg_option])
            else:
                functions_to_apply.extend([
                    get_percentile_fun(x) for x in aggregation_options[agg_option]
                ])
        return grb.agg(functions_to_apply)

scout_data_dir = "data/scout"
scout_azure_dbfs_dir = "/dbfs/user/weifan/"
scout_device_health_dir = "device_health_data"
scout_azure_device_health_dir = os.path.join(scout_azure_dbfs_dir, scout_device_health_dir)
scout_dummy_device_health_dir = os.path.join(scout_data_dir, scout_device_health_dir)
scout_guided_reduction_save_dir = "scout_guided_reduction"
scout_naive_reduction_save_dir = "scout_unstructured_reduction"
scout_dummy_label_path = os.path.join(scout_data_dir, "labels.csv")
scout_automl_evaluate_dir = "scout_automl_evaluate"
scout_dummy_automl_eval_dir = os.path.join(scout_data_dir, scout_automl_evaluate_dir)
scout_dbfs_automl_eval_dir = os.path.join(scout_azure_dbfs_dir, scout_automl_evaluate_dir)

if os.path.isdir(scout_data_dir):
    os.system("mkdir -p {}".format(scout_dummy_automl_eval_dir))

if os.path.isdir(scout_azure_dbfs_dir):
    os.system("mkdir -p {}".format(scout_dbfs_automl_eval_dir))

scout_automl_time = 180
scout_automl_mem = 30000

scout_entity_types = ['cluster_switch', 'switch', 'tor_switch', ]
scout_tiers = {
    "cluster_switch": (0, 1),
    "switch": (0, 1, 2, 3),
    "tor_switch": (0, )
}
scout_metadata = ['IncidentId', 'EntityType', 'Tier']

reduction_strengths = [0.2, 0.4, 0.6, 0.8]

random.seed(10)
np.random.seed(10)

def load_raw_incident_device_health_reports(dummy=False):
    if dummy:
        source_dir = scout_dummy_device_health_dir
        ret_df = pd.read_csv("data/scout/scout_anonymized_raw_data_compressed.gz", index_col=0)
        ret_df[[x for x in ret_df.columns if x not in scout_metadata]] = np.absolute(ret_df[[x for x in ret_df.columns if x not in scout_metadata]].values)
        return ret_df
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
        # generate random labels on the fly
        generated = list()
        for df in df_list:
            generated.append(np.random.choice([0, 1], size=df.shape[0], replace=True))
        return generated

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
        extracted_labels.append(np.array([label_dict.get(x, -1) for x in df.IncidentId.values]))

    return extracted_labels


def convert_severity_level_to_label(severity):
    severity = int(severity)
    if severity <= 3:
        return 0
    else:
        return 1


def scout_load_severity(df_list, dummy=False):
    if dummy:
        return None
    else:
        severity_fpath = os.path.join(scout_azure_dbfs_dir, "incident_severity.csv")
        label_df = pd.read_csv(severity_fpath)
        label_df['Label'] = label_df.apply(lambda row: convert_severity_level_to_label(row['severity'], axis=1))
    
    label_dict = dict()
    for _, row in label_df.iterrows():
        label_dict[row["IncidentId"]] = int(row["Label"])
    
    extracted_labels = list()
    for df in df_list:
        extracted_labels.append(np.array([label_dict.get(x, -1) for x in df.IncidentId.values]))

    return extracted_labels


def add_missing_cols_to_test(train_df, test_df):
    missing_cols = [x for x in train_df.columns if x not in test_df.columns]
    for missing_col in missing_cols:
        test_df[missing_col] = None
    test_df = test_df[train_df.columns]
    return test_df

def scout_evaluator(scout_savepath, output_save_path=None, objective='incidentRouting'):

    with open(scout_savepath, "rb") as fin:
        train_df, test_df = pickle.load(fin)

    start_time = time.time()
    
    val_cols = [x for x in train_df.columns if x not in scout_metadata]
    
    num_row, num_col = int(train_df[val_cols].shape[0]), int(train_df[val_cols].shape[1])
    num_null = int(train_df[val_cols].isna().values.sum())
    
    test_df = add_missing_cols_to_test(train_df, test_df)

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # check if Row Aggregation is applied
    if 'RowAgg' not in os.path.basename(scout_savepath):
        # not applied, apply row aggregation
        # with No granularity
        # and 1 as aggregation function option

        train_test_processed = list()
        grb_cols = ['IncidentId', ] + [x for x in train_df.columns if x not in scout_metadata]
        for raw_df in [train_df, test_df,]:
            aggregated_result = aggregation_based_reduction(raw_df[grb_cols], dir="row", grb='IncidentId', option=1)
            rename_col = list()
            for old_col in aggregated_result.columns:
                rename_col.append(":".join(old_col))
            aggregated_result.columns = rename_col
            aggregated_result.reset_index(inplace=True)
            metrics_left = [x for x in aggregated_result.columns if x not in scout_metadata]
            aggregated_result = aggregated_result[['IncidentId', ] + metrics_left]
            train_test_processed.append(aggregated_result)
        
        train_df, test_df = train_test_processed
        metadata_gran = None
    
    else:
        # applied
        metadata_gran = os.path.basename(scout_savepath).split('RowAgg_')[1].split("_")[0]
        if metadata_gran == 'None':
            metadata_gran = None
        elif "+" in metadata_gran:
            metadata_gran = metadata_gran.split("+")

    if metadata_gran is None:
        train_data_vectors, test_data_vectors = train_df, test_df
    else:
        vectorized = list()
        for data_df in [train_df, test_df]:
            grb_gran = data_df.groupby(metadata_gran)
            renamed_df = None
            for key, sub_df in grb_gran:

                if type(key) == str:
                    suffix = key
                else:
                    suffix = "+".join(key)

                new_col_names = list()
                for col_name in sub_df.columns:
                    if col_name in scout_metadata:
                        new_col_names.append(col_name)
                    else:
                        new_col_names.append("{}({})".format(col_name, suffix))
                
                sub_df.columns = new_col_names
                sub_df = sub_df[['IncidentId',] + [x for x in new_col_names if x not in scout_metadata]]
                if renamed_df is None:
                    renamed_df = sub_df
                else:
                    renamed_df = renamed_df.merge(sub_df, how='left', on='IncidentId')
            vectorized.append(renamed_df)
    
        train_incident_ids = set(train_df.IncidentId.values)
        test_incident_ids = set(test_df.IncidentId.values)

        vectorized_df = pd.concat(vectorized, axis=0)
        train_data_vectors = vectorized_df[vectorized_df.IncidentId.isin(train_incident_ids)]
        test_data_vectors = vectorized_df[vectorized_df.IncidentId.isin(test_incident_ids)]
    
    if objective == 'incidentRouting':
        train_label, test_label = scout_load_labels([train_data_vectors, test_data_vectors], dummy=dummy)
    elif objective == 'severityPrediction':
        train_label, test_label = scout_load_severity([train_data_vectors, test_data_vectors], dummy=dummy)

    pred_columns = [x for x in train_data_vectors.columns if x not in scout_metadata]

    # train AutoML evaluator with fixed budget
    train_vals = train_data_vectors[pred_columns].values
    test_vals = test_data_vectors[pred_columns].values

    # in case if there are missing labels
    train_vals, train_label = train_vals[train_label != -1], train_label[train_label != -1]
    test_vals, test_label = test_vals[test_label != -1], test_label[test_label != -1]

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=scout_automl_time,
        n_jobs=8,
        per_run_time_limit=scout_automl_time//6,
        memory_limit=scout_automl_mem
    )

    automl.fit(train_vals, train_label)
    test_pred = automl.predict(test_vals).astype(int)

    eval_result = {
        "num_row": num_row,
        "num_col": num_col,
        "num_null": num_null,
        "accuracy": accuracy_score(test_label, test_pred),
        "balanced_accuracy": balanced_accuracy_score(test_label, test_pred),
        "f1": f1_score(test_label, test_pred),
        "precision": precision_score(test_label, test_pred),
        "recall": recall_score(test_label, test_pred),
        "eval_time": float(time.time() - start_time)
    }
    
    if output_save_path is not None:
        with open(output_save_path, "w") as fout:
            json.dump(eval_result, fout, indent=2)
    else:
        return eval_result

# %%
##### SCOUT NAIVE TWO-HOP REDUCTIONS #####
save_dir = scout_naive_reduction_save_dir

if dummy:
    one_hop_out_dir = os.path.join(scout_data_dir, save_dir, "one_hop")
else:
    one_hop_out_dir = os.path.join(scout_azure_dbfs_dir, save_dir, "one_hop")
os.system("mkdir -p {}".format(one_hop_out_dir))

best_one_hop = 'ColSampling_None_col-0.6-smf-fls_sec21.96036'
boh_path = os.path.join(one_hop_out_dir, "{}.pickle".format(best_one_hop))

with open(boh_path, "rb") as fin:
    train_df, test_df = pickle.load(fin)

if dummy:
    two_hop_out_dir = os.path.join(scout_data_dir, save_dir, "two_hop")
else:
    two_hop_out_dir = os.path.join(scout_azure_dbfs_dir, save_dir, "two_hop")
os.system("mkdir -p {}".format(two_hop_out_dir))

# Row aggregation
agg_cols = ['IncidentId', ] + [x for x in train_df.columns if x not in scout_metadata]
# for option in [1, 2, 3]:
for option in [1, ]:
    str_desc = get_str_desc_of_reduction_function("RowAgg", None, dir="row", grb="IncidentId", option=option)
    th_desc = best_one_hop + "&" + str_desc
    if not if_file_w_prefix_exists(two_hop_out_dir, th_desc):
        start_time = time.time()
        train_agg_result = aggregation_based_reduction(train_df[agg_cols], dir='row', grb='IncidentId', option=option)
        end_time = time.time()
        time_taken = round(end_time - start_time, 5)
        test_agg_result = aggregation_based_reduction(test_df[agg_cols], dir='row', grb='IncidentId', option=option)
        renamed_dfs = list()
        for df in [train_agg_result, test_agg_result]:
            rename_cols = [":".join(x) for x in df.columns]
            df.columns = rename_cols
            df.reset_index(inplace=True)
            renamed_dfs.append(df)
        
        train_save, test_save = renamed_dfs
        save_file_name = os.path.join(two_hop_out_dir, "{}_sec{}.pickle".format(th_desc, time_taken))
        with open(save_file_name, "wb") as fout:
            pickle.dump((train_save, test_save), fout)

# %%
##### Run simulated ML exploration on naively reduced scout data ######
def scout_convert_to_feature_vector_max(scout_savepath, granularity=None, objective='incidentRouting'):
    with open(scout_savepath, "rb") as fin:
        train_df, test_df = pickle.load(fin)
    
    val_cols = [x for x in train_df.columns if x not in scout_metadata]
    
    num_row, num_col = int(train_df[val_cols].shape[0]), int(train_df[val_cols].shape[1])
    num_null = int(train_df[val_cols].isna().values.sum())
    
    test_df = add_missing_cols_to_test(train_df, test_df)

    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    metadata_gran = granularity

    if metadata_gran is None:
        train_data_vectors, test_data_vectors = train_df, test_df
    else:
        vectorized = list()
        for data_df in [train_df, test_df]:
            grb_gran = data_df.groupby(metadata_gran)
            renamed_df = None
            for key, sub_df in grb_gran:

                if type(key) == str:
                    suffix = key
                else:
                    suffix = "+".join(key)

                new_col_names = list()
                for col_name in sub_df.columns:
                    if col_name in scout_metadata:
                        new_col_names.append(col_name)
                    else:
                        new_col_names.append("{}({})".format(col_name, suffix))
                
                sub_df.columns = new_col_names
                sub_df = sub_df[['IncidentId',] + [x for x in new_col_names if x not in scout_metadata]]
                if renamed_df is None:
                    renamed_df = sub_df
                else:
                    renamed_df = renamed_df.merge(sub_df, how='left', on='IncidentId')
            vectorized.append(renamed_df)
    
        train_incident_ids = set(train_df.IncidentId.values)
        test_incident_ids = set(test_df.IncidentId.values)

        vectorized_df = pd.concat(vectorized, axis=0)
        train_data_vectors = vectorized_df[vectorized_df.IncidentId.isin(train_incident_ids)]
        test_data_vectors = vectorized_df[vectorized_df.IncidentId.isin(test_incident_ids)]
    
    if objective == 'incidentRouting':
        train_label, test_label = scout_load_labels([train_data_vectors, test_data_vectors], dummy=dummy)
    elif objective == 'severityPrediction':
        train_label, test_label = scout_load_severity([train_data_vectors, test_data_vectors], dummy=dummy)

    pred_columns = [x for x in train_data_vectors.columns if x not in scout_metadata]

    train_vals = train_data_vectors[pred_columns].fillna(0).values
    test_vals = test_data_vectors[pred_columns].fillna(0).values

    return train_vals, train_label, test_vals, test_label


def get_one_model_result(model, X_train, Y_train, X_test, Y_test):
   start_time = time.time()
   model.fit(X_train, Y_train)
   end_time = time.time()
   train_time = end_time - start_time

   start_time = time.time()
   Y_test_pred = model.predict(X_test)
   end_time = time.time()
   infer_time = end_time - start_time

   acc = accuracy_score(Y_test, Y_test_pred)
   balanced_acc = balanced_accuracy_score(Y_test, Y_test_pred)
   f1 = f1_score(Y_test, Y_test_pred)

   record = {
      "train_time": float(train_time),
      "infer_time": float(infer_time),
      "acc": acc,
      "balanced_acc": balanced_acc,
      "f1": f1
   }

   return record


def train_and_evaluate_all_models(X_train, Y_train, X_test, Y_test, out_dir):

   # Random Forest
   for nt in [50, 100, 500, 1000]:
      out_name = os.path.join(out_dir, "rf_nt{}.json".format(nt))
      print(out_name)
      if not os.path.isfile(out_name):
         model = RandomForestClassifier(n_estimators=nt, random_state=10)
         record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)
         
         with open(out_name, "w") as fout:
            json.dump(record, fout, indent=2)

   # MLP
   for hidden_layer_size in [10, 20, 30, 50]:
      for activation in ['logistic', 'tanh', 'relu']:
         for max_iter in [200, 500, 1000]:
            out_name = os.path.join(out_dir, "mlp_size{}_iter{}_loss{}.json".format(hidden_layer_size, max_iter, activation))
            print(out_name)
            if not os.path.isfile(out_name):
               model = MLPClassifier(
                  hidden_layer_sizes=(hidden_layer_size, hidden_layer_size),
                  activation=activation,
                  n_iter_no_change=max_iter,
                  max_iter=max_iter
               )
               record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)

               with open(out_name, "w") as fout:
                  json.dump(record, fout, indent=2)

   # Logistic Regression
   for max_iter in [50, 100, 500, 1000]:
      out_name = os.path.join(out_dir, "lr_iter{}.json".format(max_iter))
      print(out_name)
      if not os.path.isfile(out_name):
         model = LogisticRegression(max_iter=max_iter)
         record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)

         with open(out_name, "w") as fout:
            json.dump(record, fout, indent=2)

   # KNN
   for n_neighbors in [3, 5, 10, 20]:
      out_name = os.path.join(out_dir, "knn_nn{}.json".format(n_neighbors))
      print(out_name)
      if not os.path.isfile(out_name):
         model = KNeighborsClassifier(n_neighbors=n_neighbors)
         record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)

         with open(out_name, "w") as fout:
            json.dump(record, fout, indent=2)

   # Gaussian Process
   for max_iter in [50, 100, 500, 1000]:
      out_name = os.path.join(out_dir, "gp_iter{}.json".format(max_iter))
      print(out_name)
      if not os.path.isfile(out_name):
         model = GaussianProcessClassifier(max_iter_predict=max_iter)
         record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)

         with open(out_name, "w") as fout:
            json.dump(record, fout, indent=2)

   # Naive Bayes
   model = GaussianNB()
   out_name = os.path.join(out_dir, "nb.json".format(max_iter))
   print(out_name)
   if not os.path.isfile(out_name):
      record = get_one_model_result(model, X_train, Y_train, X_test, Y_test)

      with open(out_name, "w") as fout:
         json.dump(record, fout, indent=2)

# %%
naive_search_result = [x for x in os.listdir(two_hop_out_dir) if x.endswith(".pickle")][0]
if dummy:
    mle_dir = os.path.join(scout_data_dir, 'ml_exploration')
else:
    mle_dir = os.path.join(scout_azure_dbfs_dir, 'ml_exploration')
os.system('mkdir -p {}'.format(mle_dir))
os.system('cp \"{}\" {}'.format(os.path.join(two_hop_out_dir, naive_search_result), os.path.join(mle_dir, "naive.pickle")))

for name in ['naive']:
    if name == 'greedy':
        gran = 'Tier'
    elif name == 'baseline':
        gran = 'EntityType'
    elif name == 'naive':
        gran = None
    else:
        gran = ['EntityType', 'Tier']
    X_train, Y_train, X_test, Y_test = scout_convert_to_feature_vector_max(
        os.path.join(mle_dir, "{}.pickle".format(name)),
        granularity=gran
    )
    X_train = X_train[Y_train != -1, :]
    Y_train = Y_train[Y_train != -1]
    X_test = X_test[Y_test != -1, :]
    Y_test = Y_test[Y_test != -1]
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    out_dir = os.path.join(mle_dir, name)
    os.system("mkdir -p {}".format(out_dir))
    train_and_evaluate_all_models(X_train, Y_train, X_test, Y_test, out_dir)


