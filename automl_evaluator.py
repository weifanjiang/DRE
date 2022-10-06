from utils import *
from reductions import *

import autosklearn.classification
import pickle
import pandas as pd
import os
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

dummy = True


def scout_evaluator(scout_savepath, output_save_path=None):

    with open(scout_savepath, "rb") as fin:
        train_df, test_df = pickle.load(fin)
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)
    
    # check if Row Aggregation is applied
    if 'RowAgg' not in os.path.basename(scout_savepath):
        # not applied, apply row aggregation
        # with (IncidentId, EntityType, Tier) as granularity
        # and 3 as aggregation function option

        train_test_processed = list()
        grb_cols = ['IncidentId', ] + [x for x in train_df.columns if x not in scout_metadata]
        for raw_df in [train_df, test_df,]:
            grb_gran = raw_df.groupby(by=['EntityType', 'Tier'])
            processed = list()
            for keys, sub_df in grb_gran:
                aggregated_result = aggregation_based_reduction(
                    sub_df[grb_cols], dir="row", grb='IncidentId', option=3
                )
                rename_col = list()
                for old_col in aggregated_result.columns:
                    rename_col.append(":".join(old_col))
                aggregated_result.columns = rename_col
                aggregated_result.reset_index(inplace=True)
                aggregated_result['EntityType'] = keys[0]
                aggregated_result['Tier'] = keys[1]

                processed.append(aggregated_result)
            
            # reorganize columns
            to_save = pd.concat(processed, axis=0, ignore_index=True)
            metadata_left = [x for x in scout_metadata if x in ['EntityType', 'Tier']]
            metrics_left = [x for x in to_save.columns if x not in scout_metadata]
            to_save = to_save[['IncidentId', ] + metadata_left + metrics_left]
            
            train_test_processed.append(to_save)
        
        train_df, test_df = train_test_processed
        metadata_gran = ['EntityType', 'Tier']
    
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
    
    train_label, test_label = scout_load_labels([train_data_vectors, test_data_vectors], dummy=dummy)
    pred_columns = [x for x in train_data_vectors.columns if x not in scout_metadata]

    # train AutoML evaluator with fixed budget
    train_vals = train_data_vectors[pred_columns].values
    test_vals = test_data_vectors[pred_columns].values

    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=scout_automl_time,
        n_jobs=8,
        per_run_time_limit=scout_automl_time//6,
        memory_limit=scout_mem_limit
    )

    automl.fit(train_vals, train_label)
    test_pred = automl.predict(test_vals).astype(int)

    eval_result = {
        "accuracy": accuracy_score(test_label, test_pred),
        "balanced_accuracy": balanced_accuracy_score(test_label, test_pred),
        "f1": f1_score(test_label, test_pred),
        "precision": precision_score(test_label, test_pred),
        "recall": recall_score(test_label, test_pred) 
    }
    
    if output_save_path is not None:
        with open(output_save_path, "w") as fout:
            json.dump(eval_result, fout, indent=2)
    else:
        return eval_result
