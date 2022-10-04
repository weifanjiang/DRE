from utils import *
from reductions import *

import os
import time
import pickle


# dummy = False: running on Azure workspace
dummy = True
save_dir = scout_naive_reduction_save_dir


scout_raw_df = load_raw_incident_device_health_reports(dummy=dummy)
scout_raw_df, test_df = train_test_split_scout_data(scout_raw_df, 0.8)


# one hop naive reductions
# for one-hop reduction, only consider col sampling and aggregation based method
# since real scout data has too much rows to be sampled without aggregated
if dummy:
    one_hop_out_dir = os.path.join(
        scout_data_dir,
        save_dir,
        "one_hop"
    )
else:
    one_hop_out_dir = os.path.join(
        scout_azure_dbfs_dir,
        save_dir,
        "one_hop"
    )
os.system("mkdir -p {}".format(one_hop_out_dir))

# SMF
smf_cols_to_reduce = [x for x in scout_raw_df if x not in scout_metadata]
smf_input_mat = scout_raw_df[smf_cols_to_reduce].values
for keepFrac in reduction_strengths:
    for model in ['fls', 'fbs']:

        str_desc = get_str_desc_of_reduction_function(
            "ColSampling",
            None,
            method='smf',
            dir="col",
            model=model,
            keepFrac=keepFrac
        )

        # check if file exists in directory
        if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):
            
            start_time = time.time()

            selected_idx = sampling_based_reduction(
                smf_input_mat,
                None,
                method='smf',
                dir="col",
                model=model,
                keepFrac=keepFrac
            )

            end_time = time.time()
            time_taken = int(end_time - start_time)

            selected_columns = [smf_cols_to_reduce[x] for x in selected_idx]

            
            save_file_name = os.path.join(
                one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
            )

            to_save = scout_raw_df[scout_metadata + selected_columns]
            to_save_test = test_df[scout_metadata + selected_columns]

            with open(save_file_name, "wb") as fout:
                pickle.dump((to_save, to_save_test), fout)


# SSP
ssp_cols_to_reduce = [x for x in scout_raw_df if x not in scout_metadata]
ssp_input_mat = scout_raw_df[ssp_cols_to_reduce].values
for keepFrac in reduction_strengths:
    for sampler in ['volume', 'doublePhase', 'leverage']:

        str_desc = get_str_desc_of_reduction_function(
            "ColSampling",
            None,
            method='ssp',
            dir="col",
            sampler=sampler,
            keepFrac=keepFrac
        )

        # check if file exists in directory
        if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):

            start_time = time.time()

            selected_idx = sampling_based_reduction(
                ssp_input_mat,
                None,
                method='ssp',
                dir="col",
                sampler=sampler,
                keepFrac=keepFrac
            )

            end_time = time.time()
            time_taken = int(end_time - start_time)

            selected_columns = [ssp_cols_to_reduce[x] for x in selected_idx]

            
            save_file_name = os.path.join(
                one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
            )

            to_save = scout_raw_df[scout_metadata + selected_columns]
            to_save_test = test_df[scout_metadata + selected_columns]

            with open(save_file_name, "wb") as fout:
                pickle.dump((to_save, to_save_test), fout)


# Row aggregation
cols_to_aggregate = [x for x in scout_raw_df if x not in scout_metadata]
agg_input = scout_raw_df[['IncidentId', ] + cols_to_aggregate]
agg_input_test = test_df[['IncidentId', ] + cols_to_aggregate]

for option in [1, 2, 3]:

    str_desc = get_str_desc_of_reduction_function(
        "RowAgg",
        None,
        dir="row",
        grb="IncidentId",
        option=option
    )

    # check if file exists in directory
    if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):

        start_time = time.time()

        aggregated_result = aggregation_based_reduction(
            agg_input, dir="row", grb='IncidentId', option=option
        )
        aggregated_result_test = aggregation_based_reduction(
            agg_input_test, dir="row", grb='IncidentId', option=option
        )

        end_time = time.time()
        time_taken = int(end_time - start_time)

        rename_col = list()
        for old_col in aggregated_result.columns:
            rename_col.append(":".join(old_col))
        
        aggregated_result.columns = rename_col
        aggregated_result.reset_index(inplace=True)

        rename_col_test = list()
        for old_col in aggregated_result_test.columns:
            rename_col_test.append(":".join(old_col))
        
        aggregated_result_test.columns = rename_col_test
        aggregated_result_test.reset_index(inplace=True)

        save_file_name = os.path.join(
            one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
        )
    
        with open(save_file_name, "wb") as fout:
            pickle.dump((aggregated_result, aggregated_result_test), fout)

