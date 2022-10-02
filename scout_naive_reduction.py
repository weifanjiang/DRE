import utils
import reductions
import os
import time


# dummy = False: running on Azure workspace
dummy = True
save_dir = utils.scout_naive_reduction_save_dir


scout_raw_df = utils.load_raw_incident_device_health_reports(dummy=dummy)


# one hop naive reductions
# for one-hop reduction, only consider col sampling and aggregation based method
# since real scout data has too much rows to be sampled without aggregated
if dummy:
    one_hop_out_dir = os.path.join(
        utils.scout_data_dir,
        save_dir,
        "one_hop"
    )
else:
    one_hop_out_dir = os.path.join(
        utils.scout_azure_dbfs_dir,
        save_dir,
        "one_hop"
    )
os.system("mkdir -p {}".format(one_hop_out_dir))

# SMF
smf_cols_to_reduce = [x for x in scout_raw_df if x not in utils.scout_metadata]
smf_input_mat = scout_raw_df[smf_cols_to_reduce].values
for keepFrac in utils.reduction_strengths:
    for model in ['fls', 'fbs']:

        str_desc = utils.get_str_desc_of_reduction_function(
            "ColSampling",
            None,
            method='smf',
            dir="col",
            model=model,
            keepFrac=keepFrac
        )

        # check if file exists in directory
        if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):
            
            start_time = time.time()

            selected_idx = reductions.sampling_based_reduction(
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
                one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
            )

            to_save = scout_raw_df[utils.scout_metadata + selected_columns]
            to_save.to_csv(save_file_name, index=False)


# SSP
ssp_cols_to_reduce = [x for x in scout_raw_df if x not in utils.scout_metadata]
ssp_input_mat = scout_raw_df[ssp_cols_to_reduce].values
for keepFrac in utils.reduction_strengths:
    for sampler in ['volume', 'doublePhase', 'leverage']:

        str_desc = utils.get_str_desc_of_reduction_function(
            "ColSampling",
            None,
            method='ssp',
            dir="col",
            sampler=sampler,
            keepFrac=keepFrac
        )

        # check if file exists in directory
        if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):

            start_time = time.time()

            selected_idx = reductions.sampling_based_reduction(
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
                one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
            )

            to_save = scout_raw_df[utils.scout_metadata + selected_columns]
            to_save.to_csv(save_file_name, index=False)


# Row aggregation
cols_to_aggregate = [x for x in scout_raw_df if x not in utils.scout_metadata]
agg_input = scout_raw_df[['IncidentId', ] + cols_to_aggregate]

for option in [1, 2, 3]:

    str_desc = utils.get_str_desc_of_reduction_function(
        "RowAgg",
        None,
        dir="row",
        grb="IncidentId",
        option=option
    )

    # check if file exists in directory
    if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):

        start_time = time.time()

        aggregated_result = reductions.aggregation_based_reduction(
            agg_input, dir="row", grb='IncidentId', option=option
        )

        end_time = time.time()
        time_taken = int(end_time - start_time)

        rename_col = list()
        for old_col in aggregated_result.columns:
            rename_col.append(":".join(old_col))
        
        aggregated_result.columns = rename_col
        aggregated_result.reset_index(inplace=True)

        save_file_name = os.path.join(
            one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
        )
    
        aggregated_result.to_csv(save_file_name, index=False)

