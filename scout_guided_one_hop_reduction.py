import utils
import reductions
import os
import time
import pandas as pd
import pickle


# dummy = False: running on Azure workspace
dummy = True
save_dir = utils.scout_guided_reduction_save_dir


scout_raw_df = utils.load_raw_incident_device_health_reports(dummy=dummy)
scout_raw_df, test_df = utils.train_test_split_scout_data(scout_raw_df, 0.8)
granularities = ['EntityType', 'Tier', ['EntityType', 'Tier']]


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


for granularity in granularities:

    grb_gran = scout_raw_df.groupby(granularity)
    grb_gran_test = test_df.groupby(granularity)

    # sampling based
    cols_to_sample = [x for x in scout_raw_df.columns if x not in utils.scout_metadata]
    for keepFrac in utils.reduction_strengths:
        for method in ["smf", "ssp"]:

            if method == "smf":
                for model in ["fls", "fbs"]:

                    str_desc = utils.get_str_desc_of_reduction_function(
                        "ColSampling",
                        granularity,
                        method=method,
                        dir="col",
                        model=model,
                        keepFrac=keepFrac
                    )

                    if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):

                        processed = list()
                        processed_test = list()
                        time_taken = 0
                        for key, sub_df in grb_gran:
                            start_time = time.time()
                            selected_idx = reductions.sampling_based_reduction(
                                sub_df[cols_to_sample].values,
                                None,
                                method="smf",
                                dir="col",
                                model=model,
                                keepFrac=keepFrac
                            )
                            end_time = time.time()
                            time_taken += end_time - start_time
                            selected_columns = [cols_to_sample[x] for x in selected_idx]
                            processed.append(
                                sub_df[utils.scout_metadata + selected_columns]
                            )

                            # handle test data
                            sub_df_test = grb_gran_test.get_group(key)
                            processed_test.append(
                                sub_df_test[utils.scout_metadata + selected_columns]
                            )

                        time_taken = int(time_taken)
                        save_file_name = os.path.join(
                            one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                        )

                        to_save = pd.concat(processed, axis=0, ignore_index=True)
                        to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)
                        
                        with open(save_file_name, "wb") as fout:
                            pickle.dump((to_save, to_save_test), fout)

            else:  # ssp
                for sampler in ['volume', 'doublePhase', 'leverage']:

                    str_desc = utils.get_str_desc_of_reduction_function(
                        "ColSampling",
                        granularity,
                        method='ssp',
                        dir="col",
                        sampler=sampler,
                        keepFrac=keepFrac
                    )

                    # check if file exists in directory
                    if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):

                        processed = list()
                        processed_test = list()
                        time_taken = 0
                        for _, sub_df in grb_gran:
                            if sub_df.shape[0] > 2:  # pre-condition for ssp
                                start_time = time.time()
                                selected_idx = reductions.sampling_based_reduction(
                                    sub_df[cols_to_sample].values,
                                    None,
                                    method='ssp',
                                    dir="col",
                                    sampler=sampler,
                                    keepFrac=keepFrac
                                )
                                end_time = time.time()
                                time_taken += end_time - start_time
                                selected_columns = [cols_to_sample[x] for x in selected_idx]
                                processed.append(
                                    sub_df[utils.scout_metadata + selected_columns]
                                )

                                # handle test data
                                sub_df_test = grb_gran_test.get_group(key)
                                processed_test.append(
                                    sub_df_test[utils.scout_metadata + selected_columns]
                                )

                        time_taken = int(time_taken)
                        save_file_name = os.path.join(
                            one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                        )

                        to_save = pd.concat(processed, axis=0, ignore_index=True)
                        to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)
                        
                        with open(save_file_name, "wb") as fout:
                            pickle.dump((to_save, to_save_test), fout)
    
    # row aggregation
    grb_cols = [x for x in scout_raw_df.columns if x not in utils.scout_metadata]
    grb_cols = ['IncidentId', ] + grb_cols
    for option in [1, 2, 3, ]:
        str_desc = utils.get_str_desc_of_reduction_function(
            "RowAgg",
            granularity,
            dir="row",
            grb="IncidentId",
            option=option
        )

        # check if file exists in directory
        if not utils.if_file_w_prefix_exists(one_hop_out_dir, str_desc):
        
            processed = list()
            processed_test = list()
            time_taken = 0

            for keys, sub_df in grb_gran:
                start_time = time.time()
                aggregated_result = reductions.aggregation_based_reduction(
                    sub_df[grb_cols], dir="row", grb='IncidentId', option=option
                )
                time_taken += int(time.time() - start_time)

                rename_col = list()
                for old_col in aggregated_result.columns:
                    rename_col.append(":".join(old_col))
                
                aggregated_result.columns = rename_col
                aggregated_result.reset_index(inplace=True)

                if type(granularity) != str:
                    for gran_name, gran_val in zip(granularity, keys):
                        aggregated_result[gran_name] = gran_val
                else:
                    aggregated_result[granularity] = keys

                processed.append(aggregated_result)

                sub_df_test = grb_gran_test.get_group(keys)
                aggregated_result_test = reductions.aggregation_based_reduction(
                    sub_df_test[grb_cols], dir="row", grb='IncidentId', option=option
                )
                rename_col_test = list()
                for old_col in aggregated_result_test.columns:
                    rename_col_test.append(":".join(old_col))
                aggregated_result_test.columns = rename_col
                aggregated_result_test.reset_index(inplace=True)

                if type(granularity) != str:
                    for gran_name, gran_val in zip(granularity, keys):
                        aggregated_result_test[gran_name] = gran_val
                else:
                    aggregated_result_test[granularity] = keys
                
                processed_test.append(aggregated_result_test)

            to_save = pd.concat(processed, axis=0, ignore_index=True)
            to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)

            # reorganize columns
            metadata_left = [x for x in utils.scout_metadata if x in granularity]
            metrics_left = [x for x in to_save.columns if x not in utils.scout_metadata]

            out_columns = ['IncidentId', ] + metadata_left + metrics_left

            save_file_name = os.path.join(
                one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
            )
            to_save, to_save_test = to_save[out_columns], to_save_test[out_columns]

            with open(save_file_name, "wb") as fout:
                pickle.dump((to_save, to_save_test), fout)
