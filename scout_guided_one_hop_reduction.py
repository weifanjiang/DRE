import utils
import reductions
import os
import time
import pandas as pd


# dummy = False: running on Azure workspace
dummy = True
save_dir = utils.scout_guided_reduction_save_dir


scout_raw_df = utils.load_raw_incident_device_health_reports(dummy=dummy)
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
                        time_taken = 0
                        for _, sub_df in grb_gran:
                            start_time = time.time()
                            selected_idx = reductions.sampling_based_reduction(
                                sub_df[cols_to_sample].values,
                                None,
                                method='smf',
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
                        time_taken = int(time_taken)
                        save_file_name = os.path.join(
                            one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
                        )

                        to_save = pd.concat(processed, axis=0, ignore_index=True)
                        to_save.to_csv(save_file_name)

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
                        time_taken = int(time_taken)
                        save_file_name = os.path.join(
                            one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
                        )

                        to_save = pd.concat(processed, axis=0, ignore_index=True)
                        to_save.to_csv(save_file_name)
    
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
            
            to_save = pd.concat(processed, axis=0, ignore_index=True)

            # reorganize columns
            metadata_left = [x for x in utils.scout_metadata if x in granularity]
            metrics_left = [x for x in to_save.columns if x not in utils.scout_metadata]

            out_columns = ['IncidentId', ] + metadata_left + metrics_left

            save_file_name = os.path.join(
                one_hop_out_dir, "{}_sec{}.csv".format(str_desc, time_taken)
            )
            to_save[out_columns].to_csv(save_file_name, index=False)
