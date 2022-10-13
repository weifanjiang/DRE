from utils import *
from reductions import *

import os
import time
import pandas as pd
import pickle


train_df, test_df = load_train_test_split_philly_dataset()
granularities = ['MachineType', 'TraceType', ['MachineType', 'TraceType']]

save_dir = philly_guided_reduction_save_dir
one_hop_out_dir = os.path.join(
    philly_data_dir,
    save_dir,
    "one_hop"
)
os.system("mkdir -p {}".format(one_hop_out_dir))

for granularity in granularities:

    grb_gran = train_df.groupby(granularity)
    grb_gran_test = test_df.groupby(granularity)

    # sampling based
    for technique in ["ColSampling", "RowSampling"]:
        cols_to_sample = [x for x in train_df.columns if x not in philly_metadata]
        dir = technique[:3].lower()
        for keepFrac in reduction_strengths:
            for method in ["smf", "ssp"]:

                if method == "smf":
                    for model in ["fls", "fbs"]:

                        str_desc = get_str_desc_of_reduction_function(
                            technique,
                            granularity,
                            method=method,
                            dir=dir,
                            model=model,
                            keepFrac=keepFrac
                        )
                        print(str_desc)

                        if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):

                            processed = list()
                            processed_test = list()
                            time_taken = 0
                            for key, sub_df in grb_gran:
                                start_time = time.time()
                                selected_idx = sampling_based_reduction(
                                    sub_df[cols_to_sample].values,
                                    None,
                                    method="smf",
                                    dir=dir,
                                    model=model,
                                    keepFrac=keepFrac
                                )
                                end_time = time.time()
                                time_taken += end_time - start_time

                                sub_df_test = safe_get_subgroup(grb_gran_test, key)

                                if dir == 'col':
                                    selected_columns = [cols_to_sample[x] for x in selected_idx]
                                    processed.append(sub_df[philly_metadata + selected_columns])
                                    if sub_df_test is not None:
                                        processed_test.append(sub_df_test[philly_metadata + selected_columns])
                                else:
                                    processed.append(sub_df.iloc[selected_idx])
                                    if sub_df_test is not None:
                                        processed_test.append(sub_df_test)

                            time_taken = int(time_taken)
                            save_file_name = os.path.join(
                                one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                            )

                            to_save = pd.concat(processed, axis=0, ignore_index=True)
                            to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)
                            
                            with open(save_file_name, "wb") as fout:
                                pickle.dump((to_save, to_save_test), fout)
        
    # aggregation
    # for technique in ["RowAgg", "ColAgg"]:
    grb_cols = [x for x in train_df.columns if x not in philly_metadata]
    grb_cols = ['JobId', ] + grb_cols
    for technique in ["RowAgg", ]:
        dir = technique[0:3].lower()
        for option in [1, 2, 3]:
            str_desc = get_str_desc_of_reduction_function(
                technique,
                granularity,
                dir=dir,
                grb="JobId",
                option=option
            )
            print(str_desc)

            # check if file exists in directory
            if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):
            
                processed = list()
                processed_test = list()
                time_taken = 0

                for keys, sub_df in grb_gran:
                    start_time = time.time()
                    aggregated_result = aggregation_based_reduction(
                        sub_df[grb_cols], dir=dir, grb='JobId', option=option
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

                    sub_df_test = safe_get_subgroup(grb_gran_test, keys)
                    if sub_df_test is not None:
                        aggregated_result_test = aggregation_based_reduction(
                            sub_df_test[grb_cols], dir=dir, grb='JobId', option=option
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
                metadata_left = [x for x in philly_metadata if x in granularity]
                metrics_left = [x for x in to_save.columns if x not in scout_metadata]

                out_columns = ['JobId', ] + metadata_left + metrics_left

                save_file_name = os.path.join(
                    one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                )
                to_save, to_save_test = to_save[out_columns], to_save_test[out_columns]

                with open(save_file_name, "wb") as fout:
                    pickle.dump((to_save, to_save_test), fout)
