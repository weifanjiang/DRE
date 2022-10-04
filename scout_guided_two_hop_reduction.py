from utils import *
from reductions import *


import pickle
import os
import time
import pandas as pd


# dummy = False: running on Azure workspace
dummy = True
save_dir = scout_guided_reduction_save_dir


if dummy:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm

if dummy:
    one_hop_out_dir = os.path.join(
        scout_data_dir,
        save_dir,
        "one_hop"
    )
    two_hop_out_dir = os.path.join(
        scout_data_dir,
        save_dir,
        "two_hop"
    )
else:
    one_hop_out_dir = os.path.join(
        scout_azure_dbfs_dir,
        save_dir,
        "one_hop"
    )
    two_hop_out_dir = os.path.join(
        scout_azure_dbfs_dir,
        save_dir,
        "two_hop"
    )

os.system('mkdir -p {}'.format(two_hop_out_dir))

# get all guided one-hop reductions
one_hop_filepaths = [x for x in os.listdir(one_hop_out_dir) if x.endswith(".pickle")]

all_algos = set()

for one_hop_filepath in tqdm(one_hop_filepaths):
    with open(os.path.join(one_hop_out_dir, one_hop_filepath), "rb") as fin:
        train_df, test_df = pickle.load(fin)
    train_df.fillna(0, inplace=True)
    test_df.fillna(0, inplace=True)

    one_hop_str = one_hop_filepath.split(".pickle")[0]

    # get possible second hop algorithms
    prev_algo = one_hop_str.split("_")[0]
    next_algos = set()
    for candidate in ['ColSampling', 'RowSampling', 'RowAgg']:

        if candidate == 'RowSapmling' and prev_algo == 'RowAgg':
            next_algos.add(candidate)
        elif candidate != prev_algo:
            next_algos.add(candidate)
        
    # get available metadatas in the reduced dataset
    existing_metadatas = [x for x in scout_metadata if x in train_df.columns]
    possible_granularities = list()
    if 'EntityType' in existing_metadatas:
        possible_granularities.append('EntityType')
    if 'Tier' in existing_metadatas:
        possible_granularities.append('Tier')
    if len(possible_granularities) == 2:
        possible_granularities.append(['EntityType', 'Tier'])
    
    prev_granularity = one_hop_filepath.split("_")
    
    for granularity in possible_granularities:

        grb_gran = train_df.groupby(by=granularity)
        grb_gran_test = test_df.groupby(by=granularity)

        for next_algo in next_algos:

            if next_algo == 'ColSampling' or next_algo == 'RowSampling':  
                dir = next_algo[0:3].lower()
                
                cols_to_sample = [x for x in train_df.columns if x not in existing_metadatas]
                for keepFrac in reduction_strengths:

                    for method in ["smf", "ssp"]:
                        if method == "smf":
                            for model in ["fls", "fbs"]:
                                
                                str_desc = get_str_desc_of_reduction_function(
                                    next_algo,
                                    granularity,
                                    method=method,
                                    dir=dir,
                                    model=model,
                                    keepFrac=keepFrac
                                )
                                two_hop_desc = one_hop_str + "&" + str_desc

                                if not if_file_w_prefix_exists(two_hop_out_dir, two_hop_desc):
                                    processed = list()
                                    processed_test = list()
                                    time_taken = 0
                                    for key, sub_df in grb_gran:
                                        start_time = time.time()
                                        selected_idx = sampling_based_reduction(
                                            sub_df[cols_to_sample].values,
                                            None,
                                            method=method,
                                            dir=dir,
                                            model=model,
                                            keepFrac=keepFrac
                                        )
                                        end_time = time.time()
                                        time_taken += end_time - start_time
                                        
                                        sub_df_test = safe_get_subgroup(grb_gran_test, key)
                                        if dir == 'col':
                                            selected_columns = [cols_to_sample[x] for x in selected_idx]
                                            processed.append(
                                                sub_df[scout_metadata + selected_columns]
                                            )

                                            # handle test data
                                            if sub_df_test is not None:
                                                processed_test.append(
                                                    sub_df_test[scout_metadata + selected_columns]
                                                )
                                        else:  # row sampling
                                            processed.append(sub_df.iloc[selected_idx])
                                            if sub_df_test is not None:
                                                processed_test.append(sub_df_test)  # no need to row-sampling on test data

                                    time_taken = int(time_taken)
                                    save_file_name = os.path.join(
                                        two_hop_out_dir, "{}_sec{}.pickle".format(two_hop_desc, time_taken)
                                    )

                                    to_save = pd.concat(processed, axis=0, ignore_index=True)
                                    to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)
                                    
                                    with open(save_file_name, "wb") as fout:
                                        pickle.dump((to_save, to_save_test), fout)

                        elif method == "ssp":

                            for sampler in ['volume', 'doublePhase', 'leverage']:
                                str_desc = get_str_desc_of_reduction_function(
                                    next_algo,
                                    granularity,
                                    method='ssp',
                                    dir=dir,
                                    sampler=sampler,
                                    keepFrac=keepFrac
                                )
                                two_hop_desc = one_hop_str + "&" + str_desc

                                if not if_file_w_prefix_exists(two_hop_out_dir, two_hop_desc):
                                    processed = list()
                                    processed_test = list()
                                    time_taken = 0
                                    for key, sub_df in grb_gran:
                                        start_time = time.time()
                                        selected_idx = sampling_based_reduction(
                                            sub_df[cols_to_sample].values,
                                            None,
                                            method=method,
                                            dir=dir,
                                            sampler=sampler,
                                            keepFrac=keepFrac
                                        )
                                        end_time = time.time()
                                        time_taken += end_time - start_time
                                        
                                        sub_df_test = safe_get_subgroup(grb_gran_test, key)
                                        if dir == 'col':
                                            selected_columns = [cols_to_sample[x] for x in selected_idx]
                                            processed.append(
                                                sub_df[scout_metadata + selected_columns]
                                            )

                                            # handle test data
                                            if sub_df_test is not None:
                                                processed_test.append(
                                                    sub_df_test[scout_metadata + selected_columns]
                                                )
                                        else:  # row sampling
                                            processed.append(sub_df.iloc[selected_idx])
                                            if sub_df_test is not None:
                                                processed_test.append(sub_df_test)

                                    time_taken = int(time_taken)
                                    save_file_name = os.path.join(
                                        two_hop_out_dir, "{}_sec{}.pickle".format(two_hop_desc, time_taken)
                                    )

                                    to_save = pd.concat(processed, axis=0, ignore_index=True)
                                    to_save_test = pd.concat(processed_test, axis=0, ignore_index=True)
                                    
                                    with open(save_file_name, "wb") as fout:
                                        pickle.dump((to_save, to_save_test), fout)


            elif next_algo == "RowAgg":

                agg_cols = [x for x in train_df.columns if x not in scout_metadata]
                agg_cols = ['IncidentId', ] + agg_cols

                for option in [1, 2, 3, ]:
                    str_desc = get_str_desc_of_reduction_function(
                        "RowAgg",
                        granularity,
                        dir="row",
                        grb="IncidentId",
                        option=option
                    )
                    two_hop_desc = one_hop_str + "&" + str_desc

                    if not if_file_w_prefix_exists(two_hop_out_dir, two_hop_desc):
                        processed = list()
                        processed_test = list()
                        time_taken = 0

                        for keys, sub_df in grb_gran:
                            start_time = time.time()
                            aggregated_result = aggregation_based_reduction(
                                sub_df[agg_cols], dir="row", grb='IncidentId', option=option
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
                                    sub_df_test[agg_cols], dir="row", grb='IncidentId', option=option
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
                        metadata_left = [x for x in scout_metadata if x in granularity]
                        metrics_left = [x for x in to_save.columns if x not in scout_metadata]

                        out_columns = ['IncidentId', ] + metadata_left + metrics_left

                        save_file_name = os.path.join(
                            two_hop_out_dir, "{}_sec{}.pickle".format(two_hop_desc, time_taken)
                        )
                        to_save, to_save_test = to_save[out_columns], to_save_test[out_columns]

                        with open(save_file_name, "wb") as fout:
                            pickle.dump((to_save, to_save_test), fout)

