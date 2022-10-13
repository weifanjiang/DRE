from utils import *
from reductions import *


import os
import time
from tqdm import tqdm
import pickle


gcut_train_df, gcut_test_df = load_google_dataset()
save_dir = os.path.join(gcut_data_dir, gcut_naive_reduction_save_dir)
one_hop_out_dir = os.path.join(save_dir, "one_hop")

os.system("mkdir -p {}".format(one_hop_out_dir))

# column sampling based
sample_cols = [x for x in gcut_train_df.columns if x not in gcut_metadata]
for keepFrac in reduction_strengths:
    # for method in ["smf", "ssp"]:
    for method in ["smf"]:

        if method == "smf":
            for model in ["fls", "fbs"]:

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
                        gcut_train_df[sample_cols].values,
                        None,
                        method='smf',
                        dir="col",
                        model=model,
                        keepFrac=keepFrac
                    )
                    end_time = time.time()
                    time_taken = int(end_time - start_time)

                    selected_columns = [sample_cols[x] for x in selected_idx]
                    save_file_name = os.path.join(
                        one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                    )
                    to_save = gcut_train_df[gcut_metadata + selected_columns]
                    to_save_test = gcut_test_df[gcut_metadata + selected_columns]
                    with open(save_file_name, "wb") as fout:
                        pickle.dump((to_save, to_save_test), fout)
        elif method == "ssp":
            for sampler in ['doublePhase', 'leverage']:
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
                        gcut_train_df[sample_cols].values.astype(np.float16),
                        None,
                        method='ssp',
                        dir="col",
                        sampler=sampler,
                        keepFrac=keepFrac
                    )
                    end_time = time.time()
                    time_taken = int(end_time - start_time)
                    selected_columns = [sample_cols[x] for x in selected_idx]
                    save_file_name = os.path.join(
                        one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
                    )
                    to_save = gcut_train_df[gcut_metadata + selected_columns]
                    to_save_test = gcut_test_df[gcut_metadata + selected_columns]
                    with open(save_file_name, "wb") as fout:
                        pickle.dump((to_save, to_save_test), fout)

# Row aggregation
cols_to_aggregate = [x for x in gcut_train_df.columns if x not in gcut_metadata]
agg_input = gcut_train_df[['TaskId', ] + cols_to_aggregate]
agg_input_test = gcut_test_df[['TaskId', ] + cols_to_aggregate]

label_df = gcut_train_df[['TaskId', 'Label']].drop_duplicates()
label_df_test = gcut_test_df[['TaskId', 'Label']].drop_duplicates()

for option in [1, 2, 3]:

    str_desc = get_str_desc_of_reduction_function(
        "RowAgg",
        None,
        dir="row",
        grb="IncidentId",
        option=option
    )

    if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):
        start_time = time.time()
        aggregated_result = aggregation_based_reduction(
            agg_input, dir="row", grb='TaskId', option=option
        )
        aggregated_result_test = aggregation_based_reduction(
            agg_input_test, dir="row", grb='TaskId', option=option
        )
        end_time = time.time()
        time_taken = int(end_time - start_time)

        rename_col = list()
        for old_col in aggregated_result.columns:
            rename_col.append(":".join(old_col))
        aggregated_result.columns = rename_col
        aggregated_result.reset_index(inplace=True)
        aggregated_result = aggregated_result.merge(label_df, on='TaskId', how='inner')

        rename_col_test = list()
        for old_col in aggregated_result_test.columns:
            rename_col_test.append(":".join(old_col))
        aggregated_result_test.columns = rename_col_test
        aggregated_result_test.reset_index(inplace=True)
        aggregated_result_test = aggregated_result_test.merge(label_df_test, on='TaskId', how='inner')

        save_file_name = os.path.join(
            one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken)
        )
    
        with open(save_file_name, "wb") as fout:
            pickle.dump((aggregated_result, aggregated_result_test), fout)


# Two-hop reductions
two_hop_out_dir = os.path.join(save_dir, "two_hop")
os.system("mkdir -p {}".format(two_hop_out_dir))

# for row aggregated one-hop reductions, apply column/row sampling
one_hop_row_agg = [x for x in os.listdir(one_hop_out_dir) if "RowAgg" in x]
for ohra in tqdm(one_hop_row_agg, desc="column sampling row reduced datasets"):
    one_hop_desc = one_hop_str = ohra.split(".pickle")[0]

    with open(os.path.join(one_hop_out_dir, ohra), "rb") as fin:
        prev_train, prev_test = pickle.load(fin)
    
    prev_metadata = [x for x in gcut_metadata if x in prev_train.columns]
    prev_metrics = sorted([x for x in prev_train.columns if x not in gcut_metadata])
    reorder_cols = prev_metadata + prev_metrics
    prev_train, prev_test = prev_train[reorder_cols], prev_test[reorder_cols]

    for keepFrac in reduction_strengths:
        for technique in ["RowSampling", "ColSampling"]:
            dir = technique[:3].lower()
            for method in ["smf", "ssp"]:
                if method == "smf":
                    for model in ["fls", "fbs"]:
                        str_desc = get_str_desc_of_reduction_function(
                            technique,
                            None,
                            method='smf',
                            dir=dir,
                            model=model,
                            keepFrac=keepFrac
                        )
                        two_hop_desc = one_hop_str + "&" + str_desc
                        if not if_file_w_prefix_exists(two_hop_out_dir, two_hop_desc):
                            start_time = time.time()
                            selected_idx = sampling_based_reduction(
                                prev_train[prev_metrics].values,
                                None,
                                method='smf',
                                dir=dir,
                                model=model,
                                keepFrac=keepFrac
                            )
                            end_time = time.time()
                            time_taken = int(end_time - start_time)
                            save_file_name = os.path.join(two_hop_out_dir, "{}_sec{}.pickle".format(two_hop_desc, time_taken))

                            if dir == "col":
                                selected_columns = [prev_metrics[x] for x in selected_idx]
                                to_save = prev_train[prev_metadata + selected_columns]
                                to_save_test = prev_test[prev_metadata + selected_columns]
                            else:
                                to_save = prev_train.iloc[selected_idx]
                                to_save_test = prev_test
                            with open(save_file_name, "wb") as fout:
                                pickle.dump((to_save, to_save_test), fout)
                elif method == "ssp":
                    for sampler in ['doublePhase', 'leverage']:
                        str_desc = get_str_desc_of_reduction_function(
                            technique,
                            None,
                            method='ssp',
                            dir=dir,
                            sampler=sampler,
                            keepFrac=keepFrac
                        )
                        two_hop_desc = one_hop_str + "&" + str_desc
                        if not if_file_w_prefix_exists(two_hop_out_dir, two_hop_desc):
                            start_time = time.time()
                            selected_idx = sampling_based_reduction(
                                prev_train[prev_metrics].values,
                                None,
                                method='ssp',
                                dir=dir,
                                sampler=sampler,
                                keepFrac=keepFrac
                            )
                            end_time = time.time()
                            time_taken = int(end_time - start_time)
                            save_file_name = os.path.join(two_hop_out_dir, "{}_sec{}.pickle".format(two_hop_desc, time_taken))

                            if dir == "col":
                                selected_columns = [prev_metrics[x] for x in selected_idx]
                                to_save = prev_train[prev_metadata + selected_columns]
                                to_save_test = prev_test[prev_metadata + selected_columns]
                            else:
                                to_save = prev_train.iloc[selected_idx]
                                to_save_test = prev_test
                            with open(save_file_name, "wb") as fout:
                                pickle.dump((to_save, to_save_test), fout)
