from multiprocessing import reduction
from utils import *
from reductions import *

import os
import time
from tqdm import tqdm
import pickle


train_df, test_df = load_train_test_split_philly_dataset()
save_dir = os.path.join(philly_data_dir, philly_naive_reduction_save_dir)
one_hop_out_dir = os.path.join(save_dir, "one_hop")
os.system("mkdir -p {}".format(one_hop_out_dir))

print(train_df.shape, test_df.shape)

# sampling based
sample_cols = [x for x in train_df.columns if x not in philly_metadata]
for keepFrac in reduction_strengths:
    for technique in ['RowSampling', 'ColSampling']:
    # for technique in ['ColSampling']:
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
                    if not if_file_w_prefix_exists(one_hop_out_dir, str_desc):
                        start_time = time.time()
                        selected_idx = sampling_based_reduction(
                            train_df[sample_cols].values,
                            None,
                            method='smf',
                            dir=dir,
                            model=model,
                            keepFrac=keepFrac
                        )
                        end_time = time.time()
                        time_taken = int(end_time - start_time)
                        save_file_name = os.path.join(one_hop_out_dir, "{}_sec{}.pickle".format(str_desc, time_taken))

                        if dir == "col":
                            selected_columns = [sample_cols[x] for x in selected_idx]
                            to_save = train_df[philly_metadata + selected_columns]
                            to_save_test = test_df[philly_metadata + selected_columns]
                        else:
                            to_save = train_df.iloc[selected_idx]
                            to_save_test = test_df
                        with open(save_file_name, "wb") as fout:
                            pickle.dump((to_save, to_save_test), fout)
