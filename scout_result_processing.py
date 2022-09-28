import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils import *


scout_evaluation_df = pd.read_csv("data/scout/evaluation_results.csv", index_col=0)
all_reduction_strs = [x for x in scout_evaluation_df.index.values]
parsed_reductions = [parse_scout_filename(x) for x in all_reduction_strs]
parse_dict = dict()
for unparsed, parsed in zip(all_reduction_strs, parsed_reductions):
    parse_dict[unparsed] = parsed


processed = list()
for reduction_str in tqdm.tqdm(all_reduction_strs):

    hop_dict = dict()

    parsed = parse_dict[reduction_str]
    reduction_list = parsed['reduction']
    eval_time = parsed['eval_time']

    hop_dict['hop1_reduction'] = reduction_list[0][0]
    hop_dict['hop1_time'] = reduction_list[0][1]

    if len(reduction_list) == 2:
        hop_dict['hop2_reduction'] = reduction_list[1][0]
        hop_dict['hop2_time'] = reduction_list[1][1]
    else:
        hop_dict['hop2_reduction'] = np.nan
        hop_dict['hop2_time'] = np.nan
    
    hop_dict['eval_time'] = eval_time
    
    # dataset size
    data_char = get_scout_dataset_characteristics(reduction_str)
    hop_dict['keep_frac'] = (data_char.loc['num_row', '0'] * data_char.loc['num_col', '0'] - data_char.loc['null_vals', '0']) / scout_total_size
    # evaluation result
    hop_dict['acc'] = scout_evaluation_df.loc[reduction_str, 'acc']
    hop_dict['f1'] = scout_evaluation_df.loc[reduction_str, 'f1']

    processed.append(hop_dict)


processed_df = pd.DataFrame(processed)
processed_df.to_csv(os.path.join(scout_data_dir, 'processed_results.csv'), index=False)
