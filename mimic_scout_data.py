from utils import *


import uuid
import numpy as np

dummy = True
scout_raw_df = load_raw_incident_device_health_reports(dummy=dummy)


# anonymize string fields
string_col_names = ['IncidentId', 'EntityType', 'Tier']
for string_col in string_col_names:
    all_distinct_vals = scout_raw_df[string_col].unique()
    real2fake = dict()
    for val in all_distinct_vals:
        real2fake[val] = str(uuid.uuid1())
    
    scout_raw_df[string_col] = scout_raw_df.apply(lambda row: real2fake[row[string_col]], axis=1)


# anonymize numerical fields
hm_cols = [x for x in scout_raw_df if x not in string_col_names]
hm_rename_dict = dict()
for hm in hm_cols:
    hm_rename_dict[hm] = str(uuid.uuid1())

    # generate new column values that have positive/negative/zero at same positions
    old_col_values = scout_raw_df[hm].values
    new_col_values = np.zeros(old_col_values.shape)

    new_col_values[old_col_values > 0] = np.random.uniform(0.01, 1, np.sum(old_col_values > 0))
    new_col_values[old_col_values < 0] = np.random.uniform(-1, -0.01, np.sum(old_col_values < 0))
    new_col_values[old_col_values == np.nan] = np.nan

    scout_raw_df[hm] = new_col_values

scout_raw_df.rename(columns=hm_rename_dict, inplace=True)

if dummy:
    scout_raw_df.to_csv(os.path.join(scout_data_dir, "scout_anonymized_raw_data.csv"), index=False)
else:
    scout_raw_df.to_csv(os.path.join(scout_azure_dbfs_dir, "scout_anonymized_raw_data.csv"), index=False)