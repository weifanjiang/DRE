import utils
import os
import tqdm
import numpy as np
import pandas as pd


fake_incident_count = 10
reports_per_incident_lim = 20

if os.path.isdir(utils.scout_dummy_device_health_dir):
    os.system("rm -r {}".format(utils.scout_dummy_device_health_dir))
os.mkdir(utils.scout_dummy_device_health_dir)

full_cols = utils.load_device_health_report_columns()
dhr_cols = full_cols[5:]

fake_incident_ids = list()
fake_incident_labels = list()

for fake_incident_id in tqdm.tqdm(range(fake_incident_count), total=fake_incident_count):
    fake_fname = "{}.csv".format(fake_incident_id)
    report_count = np.random.randint(1, reports_per_incident_lim + 1)
    
    generated = list()
    for _ in range(report_count):
        report = {
            "IncidentId": fake_incident_id,
            "EntityName": str(np.random.rand()),
            "StartTime": str(np.random.rand()),
            "EndTime": str(np.random.rand()),
            "EntityType": np.random.choice(utils.scout_entity_types)
        }
        for dhr_col in dhr_cols:
            report[dhr_col] = np.random.rand()
        generated.append(report)
    
    incident_df = pd.DataFrame(generated)
    incident_df.to_csv(os.path.join(utils.scout_dummy_device_health_dir, fake_fname), index=False)

    fake_incident_ids.append(fake_incident_id)
    fake_incident_labels.append(np.random.choice([True, False]))

label_df = pd.DataFrame(
    data={"IncidentId": fake_incident_ids, "Label": fake_incident_labels}
)
label_df.to_csv(utils.scout_dummy_label_path, index=False)
