# Weifan Jiang, weifanjiang@g.harvard.edu
# Prepare philly traces data


import datetime
import os
import json
import random
import numpy as np


def parse_date(date_str):
    if date_str is None or date_str == '' or date_str == 'None':
        return None
    return datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')


def timedelta_to_minutes(timedelta):
    minutes = 0.0
    minutes += timedelta.days * 24 * 60
    minutes += timedelta.seconds / 60.0
    minutes += timedelta.microseconds / (60 * 1000)
    return minutes


def count_machines(detail):
    cpu_count = len(detail)
    gpu_count = 0
    for machine in detail:
        gpu_count += len(machine["gpus"])
    return cpu_count, gpu_count


if __name__ == "__main__":
    # seed
    np.random.seed(10)
    random.seed(10)


    # data location
    trace_dir = "philly-traces/trace-data/"
    job_log_path = os.path.join(trace_dir, "cluster_job_log")
    output_dir = "data/philly"
    sampled_jobs_path = os.path.join(output_dir, "sampled_jobs.json")


    if not os.path.isfile(sampled_jobs_path):
        # read data
        with open(job_log_path, "r") as fin:
            job_log = json.load(fin)
        

        # filter for jobs with one attempt
        jobs_single_attempt = [x for x in job_log if len(x["attempts"]) == 1]
        # jobs with complete runtime properties
        for job in jobs_single_attempt:
            start_time = parse_date(job["attempts"][0]["start_time"])
            end_time = parse_date(job["attempts"][0]["end_time"])
            if start_time is not None and end_time is not None:
                job["runtime_min"] = timedelta_to_minutes(end_time - start_time)
            else:
                job["runtime_min"] = None
        jobs_single_attempt = [x for x in jobs_single_attempt if x['runtime_min'] is not None]
        # filter for jobs that lasted for at list 5 minutes
        jobs_single_attempt = [x for x in jobs_single_attempt if 5 <= x['runtime_min']]
        # try to select jobs scheduled on multiple GPUs
        jobs_single_attempt = [
            x for x in jobs_single_attempt if count_machines(x["attempts"][0]["detail"])[1] > 1
        ]


        # check the distribution of jobs with different final status
        jobs_pass = [x for x in jobs_single_attempt if x["status"] == "Pass"]
        jobs_killed = [x for x in jobs_single_attempt if x["status"] == "Killed"]
        jobs_failed = [x for x in jobs_single_attempt if x["status"] == "Failed"]
        print('job status: Pass ({}), Killed ({}), Failed ({})'.format(
            len(jobs_pass), len(jobs_killed), len(jobs_failed)
        ))


        # sample size: min length of qualified jobs in each status
        output_json = list()
        job_size = np.amin([len(jobs_pass), len(jobs_killed), len(jobs_failed)])
        for jobs in [jobs_pass, jobs_killed, jobs_failed]:
            sampled_jobs = random.sample(jobs, job_size)
            for job in sampled_jobs:
                output_job = dict()
                for key in ("status", "vc", "jobid", "submitted_time", "user", "runtime_min"):
                    output_job[key] = job[key]
                for key in ("start_time", "end_time", "detail"):
                    output_job[key] = job["attempts"][0][key]
                output_json.append(output_job)
        with open(sampled_jobs_path, "w") as fout:
            json.dump(output_json, fout, indent=2)
    

    
