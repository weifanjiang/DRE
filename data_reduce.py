"""
Weifan Jiang, weifanjiang@g.harvard.edu
"""


import reductions
import utils
import os
import json
import time
import pandas as pd
from sklearn.utils import resample
from abc import ABC, abstractmethod


class DataReduce(ABC):

    def get_strategy_available_configurations(self, strategy):
        keep_fracs = [0.2, 0.4, 0.6, 0.8, ]
        agg_options = [1, 2, 3]

        all_configs = list()
        if strategy == 'ColAgg':
            for option in agg_options:
                all_configs.append({"dir": "col", "option": option})
        elif strategy == "RowAgg":
            ## For row aggregation, need to manually insert the group by criteria (if available)
            for option in agg_options:
                all_configs.append({"dir": "row", "option": option})
        elif strategy == "ColSamp":
            # smf
            for search in ["fls", "fbs"]:
                for kf in keep_fracs:
                    all_configs.append({"method": "smf", "search": search, "keepFrac": kf, "dir": "col"})
            # ssp
            for selection in ["volume", "doublePhase", "leverage"]:
                for kf in keep_fracs:
                    all_configs.append({"method": "ssp", "selection": selection, "keepFrac": kf})
        elif strategy == "RowSamp":
            # smf
            for search in ["fls", "fbs"]:
                for kf in keep_fracs:
                    all_configs.append({"method": "smf", "search": search, "keepFrac": kf, "dir": "row"})
            # al
            init_frac = 0.05
            for model in ["RF", "MLP", "KNN"]:
                for strat in ["margin", "entropy", "uncertain", "expected"]:
                    for kf in keep_fracs:
                        all_configs.append({"model": model, "initFrac": init_frac, "strat": strat, "keepFrac": kf})
        return all_configs
    

    def stringfy_reduction(self, reduction_strategy, reduction_kwargs):
        keys = sorted([x for x in reduction_kwargs.keys()])
        return "{}_{}".format(
            reduction_strategy,
            "-".join(["{}:{}".format(x, reduction_kwargs[x]) for x in keys])
        )
    

    @abstractmethod
    def get_available_reduction_strategies(self):
        pass


    @abstractmethod
    def apply_reduction(self, reduction_config):
        pass


class PhillyDataReduce(DataReduce):


    def __init__(self, sample_frac=1.0, create_from_content=None):

        if create_from_content is not None:
            self.jobs, self.df_dict, self.applied_reductions = create_from_content["jobs"], create_from_content["df_dict"], create_from_content["applied_reductions"]
        else:
            with open(os.path.join(utils.philly_data_dir, "sampled_jobs.json"), "r") as fin:
                total_jobs = json.load(fin)
            job_labels = [x["status"] for x in total_jobs]
            self.jobs = resample(
                total_jobs,
                replace=False,
                n_samples=int(sample_frac * len(total_jobs)),
                stratify=job_labels,
                random_state=10
            )
            cpu_list, gpu_list, mem_list = list(), list(), list()
            for job in self.jobs:
                cpu_df, gpu_df, mem_df = utils.get_philly_per_job_trace(job["jobid"])
                cpu_df["jobid"] = job["jobid"]
                cpu_df["status"] = job["status"]
                gpu_df["jobid"] = job["jobid"]
                gpu_df["status"] = job["status"]
                mem_df["jobid"] = job["jobid"]
                mem_df["status"] = job["status"]
                cpu_list.append(cpu_df)
                gpu_list.append(gpu_df)
                mem_list.append(mem_df)
            self.df_dict = dict()
            self.df_dict["cpu"] = pd.concat(cpu_list, axis=0, ignore_index=True)
            self.df_dict["gpu"] = pd.concat(gpu_list, axis=0, ignore_index=True)
            self.df_dict["mem"] = pd.concat(mem_list, axis=0, ignore_index=True)
            self.df_dict["cpu"] = self.df_dict["cpu"][['jobid', 'status'] + [x for x in self.df_dict["cpu"].columns if x not in ['jobid', 'status']]]
            self.df_dict["gpu"] = self.df_dict["gpu"][['jobid', 'status'] + [x for x in self.df_dict["gpu"].columns if x not in ['jobid', 'status']]]
            self.df_dict["mem"] = self.df_dict["mem"][['jobid', 'status'] + [x for x in self.df_dict["mem"].columns if x not in ['jobid', 'status']]]
        
            self.applied_reductions = list()


    def get_available_reduction_strategies(self):
        """
        Implement rules for philly reduction
        """
        applied_reduction_dict = {"cpu": set(), "gpu": set(), "mem": set()}
        for red in self.applied_reductions:
            applied_reduction_dict[red[0]].add(red[1])
        all_options = list()
        
        for group in ("cpu", "gpu", "mem"):
            group_applied = applied_reduction_dict[group]
            options = [x for x in ["ColAgg", "RowAgg", "ColSamp", "RowSamp"] if x not in group_applied]
            for op in options:
                option_configs = self.get_strategy_available_configurations(op)
                for conf in option_configs:
                    all_options.append((group, op, conf))
        
        return all_options
            

    def apply_reduction(self, reduction_config):

        
        machine_type, reduction_strategy, kwargs = reduction_config
        df_to_reduce = self.df_dict["reduction_config"]

        labels = df_to_reduce.apply(
            lambda row: utils.philly_label_dict[row["status"]],
            axis=1
        )

        if reduction_strategy.endswith("Samp"):
            useful_cols = [int(x) for x in range(utils.philly_start_time, utils.philly_end_time + 1)]
            start_time = time.time()
            selected_idx = reductions.sampling_based_reduction(df_to_reduce[useful_cols].values, labels, **kwargs)
            end_time = time.time()
            passed_time = end_time - start_time
        else:
            pass

