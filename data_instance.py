"""
Weifan Jiang, weifanjiang@g.harvard.edu
"""


import pandas as pd
import reductions
import copy
from abc import ABC, abstractmethod


class SingleHopReduction:


    def __init__(
            self,
            red_type,
            granularities=None,
            method=None,
            sampling_args=None,
            agg_criterias=None,
            agg_func_options=None
        ):
        self.red_type = red_type
        self.granularities = granularities
        self.method = method
        self.sampling_args = sampling_args
        self.agg_criterias = agg_criterias
        self.agg_func_options = agg_func_options


    def is_sampling(self):
        return self.red_type == 'sampling'
    

    def is_aggregation(self):
        return self.red_type == 'aggregation'


class DataInstance(ABC):


    def __init__(self, df, metadatas, metrics, Y=None, applied_reductions=list()):
        self.df = df
        self.metadatas = metadatas
        self.metrics = metrics
        self.Y = Y

        self.applied_reductions = applied_reductions
    

    @abstractmethod
    def get_allowed_reductions(self):
        pass


    def apply_reduction(self, reduction_to_apply, inplace=False):
        
        new_df, new_Y = None, None
        if reduction_to_apply.is_sampling():
            new_df, new_Y = reductions.sampling_based_reduction_df(
                input_df=self.df,
                granularities=reduction_to_apply.granularities,
                metrics=self.metrics,
                method=reduction_to_apply.method,
                Y=self.Y,
                **reduction_to_apply.sampling_args
            )
        elif reduction_to_apply.is_aggregation():
            new_df = reductions.aggregation_based_reduction_df(
                input_df=self.df,
                agg_criterias=reduction_to_apply.agg_criterias,
                metrics=self.metrics,
                agg_func_options=reduction_to_apply.agg_func_options
            )
        
        new_columns = [x for x in new_df.columns]
        new_metrics = [x for x in new_columns if x not in self.metadatas]
        new_metadatas = [x for x in new_columns if x not in new_metrics]
        new_applied_reductions = copy.deepcopy(self.applied_reductions)
        new_applied_reductions.append(reduction_to_apply)

        if inplace:
            self.df = new_df
            self.metadatas = new_metadatas
            self.metrics = new_metrics
            self.Y = new_Y
            self.applied_reductions = new_applied_reductions
        else:
            return DataInstance(new_df, new_metadatas, new_metrics, new_Y, new_applied_reductions)


    def __str__(self):
        pass


    @abstractmethod
    def evaluation_oracle(self, **kwargs):
        pass
