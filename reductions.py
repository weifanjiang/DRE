"""
Weifan Jiang, weifanjiang@g.harvard.edu
"""


import apricot
import numpy as np
import pandas as pd
from CSSPy.volume_sampler import k_Volume_Sampling_Sampler
from CSSPy.doublephase_sampler import double_Phase_Sampler
from CSSPy.largest_leveragescores_sampler import largest_leveragescores_Sampler
from CSSPy.dataset_tools import calculate_right_eigenvectors_k_svd
from modAL.models import ActiveLearner
from modAL.batch import uncertainty_batch_sampling
from modAL.uncertainty import margin_sampling, entropy_sampling, uncertainty_sampling
from modAL.expected_error import expected_error_reduction
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier


def submodular_function_optimization(X, Y, **kwargs):
    """
    Submodular function optimization for row/column sampling
    does not require labeled dataset

    Required params:
    - model: fls or fbs
    """
    if kwargs["dir"] == 'col':
        X = X.transpose()
    if X.shape[0] == 1:
        return [0, ]
    tokeep = max(1, int(X.shape[0] * kwargs['keepFrac']))
    if kwargs['model'] == 'fls':
        clf = apricot.FacilityLocationSelection(tokeep).fit(X)
    elif kwargs['model'] == 'fbs':
        clf = apricot.FeatureBasedSelection(tokeep).fit(X)
    else:
        return None
    toret = clf.ranking
    selected = [int(i) for i in toret]
    if len(selected) > tokeep:
        selected = selected[:tokeep]
    return selected


def subset_selection_problem(X, Y, **kwargs):
    """
    Subset selection to optimize for the span of selected rows/columns
    does not require labeled dataset

    Required params:
    - sampler: volume, doublePhase or leverage
    """
    if kwargs["dir"] == 'row':
        X = X.transpose()

    if X.shape[0] <= 2 or X.shape[1] <= 2:
        return [x for x in range(X.shape[1])]

    tokeep = int(X.shape[0] * kwargs['keepFrac'])
    tokeep = min(tokeep, X.shape[0] - 1)
    tokeep = min(tokeep, X.shape[1] - 1)
    tokeep = max(2, tokeep)

    d = np.shape(X)[1]
    N = np.shape(X)[0] - 1
    _, D, V = np.linalg.svd(X)
    V_k = calculate_right_eigenvectors_k_svd(X, tokeep)

    if kwargs['sampler'] == 'volume':
        NAL = k_Volume_Sampling_Sampler(X, tokeep, D, V, d)
        A_S = NAL.MultiRounds()
    elif kwargs['sampler'] == 'doublePhase':
        NAL = double_Phase_Sampler(X, tokeep, V_k, d, 10*tokeep)
        A_S = NAL.DoublePhase()
    elif kwargs['sampler'] == 'leverage':
        NAL = largest_leveragescores_Sampler(X, tokeep, V, d)
        A_S = NAL.MultiRounds()
    
    selected = [int(x) for x in NAL.selected]
    if len(selected) > tokeep:
        selected = selected[:tokeep]
    return selected


def active_learning(X, Y, **kwargs):
    """
    Semi-supervised learning based sample selection
    """
    selected = list()
    tokeep = int(X.shape[0] * kwargs["keepFrac"])
    model = kwargs["model"]
    if model == "RF":
        d2d = RandomForestClassifier()
    elif model == "MLP":
        d2d = MLPClassifier()
    elif model == "KNN":
        d2d = KNeighborsClassifier(n_neighbors=5)

    # sample initial points
    n_initial = max(int(kwargs["initFrac"] * tokeep), 1)
    initial_idx = np.random.choice(range(X.shape[0]), size=n_initial, replace=False)
    selected.extend(initial_idx.tolist())
    X_initial = X[initial_idx]
    Y_initial = Y[initial_idx]
    X_pool = np.delete(X, initial_idx, axis=0)
    Y_pool = np.delete(Y, initial_idx, axis=0)

    # construct learner
    strat = kwargs["strat"]
    if strat == "margin":
        samp = margin_sampling
    elif strat == "entropy":
        samp = entropy_sampling
    elif strat == "uncertain":
        samp = uncertainty_batch_sampling
    elif strat == "expected":
        samp = expected_error_reduction
    learner = ActiveLearner(estimator=d2d, query_strategy=samp, X_training=X_initial, y_training=Y_initial)

    n_queries = 10
    n_instances = int((tokeep  - n_initial)/n_queries) + 1

    for _ in range(n_queries):
        query_idx, _ = learner.query(X_pool, n_instances=n_instances)
        learner.teach(X=X_pool[query_idx], y=Y_pool[query_idx], only_new=True)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        Y_pool = np.delete(Y_pool, query_idx, axis=0)
        selected.extend(query_idx)
    
    return [int(x) for x in selected][:tokeep]


def sampling_based_reduction(X, Y, **kwargs):
    """
    Returns list of selected col/row index
    """
    mapper = {
        "smf": submodular_function_optimization,
        "ssp": subset_selection_problem,
        "al": active_learning
    }

    return mapper[kwargs["method"]](X, Y, **kwargs)


# helper function for percentiles used in pandas
# https://stackoverflow.com/questions/17578115/pass-percentiles-to-pandas-agg-function
def get_percentile_fun(pct):

    def percentile_(x):
        return np.percentile(x, pct)
    percentile_.__name__ = "percentile_{}".format(pct)

    return percentile_


def aggregation_based_reduction(X, **kwargs):
    """
    Returns aggregated matrix X
    """

    # options for aggregation
    # 1: only common aggregation functions
    # 2: in addition to 1, add 4 percentiles
    # 3: in addition to 1 & 2, add 2 percentiles
    aggregation_options = {
        # function set 1: common
        1: [np.amax, np.amin, np.median, np.average, np.std],

        # function set 2: percentiles
        2: [10, 25, 75, 90],

        # function set 3: more detailed percentiles
        3: [1, 99]
    }

    aggregation_option_labels = {
        1: ["max", "min", "median", "avg", "std"],
        2: ["pct10", "pct25", "pct75", "pct90"],
        3: ["pct1", "pct99"]
    }

    if kwargs["dir"] == "col":
        processed, processed_labels = list(), list()
        for agg_option in range(1, kwargs["option"] + 1):
            funcs, func_labels = aggregation_options[agg_option], aggregation_option_labels[agg_option]
            if agg_option == 1:
                for func, func_label in zip(funcs, func_labels):
                    agg_stat = func(X.values, axis=1)
                    processed.append(agg_stat)
                    processed_labels.append(func_label)
            else:
                agg_stat = np.percentile(a=X.values, q=funcs, axis=1).transpose()
                processed.append(agg_stat)
                processed_labels.extend(func_labels)
        output_values = np.vstack(processed).transpose()
        
        if kwargs["prefix"] is not None:
            processed_labels = [kwargs["prefix"] + "_" + x for x in processed_labels]
        
        return pd.DataFrame(data=output_values, columns=processed_labels)
    
    else:  # row aggregation
        grb_criteria = kwargs["grb"]
        grb = X.groupby(by=grb_criteria)
        functions_to_apply = list()
        for agg_option in range(1, kwargs["option"] + 1):
            if agg_option == 1:
                functions_to_apply.extend(aggregation_options[agg_option])
            else:
                functions_to_apply.extend([
                    get_percentile_fun(x) for x in aggregation_options[agg_option]
                ])
        return grb.agg(functions_to_apply)
