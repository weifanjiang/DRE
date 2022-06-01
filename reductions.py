"""
Weifan Jiang, weifanjiang@g.harvard.edu
"""


import apricot
import numpy as np
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
    - dir: row or col
    - model: fls or fbs
    """
    if kwargs["dir"] == 'col':
        X = X.transpose()
    tokeep = int(X.shape[0] * kwargs['keepFrac'])
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
    - dir: row or col
    - sampler: volume, doublePhase or leverage
    """
    if kwargs["dir"] == 'row':
        X = X.transpose()
    tokeep = int(X.shape[0] * kwargs['keepFrac'])
    tokeep = min(tokeep, X.shape[0] - 1)
    tokeep = min(tokeep, X.shape[1] - 1)
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
    Active learning to sample rows in a batched fashion
    requires labeled dataset
    only for row sampling

    Required params:
    - model: RF, MLP, KNN (5 neighbors), KNNX (X = number of neighbors)
    - initFrac: fraction of target samples randomly selected to initialize active learning (default 0.1)
    - sampling: margin, entropy, uncertain (default), uncertainBatch, expected
    - nQueries: number of iterations of active learning (default 10)
    """
    selected = list()
    tokeep = int(X.shape[0] * kwargs["keepFrac"])
    model = kwargs["model"]
    if model == "RF":
        d2d = RandomForestClassifier()
    elif model == "MLP":
        d2d = MLPClassifier()
    elif model.startswith("KNN"):
        if model == "KNN":
            nn = 5
        else:
            nn = int(model.replace("KNN", ""))
        d2d = KNeighborsClassifier(n_neighbors=nn)
    
    n_initial = max(int(kwargs.get("initFrac", 0.1) * tokeep), 1)
    initial_idx = np.random.choice(range(X.shape[0]), size=n_initial, replace=False)
    selected.extend(initial_idx.tolist())

    X_initial = X[initial_idx]
    Y_initial = Y[initial_idx]
    X_pool = np.delete(X, initial_idx, axis=0)
    Y_pool = np.delete(Y, initial_idx, axis=0)

    strat = kwargs.get("sampling", "uncertain")
    if strat == "margin":
        samp = margin_sampling
    elif strat == "entropy":
        samp = entropy_sampling
    elif strat == "uncertain":
        samp = uncertainty_sampling
    elif strat == "uncertainBatch":
        samp = uncertainty_batch_sampling
    elif strat == "expected":
        samp = expected_error_reduction
    learner = ActiveLearner(estimator=d2d, query_strategy=samp, X_training=X_initial, y_training=Y_initial)
    
    n_queries = kwargs.get("nQueries", 10)
    n_instances = int((tokeep  - n_initial)/n_queries) + 1

    for idx in range(n_queries):
        query_idx, query_instance = learner.query(X_pool, n_instances=n_instances)
        learner.teach(X=X_pool[query_idx], y=Y_pool[query_idx], only_new=True)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        Y_pool = np.delete(Y_pool, query_idx, axis=0)
        selected.extend(query_idx)
    
    selected = selected[:tokeep]
    return selected


def sampling_based_reductions(X, Y, method, **kwargs):
    """
    Sampling based data reduction.
    Pass Y = None for sampling strategies that do not require labeled dataset

    method:
    - smf: submodular function optimization
    - ssp: subset selection problem
    - al: active learning

    kwargs should include:
    - keepFrac: fraction of rows/columns that the reduced data should keep
    - other argument that the specific sampling strategy requires
    """
    mapper = {
        "smf": submodular_function_optimization,
        "ssp": subset_selection_problem,
        "al": active_learning
    }

    return mapper[method](X, Y, **kwargs)
