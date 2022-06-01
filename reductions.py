"""
Weifan Jiang, weifanjiang@g.harvard.edu
"""


import apricot


def submodular_function_optimization(X, Y, **kwargs):
    if kwargs["dir"] == 'col':
        X = X.transpose()
    tokeep = int(X.shape[0] * kwargs['keep_frac'])
    if kwargs['model'] == 'fls':
        clf = apricot.FacilityLocationSelection(tokeep).fit(X)
    else:  # fbs
        clf = 


def sampling_based_reductions(X, Y, method, **kwargs):
    mapper = {
        "smf": submodular_function_optimization
    }

    return mapper[method](X, Y, **kwargs)
