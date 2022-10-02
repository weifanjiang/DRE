import autosklearn.classification


automl = autosklearn.classification.AutoSklearnClassifier(
                    time_left_for_this_task=1000,
                    n_jobs=8,
                    per_run_time_limit=1000//6,
                    memory_limit=50000
                )
