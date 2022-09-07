import reductions
import utils
import os
import json
import time
import pandas as pd
from sklearn.utils import resample
from data_reduce import DataReduce


if __name__ == "__main__":
    philly = PhillyDataReduce(sample_frac=0.01)
    print(philly.df_dict["cpu"].head())
