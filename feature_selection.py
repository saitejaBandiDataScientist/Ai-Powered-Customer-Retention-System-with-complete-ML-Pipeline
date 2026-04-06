import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
from seaborn import boxplot
from scipy import stats
import warnings
warnings.filterwarnings("ignore")
import logging
from Logging import setup_logging
logger = setup_logging('feature_selection')
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

def filter_method(X_train_num_cols, X_test_num_cols,y_train,y_test):
    try:
        logger.info(f"Before Train Columns : {X_train_num_cols.shape} \n : {X_train_num_cols.columns}")
        logger.info(f"Before Test Columns : {X_test_num_cols.shape} \n : {X_test_num_cols.columns}")

        # constant Technique (Threshold as 0)
        var = VarianceThreshold(threshold=0)
        var.fit(X_train_num_cols)
        logger.info(f'The good columns(Train) from constant technique: {sum(var.get_support())} : {X_train_num_cols.columns[var.get_support()]}')
        logger.info(
            f'The good columns(Test) from constant technique: {sum(var.get_support())} : {X_test_num_cols.columns[var.get_support()]}')
        logger.info(
            f'The Bad columns(Train) from constant technique: {sum(~var.get_support())} : {X_train_num_cols.columns[~var.get_support()]}')
        logger.info(
            f'The Bad columns(Test) from constant technique: {sum(~var.get_support())} : {X_test_num_cols.columns[~var.get_support()]}')

        # Quasi Constant Technique (Threshold as 0.01)
        var2 = VarianceThreshold(threshold=0.01)
        var2.fit(X_test_num_cols)
        logger.info(
            f'The good columns(Train) from Quasi Constant technique: {sum(var2.get_support())} : {X_train_num_cols.columns[var2.get_support()]}')
        logger.info(
            f'The good columns(Test) from Quasi Constant technique: {sum(var2.get_support())} : {X_test_num_cols.columns[var2.get_support()]}')
        logger.info(
            f'The Bad columns(Train) from Quasi Constant technique: {sum(~var2.get_support())} : {X_train_num_cols.columns[~var2.get_support()]}')
        logger.info(
            f'The Bad columns(Test) from Quasi Constant technique: {sum(~var2.get_support())} : {X_test_num_cols.columns[~var2.get_support()]}')

        # Hypothesis Testing
        logger.info('---------------------------------Hypothesis Testing-----------------------------------------------------------')
        c = []
        for i in X_train_num_cols.columns:
            res = pearsonr(X_train_num_cols[i], y_train)
            c.append(res)
        t = np.array(c)
        p_value = pd.Series(t[:,1] , index = X_train_num_cols.columns)

        p = 0
        f = []
        for i in p_value:
            if i < 0.05:
                f.append(X_train_num_cols.columns[p])
            p = p+1
        logger.info(f'The Good Columns from Hypothesis Testing are : {f}')

        return X_train_num_cols,X_test_num_cols

    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')