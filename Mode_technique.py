import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from Logging import setup_logging
logger = setup_logging('Mode_technique')
import warnings
warnings.filterwarnings("ignore")

def handle_missing_values(X_train,X_test):
    try:
        logger.info(
            f"Before Handling NUll values X_train Column names and shape : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}")
        logger.info(
            f"Before Handling NUll values X_test Column names and shape : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}")
        for i in X_train.columns:
            if X_train[i].isnull().sum() > 0:
                X_train[i+'_replaced'] = X_train[i].copy()
                X_test[i+'_replaced'] = X_test[i].copy()
                value1 = X_train[i].mode()[0]
                value2 = X_test[i].mode()[0]
                X_train[i+'_replaced'].fillna(value1,inplace=True)
                X_test[i+'_replaced'].fillna(value2,inplace=True)
                X_train = X_train.drop([i],axis=1)
                X_test = X_test.drop([i],axis=1)

        logger.info(
            f"After Handling NUll values X_train Column names and shape : {X_train.shape} \n : {X_train.columns} : {X_train.isnull().sum()}")
        logger.info(
            f"After Handling NUll values X_test Column names and shape : {X_test.shape} \n : {X_test.columns} : {X_test.isnull().sum()}")
        return X_train,X_test

    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')
