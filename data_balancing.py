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
logger = setup_logging('data_balancing')
from imblearn.over_sampling import SMOTE

def dt_bal(trainig_data,y_train):
    try:
        logger.info(f'Training data shape : {trainig_data.shape}')
        logger.info(f'y_train shape : {y_train.shape}')
        logger.info(f' y_train data : {y_train}')

        sm = SMOTE(sampling_strategy=1.0,random_state=42)
        trainig_data,y_train = sm.fit_resample(trainig_data,y_train)

        logger.info(f'SMOTE training data shape : {trainig_data.shape}')
        logger.info(f'SMOTE y_train shape : {y_train.shape}')

        return trainig_data,y_train


    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')