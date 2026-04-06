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
logger = setup_logging('feature_scaling')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import pickle


def feature_sc(X_train, y_train,X_test,y_test):
    try:
        logger.info(f'Feature scaling started ...')
        logger.info(f' before feature scaling X_train : {X_train.columns} and shape : {X_train.shape}')
        sc = StandardScaler()
        sc.fit(X_train)
        X_train_sc = sc.transform(X_train)
        X_test_sc = sc.transform(X_test)

        with open('standard_scaler.pkl', 'wb') as f:
            pickle.dump(sc, f)

        lr_reg = LogisticRegression(C=1, class_weight='balanced', max_iter=100, penalty='l2', solver='sag')
        lr_reg.fit(X_train,y_train)
        predictions = lr_reg.predict(X_test)
        logger.info(f'confusion matrix : \n {confusion_matrix(y_test, predictions)}')
        logger.info(f'Accuracy score : {accuracy_score(y_test, predictions)}')
        logger.info(f'Classification report : \n {classification_report(y_test, predictions)}')

        with open('Model.pkl', 'wb') as t:
            pickle.dump(lr_reg, t)

        #logger.info(f' after feature scaling X_train: {X_train_sc}')

        #return X_train_sc, X_test_sc



    except Exception as e:
        err_type, err_msg, err_line = sys.exc_info()
        logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')