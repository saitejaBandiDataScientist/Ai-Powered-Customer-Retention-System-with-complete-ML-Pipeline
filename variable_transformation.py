import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import os
import sys
import seaborn as sns
from scipy.stats import boxcox
from scipy.stats import yeojohnson
from seaborn import boxplot
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


#modularization
import logging
from Logging import setup_logging
logger = setup_logging('v_transformation')


def v_transformation(X_train_nums_cols, X_test_nums_cols):
    try:
        logger.info(f'===================  Variable Transformation  =============================')
        logger.info(f'Before apply Train numerical columns and shapes variable Transformation \n : {X_train_nums_cols.columns} : {X_train_nums_cols.shape}')
        logger.info(f'Before apply Test numerical columns and shapes variable Transformation \n : {X_test_nums_cols.columns} : {X_test_nums_cols.shape}')
        for i in X_train_nums_cols.columns:
            if X_train_nums_cols[i].nunique()<=2: # SeniorCitizen is binary column( 0  and 1)
                logger.info(f'{i} column is binary column( 0 or 1)')
            elif i == 'tenure':
                X_train_nums_cols[i+'_yeo'], lam_val = yeojohnson(X_train_nums_cols[i])
                X_test_nums_cols[i+'_yeo'], lam_val = yeojohnson(X_test_nums_cols[i])

                # Visualization
                '''
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 4, 1)
                plt.title('Normal Distribution')
                X_train_nums_cols[i].plot(kind='kde', color='r')
                X_train_nums_cols[i+'_yeo'].plot(kind='kde', color='black')
                plt.subplot(1, 4, 2)
                sns.boxplot(x=X_train_nums_cols[i])
                sns.boxplot(x=X_train_nums_cols[i + '_yeo'])

                plt.subplot(1, 4, 3)
                plt.title("Probplot - Original")
                stats.probplot(X_train_nums_cols[i], dist="norm", plot=plt)

                plt.subplot(1, 4, 4)
                plt.title("Probplot - Yeo-Johnson")
                stats.probplot(X_train_nums_cols[i + '_yeo'], dist="norm", plot=plt)

                plt.legend()
                plt.show()
                '''

                X_train_nums_cols = X_train_nums_cols.drop(i,axis = 1)
                X_test_nums_cols = X_test_nums_cols.drop(i,axis = 1)
            elif i == 'MonthlyCharges':
                X_train_nums_cols[i+'_box'] ,lam_box = boxcox(X_train_nums_cols[i]+1)
                X_test_nums_cols[i+'_box'],lam_box = boxcox(X_test_nums_cols[i])

                # Visualization
                '''
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 4, 1)
                plt.title('Normal Distribution')
                X_train_nums_cols[i].plot(kind='kde', color='r')
                X_train_nums_cols[i + '_box'].plot(kind='kde', color='black')
                plt.subplot(1, 4, 2)
                sns.boxplot(x=X_train_nums_cols[i])
                sns.boxplot(x=X_train_nums_cols[i + '_box'])

                plt.subplot(1, 4, 3)
                plt.title("Probplot - Original")
                stats.probplot(X_train_nums_cols[i], dist="norm", plot=plt)

                plt.subplot(1, 4, 4)
                plt.title("Probplot - Box-Cox")
                stats.probplot(X_train_nums_cols[i + '_box'], dist="norm", plot=plt)

                plt.legend()
                plt.show()
                '''

                X_train_nums_cols = X_train_nums_cols.drop(i,axis = 1)
                X_test_nums_cols = X_test_nums_cols.drop(i,axis = 1)
            else :
                X_train_nums_cols[i+'_yeo'] ,lam_val = yeojohnson(X_train_nums_cols[i])
                X_test_nums_cols[i+'_yeo'] ,lam_val = yeojohnson(X_test_nums_cols[i])

                # Visualization
                '''
                plt.figure(figsize=(8, 3))
                plt.subplot(1, 4, 1)
                plt.title('Normal Distribution')
                X_train_nums_cols[i].plot(kind='kde', color='r')
                X_train_nums_cols[i + '_yeo'].plot(kind='kde', color='black')
                plt.subplot(1, 4, 2)
                sns.boxplot(x=X_train_nums_cols[i])
                sns.boxplot(x=X_train_nums_cols[i + '_yeo'])

                plt.subplot(1, 4, 3)
                plt.title("Probplot - Original")
                stats.probplot(X_train_nums_cols[i], dist="norm", plot=plt)

                plt.subplot(1, 4, 4)
                plt.title("Probplot - Yeo-Johnson")
                stats.probplot(X_train_nums_cols[i + '_yeo'], dist="norm", plot=plt)

                plt.legend()
                plt.show()
                '''


                X_train_nums_cols = X_train_nums_cols.drop(i,axis = 1)
                X_test_nums_cols = X_test_nums_cols.drop(i,axis = 1)



        logger.info(f'After apply Train numerical columns and shapes variable Transformation \n : {X_train_nums_cols.columns} : {X_train_nums_cols.shape}')
        logger.info(f'After apply Test numerical columns and shapes variable Transformation \n : {X_test_nums_cols.columns} : {X_test_nums_cols.shape}')

        return X_train_nums_cols, X_test_nums_cols



    except Exception as e:
        er_type, er_msg, er_line = sys.exc_info()
        logger.info(f'Error in lineno {er_line.tb_lineno} due to {er_type} and Reason {er_msg}')