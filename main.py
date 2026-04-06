'''
In this project we are going to develop the AI-Powered Customer Retention Prediction System
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from Logging import setup_logging
logger = setup_logging('main')
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from Mode_technique import handle_missing_values
from variable_transformation import v_transformation
from outlier_handling import out_handling
from feature_selection import filter_method
from categorical_to_numerical import cat_to_numeric
from imblearn.over_sampling import SMOTE
from data_balancing import dt_bal
from feature_scaling import feature_sc
from Best_model_selection import all_model
from hyper_parameter import tuning

class CUSTOMER:
    def __init__(self,path):
        try:
            self.path = path
            self.df = pd.read_csv(path)
            logger.info(f'The shape of the data is : {self.df.shape}')
            logger.info(self.df.isnull().sum())
            logger.info(self.df.info())
            self.df['Telecom_Services'] = self.df['PaymentMethod'].copy()
            self.df['Telecom_Services'] = self.df['Telecom_Services'].map({'Electronic check':'jio','Mailed check':'Vi','Bank transfer (automatic)':'airtel','Credit card (automatic)':'BSNL'})
            logger.info(f'After adding Telecom_Services column :{self.df.shape} \n {self.df.columns}')
            logger.info(self.df['Telecom_Services'].unique())
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ', np.nan)
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
            logger.info('The null values:')
            logger.info(self.df['TotalCharges'].isnull().sum())

            self.X = self.df.drop(['Churn'],axis = 1) # Independent
            self.y = self.df['Churn'] # Dependent
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0}).astype(int)
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0}).astype(int)
            logger.info(f' shape of X_train: {self.X_train.shape} \n coluumns:{self.X.columns}')
            logger.info(
                f' Self.X_train : {len(self.X_train)} : self.y_train :{len(self.y_train)} \n Total training data : {self.X_train.shape}')
            logger.info(
                f'self.X_test  : {len(self.X_test)} : self.y_train : {len(self.y_test)} \n Total training data : {self.X_test.shape}')


        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def missing_values(self):
        try:
            logger.info(
                f"Before Handling NUll values X_train Column names and shape : {self.X_train.shape} \n : {self.X_train.columns} : {self.X_train.isnull().sum()}")
            logger.info(
                f"Before Handling NUll values X_test Column names and shape : {self.X_test.shape} \n : {self.X_test.columns} : {self.X_test.isnull().sum()}")
            self.X_train,self.X_test = handle_missing_values(self.X_train,self.X_test)
            logger.info(
                f"After Handling NUll values X_train Column names and shape : {self.X_train.shape} \n : {self.X_train.columns} : {self.X_train.isnull().sum()}")
            logger.info(
                f"After Handling NUll values X_test Column names and shape : {self.X_test.shape} \n : {self.X_test.columns} : {self.X_test.isnull().sum()}")

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def data_seperation(self):
        try:
            self.X_train_num_cols = self.X_train.select_dtypes(exclude='object')
            self.X_test_num_cols = self.X_test.select_dtypes(exclude='object')

            self.X_train_cat_cols = self.X_train.select_dtypes(include='object')
            self.X_test_cat_cols = self.X_test.select_dtypes(include='object')

            logger.info(f'X_train_num_cols:\n {self.X_train_num_cols.columns} : {self.X_train_num_cols.shape}')
            logger.info(f'X_test_num_cols:\n {self.X_test_num_cols.columns} : {self.X_test_num_cols.shape}')
            logger.info(f'X_train_cat_cols:\n {self.X_train_cat_cols.columns} : {self.X_train_cat_cols.shape}')
            logger.info(f'X_test_cat_cols:\n {self.X_test_cat_cols.columns} : {self.X_test_cat_cols.shape}')


        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def variable_transformation(self):
        try:
            logger.info(f'Before vairable transformation train column names : {self.X_train_num_cols.columns}')
            logger.info(f'Before variable transformation test column names : {self.X_test_num_cols.columns}')
            self.X_train_num_cols,self.X_test_num_cols = v_transformation(self.X_train_num_cols,self.X_test_num_cols)
            logger.info(f'After vairable transformation train column names : {self.X_train_num_cols.columns} and shape : {self.X_train_num_cols.shape}')
            logger.info(f'After variable transformation test column names : {self.X_test_num_cols.columns} and shape : {self.X_test_num_cols.shape}')

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def outlier(self):
        try:
            logger.info('---------------------Outlier Handling-----------------------------')
            logger.info(f'Before outlier training data shape : {self.X_train_num_cols.shape} \n Before outlier traning data columns : {self.X_train_num_cols.columns}')
            logger.info(f'Before outlier testing data shape : {self.X_test_num_cols.shape} \n Before outlier testing data columns : {self.X_test_num_cols.columns}')

            self.X_train_num_cols,self.X_test_num_cols = out_handling(self.X_train_num_cols,self.X_test_num_cols)

            logger.info(
                f'After outlier training data shape : {self.X_train_num_cols.shape} \n After outlier traning data columns : {self.X_train_num_cols.columns}')
            logger.info(
                f'After outlier testing data shape : {self.X_test_num_cols.shape} \n After outlier testing data columns : {self.X_test_num_cols.columns}')

        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def feature(self):
        try:
            logger.info('---------------------Feature Selection-----------------------------')
            logger.info(
                f'Before feature selection training(X_train) data shape : {self.X_train_num_cols.shape} \n Before feature selection training(X_train) data columns : {self.X_train_num_cols.columns}')
            logger.info(
                f'Before feature selection testing(X_test) data shape : {self.X_test_num_cols.shape} \n Before feature selection testing(X_test) data columns : {self.X_test_num_cols.columns}')

            self.X_train_num_cols,self.X_test_num_cols = filter_method(self.X_train_num_cols,self.X_test_num_cols,self.y_train,self.y_test)

            logger.info(
                f'After feature selection training(X_train) data shape : {self.X_train_num_cols.shape} \n After feature selection training(X_train) data columns : {self.X_train_num_cols.columns}')
            logger.info(
                f'After feature selection testing(X_test) data shape : {self.X_test_num_cols.shape} \n After feature selection testing(X_test) data columns : {self.X_test_num_cols.columns}')
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def cat_to_num(self):
        try:
            logger.info('---------------------Cat to Num Handling--------------------------')
            logger.info(f'Before Conversion the X_train categorical Columns are: {self.X_train_cat_cols.columns} and shape : {self.X_train_cat_cols.shape}')
            logger.info(f'Before conversion the X_test Categorical columns are: {self.X_test_cat_cols.columns} and shape : {self.X_test_cat_cols.shape}')

            self.X_train_cat_cols , self.X_test_cat_cols = cat_to_numeric(self.X_train_cat_cols,self.X_test_cat_cols)

            logger.info(
                f'After Conversion the X_train categorical Columns are: {self.X_train_cat_cols.columns} and shape : {self.X_train_cat_cols.shape}')
            logger.info(
                f'After conversion the X_test Categorical columns are: {self.X_test_cat_cols.columns} and shape : {self.X_test_cat_cols.shape}')

            logger.info('----------------------Combining the Data(categorical and numerical)-----------------------------')
            self.X_train_cat_cols.reset_index(drop=True,inplace=True)
            self.X_test_cat_cols.reset_index(drop=True,inplace=True)
            self.X_train_num_cols.reset_index(drop=True,inplace=True)
            self.X_test_num_cols.reset_index(drop=True,inplace=True)

            self.training_data = pd.concat([self.X_train_num_cols,self.X_train_cat_cols],axis = 1)
            self.testing_data = pd.concat([self.X_test_num_cols,self.X_test_cat_cols],axis = 1)

            logger.info(f'The complete training data after combining : {self.training_data.shape}')
            logger.info(f'the training data columns are : {self.training_data.columns}')
            logger.info(f'{self.training_data.isnull().sum()}')

            logger.info(f'The complete testing data after combining : {self.testing_data.shape}')
            logger.info(f'the testing data columns are : {self.testing_data.columns}')
            logger.info(f'{self.testing_data.isnull().sum()}')


        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def data_balance(self):
        try:
            logger.info('-----------------------Data Balancing------------------------------')
            logger.info(f" Before Data balancing The churn customer's(1) rows are : {sum(self.y_train == 1)}")
            logger.info(f" Before Data balancing The Non churn customer's(0) rows are : {sum(self.y_train == 0)}")
            logger.info(f'Before Data balancing Training data shape : {self.training_data.shape}')
            logger.info(f'{self.training_data.isnull().sum()}')

            self.training_data_bal,self.y_train_bal = dt_bal(self.training_data,self.y_train)

            logger.info(f"After Data balancing The churn customer's(1) rows are : {sum(self.y_train_bal)}")
            logger.info(f"After Data balancing The Non churn customer's(0) rows are : {sum(self.y_train_bal)}")
            logger.info(f'After Data balancing Training data shape : {self.training_data_bal.shape}')
            logger.info(f'{self.training_data_bal.isnull().sum()}')
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def featue_scaling(self):
        try:
            logger.info('---------------------Feature scaling------------------------------')
            logger.info(f'Before feature scaling training data shape : {self.training_data_bal.shape}')
            logger.info(f'Before feature scaling testing data shape : {self.testing_data.shape}')

            feature_sc(self.training_data_bal,self.y_train_bal,self.testing_data,self.y_test)

            #logger.info(f'After feature scaling training data : {self.training_data_bal_sc}')
            #logger.info(f'After feature scaling testing data : {self.testing_data_sc}')

            logger.info(f' Feature scaling completed Successfully ')


        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')

    def best_model(self):
        try:
            logger.info('--------------------- Finding Best Mode -------------------------------------')

            all_model(self.training_data_bal_sc,self.testing_data_sc, self.y_train_bal,self.y_test)

            logger.info(f' from the AUC and ROC curve  the best model is logistic Regression ')
            logger.info(f'----------------------Best Model Selection Completed Successfully ------------------------')
        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')


    def hyper_param(self):
        try:
            logger.info(f'-------------------Hyper parameter Tuning ------------------------------------------')

            tuning(self.training_data_bal_sc,self.y_train_bal,self.testing_data_sc,self.y_test)




        except Exception as e:
            err_type, err_msg, err_line = sys.exc_info()
            logger.info(f' Error in line no: {err_line.tb_lineno} due to : {err_msg}')



if __name__ == '__main__':
    obj = CUSTOMER('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    obj.missing_values()
    obj.data_seperation()
    obj.variable_transformation()
    obj.outlier()
    obj.feature()
    obj.cat_to_num()
    obj.data_balance()
    obj.featue_scaling()
    #obj.best_model()
    #obj.hyper_param()