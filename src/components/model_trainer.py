import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import  accuracy_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model
from dataclasses import dataclass
import os , sys

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerconfig()

    def initiate_model_training(self,train_arr,test_arr):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            #based on scree plot the best number of principal components = 5
            pca = PCA(n_components=5)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)
            logging.info('X data has undergone PCA transformation into 5 features')

            models={
            'LogisticRegression':LogisticRegression(),
            'RidgeClassifier':RidgeClassifier(),
            'BernoulliNB':BernoulliNB(),
            'DecisionTreeClassifier':DecisionTreeClassifier(),
            'KNeighborsClassifier':KNeighborsClassifier(),
            'AdaBoostClassifier':AdaBoostClassifier(),
            'GradientBoostingClassifier':GradientBoostingClassifier(),
            'BaggingClassifier':BaggingClassifier(),
            'RandomForestClassifier':RandomForestClassifier(),
            'SVC':SVC(),
            'XGBClassifier':XGBClassifier()
            }

            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            print('\n=================================================================================')
            logging.info(f'Model Report: {model_report}')

            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')
            print('\n==================================================================================')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , Accuracy Score : {best_model_score}')

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            logging.info("Model training complete and best model saved in artifacts as model.pkl")
        except Exception as e:
            raise CustomException(e,sys)

