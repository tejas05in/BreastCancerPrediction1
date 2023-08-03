from sklearn.impute import SimpleImputer ## handling missing values
from sklearn.preprocessing import StandardScaler ## handling feature scaling
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

##logging and exception
from src.logger import logging
from src.exception import CustomException
import sys , os

from dataclasses import dataclass
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data transformation initiated')

            numerical_cols = ['mean texture', 'mean area', 'mean smoothness', 'mean compactness',
                                'mean concavity', 'mean symmetry', 'mean fractal dimension',
                                'texture error', 'perimeter error', 'smoothness error',
                                'compactness error', 'concavity error', 'concave points error',
                                'symmetry error', 'fractal dimension error', 'worst smoothness',
                                'worst compactness', 'worst concavity', 'worst symmetry',
                                'worst fractal dimension']
            logging.info('Data Transformation Pipeline Initiated')
            
            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline , numerical_cols)
            ])

            logging.info("Data Transformation Completed")


            return preprocessor


        except Exception as e:
            logging.info('Exception occured in data transformation')
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_data_path,test_data_path):


        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data completed")
            logging.info(f'Train Datatrame Head : \n {train_df.head().to_string()}')
            logging.info(f'Test Datatrame Head : \n {test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformation_object()

            target_column = 'target'
            drop_columns = [target_column]
            ##dividing the dataset into independent and dependent features

            ## Training data
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column]

            ## Test data
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column]

            ## Data transformation
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )
            
            logging.info("Applying preprocessing object on training and testing dataset")

            return (
                train_arr ,
                test_arr ,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
            

        except Exception as e:

            raise CustomException(e,sys)



