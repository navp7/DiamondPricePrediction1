import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np 
import pandas as pd
from dataclasses import dataclass
from src.utils import save_object


## initialize the data transformation path:
@dataclass
class DataTransformationConfig: # Creating DataTransformationConfig class and creating a path for Preprocessor Object alonfg with file name 
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')  

# Creating Data Transformation Class:
# This is the main class which will be called from training.py file to Apply data transformation
# It contains all logical codes and pipelines, to perform imputing, scaling, ordinal encoding and transforming data
class DataTransformation:
    def __init__(self):   #Self init method to get the path from DataTranformationConfig class
        self.data_transformation_config = DataTransformationConfig() 

    def get_DataTransformation_obj(self):  # Method to preprocess and return a obj file/ pickle file
        try:
            logging.info("Data Transformation Initiated")
            
            # Dividing Categorical and Numerical Column:
            categorical_cols = ['cut', 'color', 'clarity']
            numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z'] 

            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ])
            ## Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinal',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                     ])

            ## Column Transformer
            preprocessor =ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline',cat_pipeline,categorical_cols),
                ])

            logging.info("Data Transformation completed")
            
            return preprocessor # Returning preprocessing file
        
        except Exception as e:
            logging.info("Exception occured at Data Transformation Stage")
            raise CustomException(e,sys)


# initiate_data_transformation: This method takes test,train path and it returns a preprocessed train and test dataframe along with preprocessed pkl file path.
# It takes data and splits independent and dependent features for both test and train dataset
# After that fit_transform and transform is applied on independent(train and test) data by using preprocessing_obj 
# Further the data is merged and we got a preprocessed train and test data set

    def initiate_data_transformation(self,train_data_path,test_data_path): 
        try:
            ##
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Train and test data read sucessful')
            logging.info(f'Train DataFrame Head: /n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame Head: /n{test_df.head().to_string()}')

            logging.info("Obtaining Preprocessor File/Object")
            preprocessing_obj = self.get_DataTransformation_obj()

            target_column ='price'
            drop_column = [target_column,'id']

            ## Dividing data into dependent and independent features:
            ## Training Data:
            input_train_df = train_df.drop(columns=drop_column,axis=1)
            target_train_df = train_df[target_column]

            ## Testing Data:
            input_test_df = test_df.drop(columns=drop_column,axis=1)
            target_test_df = test_df[target_column]
            ## Data Transformation:
            input_train_df_arr = preprocessing_obj.fit_transform(input_train_df)
            input_test_df_arr = preprocessing_obj.transform(input_test_df)

            train_arr = np.c_[input_train_df_arr,np.array(target_train_df)]
            test_arr = np.c_[input_test_df_arr,np.array(target_test_df)]
            
            # Calling save_object from utlis.py 
            # Saving preprocessing_obj at the artifacts destination
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            logging.info('Completing preprocessing on train and test data')
        except Exception as e:
            logging.info('Error occures at Data Transformation Stage')

            raise CustomException(e,sys)