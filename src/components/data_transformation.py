import os
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transform_config = DataTransformationConfig()

    # Defining a function to fenerate the transformation object, which can then be applied to anyd ataset to transform it
    def get_data_transform_object(self):
        logging.info("Data transformation initiated")
        try:
            # Get the features seperatey
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            # Now define the operations to be done on both type of features in the form of pipeline 
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical features scaling completed")

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("encoder", OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False)),
                ]
            )

            logging.info("Categorical features encoding completed")

            logging.info(" Combining both the pipelines by Column Transformer and building the preprocessor")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])
            logging.info("preprocessor built, ready to transform the data")

            return preprocessor

    
        except Exception as e:
            raise CustomException(e,sys)
        
    # Defining the main function to initiate the data transformation
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data transformation initiated")
            # a) Read the data from dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data is read inside transformation function")

            # b) Obtain the preprocessor object from above defined function
            preprocessor_obj = self.get_data_transform_object()

            # c) define the target column
            target_column_name = 'math_score'
            numeric_columns = ['reading_score', 'writing_score']
            cat_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            # d) define the input and output features or X and Y as df
            input_train = train_df.drop(columns=[target_column_name], axis = 1)
            output_train = train_df[target_column_name]

            input_test = test_df.drop(columns=[target_column_name], axis= 1)
            output_test = test_df[target_column_name]

            # e) applying the preprocessor now
            logging.info("Preprocessor application started")

            X_train_array = preprocessor_obj.fit_transform(input_train)
            X_test_array = preprocessor_obj.transform(input_test)

            # f) Combine the input and trgte features
            print(X_test_array.shape)
            print(output_test.shape)
            train_array = np.c_[X_train_array, np.array(output_train)]
            test_array = np.c_[X_test_array, np.array(output_test)]


            # g) Save the preprocessor in a pickle file
            save_object(
                file_path=self.data_transform_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("Preprocessing finished, returning the arrays of preprocessor train and test data")

            # h) Return the arrays of train and test data
            return (
                train_array,
                test_array,
                # self.data_transform_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            raise CustomException(e,sys)
