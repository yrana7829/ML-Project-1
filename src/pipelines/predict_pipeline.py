import sys
import os
from src.exception import CustomException
from src.logger import logging
from utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass


    # Define the prediction function
    def predict(self, features):
        try:
            model_path = os.path.join('artifact', 'model.pkl')
            preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')
            model = load_object(file_path='artifact\model.pkl')
            preprocessor = load_object(file_path='artifact\preprocessor.pkl')
            print("After Loading")

            # Apply the preprocessor on the data before predictions
            preprocessed_data = preprocessor.transform(features)
            prediction = model.predict(preprocessed_data)
            return prediction
    
        except Exception as e:
            raise CustomException(e,sys)
         



class CustomData:
    def __init__(self, gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

        # define a function to convert this data as dataframe
    def get_data_as_df(self):
        try:
        # craete a dictionary
            custom_data_input_dict = {
            "gender":[self.gender],
            "race_ethnicity":[self.race_ethnicity],
            "parental_level_of_education":[self.parental_level_of_education],
            "lunch":[self.lunch],
            "test_preparation_course":[self.test_preparation_course],
            "reading_score":[self.reading_score],
            "writing_score":[self.writing_score]
                    }
            # convert it to df now
            return pd.DataFrame(custom_data_input_dict)
                
        except Exception as e:
            raise CustomException(e,sys)


