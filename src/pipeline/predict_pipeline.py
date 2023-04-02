import sys
import pandas as pd
from src.exception import CustomException

from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        # load model and preprocessing pipeline
        self.model = load_object(os.path.join(os.getcwd(), "artifacts/model.pkl"))
        self.preprocessing_pipeline = load_object(
            os.path.join(os.getcwd(), "artifacts/preprocessor.pkl")
        )

    def predict(self, data: pd.DataFrame) -> list:
        transformed_df = self.preprocessing_pipeline.transform(data)
        predictions = self.model.predict(transformed_df)

        return predictions.tolist()


class CustomData:
    '''
    Class to create CustomData object from input JSON data received from API
    '''
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            # create custom data dictionary
            custom_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # return the custom input data as dataframe as this is the format the model expects
            return pd.DataFrame(custom_data_dict)

        except Exception as e:
            raise (e, sys)
