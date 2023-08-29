"""
Model Service Module
This module defines a class for handling the machine learning model and text preprocessing.

Classes:
    ModelService: A class that provides methods for loading the model, preprocessing text data, and making predictions.
"""

import re
import os
import string

import mlflow
import pandas as pd
import unidecode
import contractions

from mlflow.tracking import MlflowClient
from utils.emoticons import EMOTICONS

MODEL_NAME = os.getenv("MODEL_NAME", "logistic-regression")
TRACKING_SERVER_HOST = os.getenv(
    "TRACKING_SERVER_HOST", "localhost"
)

client = MlflowClient(tracking_uri=f"http://{TRACKING_SERVER_HOST}:8888")
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:8888")


class ModelService:
    """
    Model Service Class
    This class encapsulates the machine learning model and provides methods for loading the model,
    preprocessing text data, and making predictions.

    Args:
        model_bucket (str): The name of the S3 bucket containing the model artifacts.
        experiment_id (int): The ID of the MLflow experiment containing the model.
        run_id (str): The ID of the MLflow run containing the model.

    Methods:
        get_model_location(): Get the S3 location of the model artifacts.
        load_model(): Load the machine learning model.
        prepare_data(data: str): Prepare input data for prediction by cleaning and transforming it.
        clean_text(text: str): Preprocess the given text by removing noise, special characters, etc.
        predict(data: str): Make a prediction using the loaded model on the provided data.
    """

    def __init__(self, stage='Staging'):
        self.clf = self.load_model(stage)

    def load_model(self, stage):
        """
        Load Model
        Load the machine learning model from the specified S3 location.

        Returns:
            Any: The loaded machine learning model.
        """

        mlflow_model = client.search_model_versions(
            filter_string=f"name = '{MODEL_NAME}'", order_by=["version_number DESC"]
        )
        for model in mlflow_model:
            if model.current_stage == stage:
                clf = mlflow.pyfunc.load_model(model_uri=f"models:/{model.name}/{model.version}")
                break
        return clf

    def prepare_data(self, data):
        """
        Prepare Data
        Preprocess input data for prediction by cleaning and transforming it.

        Args:
            data (str): Input data for prediction.

        Returns:
            dict: A dictionary containing preprocessed features.
        """
        features = {}
        features['cleaned_text'] = self.clean_text(data)
        df = pd.DataFrame(features, index=[0])
        return df

    def clean_text(self, text):
        """
        Clean Text
        Preprocess the given text by removing noise, special characters, URLs, etc.

        Args:
            text (str): Input text to be cleaned.

        Returns:
            str: Cleaned and preprocessed text.
        """
        # Convert the text to lowercase
        text = text.lower()

        # Remove HTML entities and special characters
        text = re.sub(r'(&amp;|&lt;|&gt;|\n|\t)', ' ', text)

        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # remove urls

        # Remove email addresses
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove dates in various formats (e.g., DD-MM-YYYY, MM/DD/YY)
        text = re.sub(r'\d{1,2}(st|nd|rd|th)?[-./]\d{1,2}[-./]\d{2,4}', ' ', text)

        # Remove month-day-year patterns (e.g., Jan 1st, 2022)
        pattern = re.compile(
            r'(\d{1,2})?(st|nd|rd|th)?[-./,]?\s?(of)?\s?([J|j]an(uary)?|[F|f]eb(ruary)?|[Mm]ar(ch)?|[Aa]pr(il)?|[Mm]ay|[Jj]un(e)?|[Jj]ul(y)?|[Aa]ug(ust)?|[Ss]ep(tember)?|[Oo]ct(ober)?|[Nn]ov(ember)?|[Dd]ec(ember)?)\s?(\d{1,2})?(st|nd|rd|th)?\s?[-./,]?\s?(\d{2,4})?'
        )
        text = pattern.sub(r' ', text)

        # Remove emoticons
        emoticons_pattern = re.compile(
            u'(' + u'|'.join(emo for emo in EMOTICONS) + u')'
        )
        text = emoticons_pattern.sub(r' ', text)

        # Remove mentions (@) and hashtags (#)
        text = re.sub(r'(@\S+|#\S+)', ' ', text)

        # Fix contractions (e.g., "I'm" becomes "I am")
        text = contractions.fix(text)

        # Remove punctuation
        PUNCTUATIONS = string.punctuation
        text = text.translate(str.maketrans('', '', PUNCTUATIONS))

        # Remove unicode
        text = unidecode.unidecode(text)

        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)

        return text

    def predict(self, data):
        """
        Predict
        Make a prediction using the loaded model on the provided data.

        Args:
            data (str): Input data for prediction.

        Returns:
            Any: The prediction result.
        """
        features = self.prepare_data(data)
        prediction = self.clf.predict(features['cleaned_text'])
        return prediction[0]
