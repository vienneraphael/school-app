from typing import Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

base_path = "data/student_data.csv"


def assign_data(csv_file: str):
    if csv_file is None:
        csv_file = base_path
    return pd.read_csv(csv_file)


class ScoringModel:
    """Scoring model containing a preprocessor and a ML model. Can be used with differents features and models."""

    def __init__(self, FEATURES, TARGET, reg):
        self.FEATURES = FEATURES
        self.TARGET = TARGET
        self.reg = reg
        self.fitted = False

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataframe by making a one-hot encoding of all categorical features.

        Args:
            data (pd.DataFrame): the data.

        Returns:
            pd.DataFrame: preprocessed data.
        """
        df = pd.get_dummies(data)
        transformed_df = df[self.FEATURES + [self.TARGET]]
        return transformed_df

    def fit(self, data: pd.DataFrame) -> None:
        """
        Preprocess data and fit it to the model (Gradient Boosted Regressor).

        Args:
            data (pd.DataFrame): the raw data.
        """
        df = self.preprocess(data)
        X = df[self.FEATURES]
        y = df[self.TARGET]
        self.reg.fit(X, y)
        self.fitted = True

    def get_features_importance(self) -> Dict[str, float]:
        d = dict(zip(self.reg.feature_names_in_, self.reg.feature_importances_))
        return dict(sorted(d, key=lambda item: item[1]))

    def predict(self, data: pd.DataFrame) -> float:
        """
        given data, make predictions.
        The improvability score is given by: predicted grade - actual grade.
        This way it is higher for students with lower grades, and if the model predicts that this students should have a good grade based on his/her situation.

        Args:
            data (pd.DataFrame): the input data to be predicted

        Returns:
            float: improvability score.
        """
        assert self.fitted, "Need to fit the model before making inference"
        df = self.preprocess(data)
        potential_grade = self.reg.predict(df[self.FEATURES])
        improvability_score = potential_grade - df[self.TARGET]
        return improvability_score


def compute_manual_score(data, coeff_dict: dict):
    """
    Compute manual score based on study time, absences and alcohol consumption on weekdays.

        Args:
            data (pd.DataFrame): DataFrame containing 'studytime', 'Dalc', 'absences'.

        Returns:
            float: improvability score.
    """
    scaler = MinMaxScaler().fit_transform

    # Scale data and compute score. Dalc and absence impact negatively the score.
    score = (
        coeff_dict["studytime"] * scaler(data[["studytime"]])
        - coeff_dict["Dalc"] * scaler(data[["Dalc"]])
        - coeff_dict["absences"] * scaler(data[["absences"]])
    )
    # Score range from 0 to 10
    score = scaler(score).flatten() * 10
    return score
