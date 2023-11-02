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
        """
        Parameters
        ----------
        FEATURES : list
            Features to use
        TARGET : list
            Target
        reg : sklearn model
            Scikit-learn regression model
        """

        self.FEATURES = FEATURES

        self.TARGET = TARGET

        self.reg = reg

        self.fitted = False

    def preprocess(self, data):
        df = pd.get_dummies(data)
        transformed_df = df[self.FEATURES + [self.TARGET]]
        return transformed_df

    def fit(self, data):
        df = self.preprocess(data)
        X = df[self.FEATURES]
        y = df[self.TARGET]
        self.reg.fit(X, y)
        self.fitted = True

    def predict(self, data):
        assert self.fitted, "Need to fit the model before making inference"
        df = self.preprocess(data)
        potential_grade = self.reg.predict(df[self.FEATURES])
        improvability_score = potential_grade - df[self.TARGET]
        return improvability_score


def compute_manual_score(data, coeff_dict: dict):
    """

    Parameters
    ----------
    data : Pandas dataframe
        DataFrame containing 'studytime', 'Dalc', 'absences'.
    coeff_dict : dict
        Coefficient of 'studytime', 'Dalc', 'absences' used to compute a weighted score.

    Returns
    -------
    score : ndarray
        Score of each student in the dataframe.

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
