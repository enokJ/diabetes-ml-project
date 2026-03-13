import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ZERO_AS_MISSING = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']


def get_default_data_path() -> Path:
    """Return the expected path for the dataset relative to the project root."""
    # This module lives in <project_root>/src/.
    return Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'diabetes.csv'


def load_and_split(path: str | Path | None = None, test_size=0.2, random_state=50):
    if path is None:
        path = get_default_data_path()

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download the Pima Indians Diabetes dataset and place it at this path:\n"
            "  httpswww.kaggle.c://om/datasets/uciml/pima-indians-diabetes-database"
        )

    df = pd.read_csv(path)

    # Replace impossible zeros with NaN BEFORE splitting
    df[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Stratified split — keeps class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

from sklearn.feature_selection import SelectKBest, f_classif

def build_preprocessor(k_features=8):
    """Return preprocessing steps for a pipeline, including feature selection.

    This is intentionally a list of (name, transformer) tuples so it can be
    composed into both sklearn and imblearn pipelines.
    """
    return [
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif, k=k_features))
    ]


