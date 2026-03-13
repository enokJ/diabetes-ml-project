import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


