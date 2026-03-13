# Diabetes Prediction ML Project

This project uses machine learning to predict diabetes based on the Pima Indians Diabetes Dataset.

## Setup

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt` (create if needed)
5. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it at `data/raw/diabetes.csv`
6. Run preprocessing: `python src/preprocess.py`
7. Train models: `python src/train.py`
8. Evaluate: `python src/evaluate.py`

## Files

- `src/preprocess.py`: Data loading and preprocessing
- `src/train.py`: Model training with hyperparameter tuning
- `src/evaluate.py`: Model evaluation
- `src/eda.py`: Exploratory data analysis

## Models

Trained models are saved in `models/` directory.