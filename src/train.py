from pathlib import Path

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from preprocess import load_and_split, build_preprocessor
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

X_train, X_test, y_train, y_test = load_and_split()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── Logistic Regression ──────────────────────────────────────
# Build preprocessing + resampling pipeline.
# Note: imblearn.Pipeline does not support nested Pipeline objects.
lr_pipeline = ImbPipeline(build_preprocessor() + [
    ('smote', SMOTE(random_state=42)),     # SMOTE inside pipeline = safe
    ('model', LogisticRegression(max_iter=1000))
])

lr_params = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'model__l1_ratio': [0, 0.5, 1],  # 0 = L2, 1 = L1, 0.5 = elastic net
    'model__solver': ['saga']  # Saga supports l1_ratio; liblinear does not
}

lr_search = GridSearchCV(lr_pipeline, lr_params, cv=cv,
                         scoring='roc_auc', n_jobs=-1, verbose=1)
lr_search.fit(X_train, y_train)
print("Best LR AUC:", lr_search.best_score_.round(4))
print("Best LR params:", lr_search.best_params_)

# ── SVM ─────────────────────────────────────────────────────
svm_pipeline = ImbPipeline(build_preprocessor() + [
    ('smote', SMOTE(random_state=42)),
    ('model', SVC(probability=True, random_state=42))
])

svm_params = {'model__C': [0.1, 1, 10],
              'model__gamma': ['scale', 'auto'],
              'model__kernel': ['rbf', 'linear'],
              'model__degree': [3]}  # For poly kernel

svm_search = GridSearchCV(svm_pipeline, svm_params, cv=cv,
                          scoring='roc_auc', n_jobs=-1, verbose=1)
svm_search.fit(X_train, y_train)
print("Best SVM AUC:", svm_search.best_score_.round(4))

# Save both
joblib.dump(lr_search.best_estimator_, MODELS_DIR / 'lr_pipeline.joblib')
joblib.dump(svm_search.best_estimator_, MODELS_DIR / 'svm_pipeline.joblib')
