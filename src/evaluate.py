import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, ConfusionMatrixDisplay
)

from preprocess import load_and_split

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'diabetes.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

LR_PATH  = os.path.join(MODEL_DIR, 'lr_pipeline.joblib')
SVM_PATH = os.path.join(MODEL_DIR, 'svm_pipeline.joblib')

# ── Check models exist ─────────────────────────────────────────
for path in [LR_PATH, SVM_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing model: {path}\nRun src/train.py first."
        )

# ── Load data & models ─────────────────────────────────────────
_, X_test, _, y_test = load_and_split(DATA_PATH)

lr  = joblib.load(LR_PATH)
svm = joblib.load(SVM_PATH)
print("Models loaded successfully ✓")

# ── Evaluation function ────────────────────────────────────────
def evaluate(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred,
          target_names=['No Diabetes', 'Diabetes']))
    print(f"  ROC-AUC     : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"  MCC         : {matthews_corrcoef(y_test, y_pred):.4f}")
    print(f"  Sensitivity : {sensitivity:.4f}")
    print(f"  Specificity : {specificity:.4f}")

    return y_pred, y_prob

lr_pred,  lr_prob  = evaluate(lr,  X_test, y_test, "Logistic Regression")
svm_pred, svm_prob = evaluate(svm, X_test, y_test, "SVM")

# ── ROC Curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for name, prob, color in [("LR",  lr_prob,  "#2D6A4F"),
                           ("SVM", svm_prob, "#D62828")]:
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves", fontweight='bold')
ax.legend()

# ── Precision-Recall Curves ────────────────────────────────────
ax2 = axes[1]
for name, prob, color in [("LR",  lr_prob,  "#2D6A4F"),
                           ("SVM", svm_prob, "#D62828")]:
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
ax2.axhline(y=y_test.mean(), color='k', linestyle='--', lw=1, label='Baseline')
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curves", fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports', 'roc_pr_curves.png'), dpi=150)
plt.show()

# ── Confusion Matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, pred, cmap, name in [
    (axes[0], lr_pred,  'Greens', 'Logistic Regression'),
    (axes[1], svm_pred, 'Reds',   'SVM')
]:
    cm = confusion_matrix(y_test, pred)
    ConfusionMatrixDisplay(cm, display_labels=['No Diabetes','Diabetes']).plot(
        ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(name, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'reports', 'confusion_matrices.png'), dpi=150)
plt.show()

print("\nDone. Charts saved to reports/")


