import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
ROOT = Path(__file__).resolve().parents[1]
data_path = ROOT / 'data' / 'raw' / 'diabetes.csv'
df = pd.read_csv(data_path)

# Add column names since the CSV has no headers
df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# Basic info
print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))
print("Class distribution:\n", df['Outcome'].value_counts())

# Handle zeros as missing
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

# Summary stats
print("\nSummary statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.savefig(ROOT / 'reports' / 'correlation_heatmap.png')
plt.close()

# Histograms
df.hist(figsize=(12, 10), bins=20)
plt.tight_layout()
plt.savefig(ROOT / 'reports' / 'histograms.png')
plt.close()

print("EDA plots saved to reports/")

# Boxplots for outliers
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(df.columns[:-1]):
    sns.boxplot(y=df[col], ax=axes[i//4, i%4])
plt.tight_layout()
plt.show()