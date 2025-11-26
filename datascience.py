import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/content/stress_detection_data.csv', low_memory=False)
df = df.drop_duplicates().copy()

if 'Gender' in df.columns:
    df['Gender'] = (df['Gender'].astype(str).str.strip().str.lower()
                    .replace({'male':'Male','female':'Female'}))

# convert numeric-like columns to numeric safely
num_candidates = ['Blood_Pressure','Cholesterol_Level','Blood_Sugar_Level']
for c in num_candidates:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')



numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
target = 'Stress_Detection'
if target in numeric_cols:
    numeric_cols.remove(target)
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
if target in cat_cols:
    cat_cols.remove(target)

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numeric_cols),
    ('cat', cat_pipeline, cat_cols)
], remainder='drop')


df['Sleep_Efficiency'] = np.nan
if {'Sleep_Quality','Sleep_Duration'}.issubset(df.columns):
    df['Sleep_Efficiency'] = df['Sleep_Quality'] / df['Sleep_Duration']

df['Daily_Load'] = 0
for c in ['Work_Hours','Travel_Time','Screen_Time']:
    if c in df.columns:
        df['Daily_Load'] = df['Daily_Load'] + df[c].fillna(0)

if {'Meditation_Practice','Smoking_Habit','Alcohol_Intake'}.issubset(df.columns):
    # convert to numeric flags first if they are strings like 'Yes'/'No'
    for col in ['Meditation_Practice','Smoking_Habit','Alcohol_Intake']:
        if df[col].dtype == 'object':
            df[col] = df[col].map({'yes':1,'no':0,'Yes':1,'No':0}).fillna(0)
    df['Health_Score'] = (df['Meditation_Practice'] + (1 - df['Smoking_Habit']) + (1 - df['Alcohol_Intake'])) / 3
else:
    df['Health_Score'] = 0.0

# recompute cols lists after feature engineering
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if target in numeric_cols:
    numeric_cols.remove(target)
cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
if target in cat_cols:
    cat_cols.remove(target)


# === prepare X, y ===
X = df.drop(columns=[target])
y = df[target].copy()

# if target is categorical, ensure it's encoded correctly
if y.dtype == 'object':
    y = y.str.strip()
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

# === train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42)

# === full pipeline with classifier ===
svc_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', SVC(probability=True, random_state=42))
])

dt_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'clf__C': [0.1, 1, 10],
    'clf__gamma': ['scale','auto'],
    'clf__kernel': ['rbf','linear']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(svc_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best params (SVC):", grid.best_params_)
best_svc = grid.best_estimator_

# Evaluate on test set
y_pred = best_svc.predict(X_test)
y_proba = best_svc.predict_proba(X_test)[:,1] if hasattr(best_svc.named_steps['clf'],'predict_proba') else None

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
if y_proba is not None:
    print("ROC AUC:", roc_auc_score(y_test, y_proba))

dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)
print("Decision Tree accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
