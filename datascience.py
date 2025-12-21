import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           roc_auc_score, confusion_matrix, f1_score,
                           precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== LOAD DATA ====================
print("="*60)
print("LOADING AND EXPLORING DATA")
print("="*60)

# Load data (using the Google Drive link from second code)
df = pd.read_csv('https://drive.google.com/uc?export=download&id=1B0ehluDBErj66uvlr__TMqQOdnTcx9ry', low_memory=False)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# ==================== DATA CLEANING ====================
print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates().copy()
print(f"Removed {initial_rows - len(df)} duplicate rows")

# Check for missing values
missing = df.isnull().sum()
print(f"\nMissing values per column:")
print(missing[missing > 0])

# ==================== EXPLORATORY DATA ANALYSIS ====================
print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# 1. Target distribution
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
target_counts = df['Stress_Detection'].value_counts()
plt.bar(target_counts.index.astype(str), target_counts.values)
plt.title('Target Variable Distribution')
plt.xlabel('Stress Level')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
plt.title('Class Distribution')
plt.tight_layout()
plt.show()

print(f"Class distribution:\n{target_counts}")
print(f"Class imbalance ratio: {target_counts.max()/target_counts.min():.2f}:1")

# 2. Numeric features distribution
numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
if 'Stress_Detection' in numeric_cols:
    numeric_cols.remove('Stress_Detection')

if numeric_cols:
    n_cols = min(3, len(numeric_cols))
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(numeric_cols[:len(axes)]):
        axes[idx].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
    
    # Hide empty subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

# 3. Correlation matrix
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

# ==================== FEATURE ENGINEERING ====================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

# Clean Gender column
if 'Gender' in df.columns:
    df['Gender'] = (
        df['Gender'].astype(str)
        .str.strip()
        .str.lower()
        .replace({'male': 'Male', 'female': 'Female', 'm': 'Male', 'f': 'Female'})
    )
    print(f"Gender values: {df['Gender'].unique()}")

# Convert numeric-like columns
num_candidates = ['Blood_Pressure', 'Cholesterol_Level', 'Blood_Sugar_Level']
for col in num_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create Sleep Efficiency
if {'Sleep_Quality', 'Sleep_Duration'}.issubset(df.columns):
    df['Sleep_Efficiency'] = df['Sleep_Quality'] / df['Sleep_Duration'].replace(0, np.nan)
    print(f"Created Sleep_Efficiency feature")

# Create Daily Load
df['Daily_Load'] = 0
for col in ['Work_Hours', 'Travel_Time', 'Screen_Time']:
    if col in df.columns:
        df['Daily_Load'] += df[col].fillna(0)
print(f"Created Daily_Load feature")

# Create Health Score
if {'Meditation_Practice', 'Smoking_Habit', 'Alcohol_Intake'}.issubset(df.columns):
    # Standardize categorical values
    for col in ['Meditation_Practice', 'Smoking_Habit', 'Alcohol_Intake']:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].map({'yes': 1, 'no': 0, '1': 1, '0': 0, 'y': 1, 'n': 0}).fillna(0)
    
    df['Health_Score'] = (
        df['Meditation_Practice'] + 
        (1 - df['Smoking_Habit']) + 
        (1 - df['Alcohol_Intake'])
    ) / 3
    print(f"Created Health_Score feature")
else:
    df['Health_Score'] = 0.5  # Default neutral score

print(f"\nNew features created:")
new_features = ['Sleep_Efficiency', 'Daily_Load', 'Health_Score']
for feat in new_features:
    if feat in df.columns:
        print(f"  - {feat}: mean={df[feat].mean():.2f}, std={df[feat].std():.2f}")

# ==================== PREPARE DATA FOR MODELING ====================
print("\n" + "="*60)
print("PREPARING DATA FOR MODELING")
print("="*60)

# Define target
target = 'Stress_Detection'
if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset")

X = df.drop(columns=[target])
y = df[target].copy()

# Encode target variable
if y.dtype == 'object':
    y = y.str.strip()
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Encoded target variable. Classes: {le.classes_}")

# Identify feature types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"\nNumeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ==================== SPLIT DATA ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    stratify=y, 
    random_state=42
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train class distribution: {pd.Series(y_train).value_counts().to_dict()}")
print(f"Test class distribution: {pd.Series(y_test).value_counts().to_dict()}")

# ==================== CREATE PREPROCESSING PIPELINE ====================
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('numeric', numeric_pipeline, numeric_cols),
    ('categorical', categorical_pipeline, categorical_cols)
])

# ==================== MODEL 1: BASELINE WITHOUT SMOTE ====================
print("\n" + "="*60)
print("MODEL 1: BASELINE SVC (WITHOUT SMOTE)")
print("="*60)

baseline_svc = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(probability=True, random_state=42))
])

print("Training baseline model without SMOTE...")
baseline_svc.fit(X_train, y_train)

# Predictions
y_pred_baseline = baseline_svc.predict(X_test)
y_proba_baseline = baseline_svc.predict_proba(X_test)

# Calculate metrics
metrics_baseline = {
    'accuracy': accuracy_score(y_test, y_pred_baseline),
    'precision': precision_score(y_test, y_pred_baseline, average='weighted'),
    'recall': recall_score(y_test, y_pred_baseline, average='weighted'),
    'f1': f1_score(y_test, y_pred_baseline, average='weighted'),
    'roc_auc': roc_auc_score(y_test, y_proba_baseline, multi_class='ovr', average='weighted')
}

print("\n=== BASELINE PERFORMANCE (NO SMOTE) ===")
print(f"Accuracy:  {metrics_baseline['accuracy']:.4f}")
print(f"Precision: {metrics_baseline['precision']:.4f}")
print(f"Recall:    {metrics_baseline['recall']:.4f}")
print(f"F1-Score:  {metrics_baseline['f1']:.4f}")
print(f"ROC-AUC:   {metrics_baseline['roc_auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline))

# Confusion Matrix for baseline
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - Baseline (No SMOTE)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== MODEL 2: SVC WITH SMOTE ====================
print("\n" + "="*60)
print("MODEL 2: SVC WITH SMOTE OVERSAMPLING")
print("="*60)

# Create pipeline with SMOTE
svc_smote_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('classifier', SVC(probability=True, random_state=42))
])

print("Training model with SMOTE...")
svc_smote_pipeline.fit(X_train, y_train)

# Predictions
y_pred_smote = svc_smote_pipeline.predict(X_test)
y_proba_smote = svc_smote_pipeline.predict_proba(X_test)

# Calculate metrics
metrics_smote = {
    'accuracy': accuracy_score(y_test, y_pred_smote),
    'precision': precision_score(y_test, y_pred_smote, average='weighted'),
    'recall': recall_score(y_test, y_pred_smote, average='weighted'),
    'f1': f1_score(y_test, y_pred_smote, average='weighted'),
    'roc_auc': roc_auc_score(y_test, y_proba_smote, multi_class='ovr', average='weighted')
}

print("\n=== PERFORMANCE WITH SMOTE ===")
print(f"Accuracy:  {metrics_smote['accuracy']:.4f}")
print(f"Precision: {metrics_smote['precision']:.4f}")
print(f"Recall:    {metrics_smote['recall']:.4f}")
print(f"F1-Score:  {metrics_smote['f1']:.4f}")
print(f"ROC-AUC:   {metrics_smote['roc_auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote))

# Confusion Matrix with SMOTE
cm_smote = confusion_matrix(y_test, y_pred_smote)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_smote, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix - With SMOTE')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==================== HYPERPARAMETER TUNING WITH SMOTE ====================
print("\n" + "="*60)
print("MODEL 3: TUNED SVC WITH SMOTE (GRID SEARCH)")
print("="*60)

# Define parameter grid for tuning
param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'classifier__kernel': ['rbf', 'linear']
}

# Create Stratified K-Fold cross-validator
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create pipeline for tuning
tuning_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', SVC(probability=True, random_state=42))
])

# Perform Grid Search
print("Performing hyperparameter tuning with GridSearchCV...")
grid_search = GridSearchCV(
    tuning_pipeline,
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Get best model
best_model = grid_search.best_estimator_

# Predictions with tuned model
y_pred_tuned = best_model.predict(X_test)
y_proba_tuned = best_model.predict_proba(X_test)

# Calculate metrics for tuned model
metrics_tuned = {
    'accuracy': accuracy_score(y_test, y_pred_tuned),
    'precision': precision_score(y_test, y_pred_tuned, average='weighted'),
    'recall': recall_score(y_test, y_pred_tuned, average='weighted'),
    'f1': f1_score(y_test, y_pred_tuned, average='weighted'),
    'roc_auc': roc_auc_score(y_test, y_proba_tuned, multi_class='ovr', average='weighted')
}

print("\n=== TUNED MODEL PERFORMANCE ===")
print(f"Accuracy:  {metrics_tuned['accuracy']:.4f}")
print(f"Precision: {metrics_tuned['precision']:.4f}")
print(f"Recall:    {metrics_tuned['recall']:.4f}")
print(f"F1-Score:  {metrics_tuned['f1']:.4f}")
print(f"ROC-AUC:   {metrics_tuned['roc_auc']:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned))

# ==================== COMPARISON VISUALIZATION ====================
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Baseline (No SMOTE)': metrics_baseline,
    'With SMOTE': metrics_smote,
    'Tuned with SMOTE': metrics_tuned
}).T

print("\nPerformance Comparison:")
print(comparison_df)

# Visual comparison
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics_to_plot))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width, comparison_df.loc['Baseline (No SMOTE)', metrics_to_plot], 
                width, label='Baseline (No SMOTE)', color='skyblue')
rects2 = ax.bar(x, comparison_df.loc['With SMOTE', metrics_to_plot], 
                width, label='With SMOTE', color='lightgreen')
rects3 = ax.bar(x + width, comparison_df.loc['Tuned with SMOTE', metrics_to_plot], 
                width, label='Tuned with SMOTE', color='salmon')

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels([m.upper() for m in metrics_to_plot])
ax.legend()
ax.set_ylim([0, 1.1])

# Add value labels on bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()

# ==================== ADDITIONAL MODEL: DECISION TREE ====================
print("\n" + "="*60)
print("BONUS: DECISION TREE CLASSIFIER")
print("="*60)

# Decision Tree without SMOTE
dt_baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
])

dt_baseline.fit(X_train, y_train)
y_pred_dt = dt_baseline.predict(X_test)

print("\nDecision Tree (No SMOTE):")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

# ==================== FEATURE IMPORTANCE (if available) ====================
try:
    # For Decision Tree
    if hasattr(dt_baseline.named_steps['classifier'], 'feature_importances_'):
        # Get feature names after preprocessing
        preprocessor.fit(X_train)
        feature_names = []
        
        # Get numeric feature names
        feature_names.extend(numeric_cols)
        
        # Get categorical feature names
        encoder = preprocessor.named_transformers_['categorical'].named_steps['encoder']
        cat_feature_names = encoder.get_feature_names_out(categorical_cols)
        feature_names.extend(cat_feature_names)
        
        # Add engineered features if they're not already included
        for feat in ['Sleep_Efficiency', 'Daily_Load', 'Health_Score']:
            if feat in df.columns and feat not in feature_names:
                feature_names.append(feat)
        
        # Get feature importances
        importances = dt_baseline.named_steps['classifier'].feature_importances_
        
        # Create feature importance DataFrame
        feat_imp_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feat_imp_df)), feat_imp_df['importance'], color='teal')
        plt.yticks(range(len(feat_imp_df)), feat_imp_df['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances (Decision Tree)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
except Exception as e:
    print(f"Could not plot feature importance: {e}")

# ==================== FINAL SUMMARY ====================
print("\n" + "="*60)
print("FINAL RECOMMENDATIONS")
print("="*60)

print("\n1. BEST PERFORMING MODEL:")
best_model_name = comparison_df['f1'].idxmax()
print(f"   {best_model_name} with F1-Score: {comparison_df.loc[best_model_name, 'f1']:.4f}")

print("\n2. SMOTE IMPACT ANALYSIS:")
print(f"   F1-Score Improvement: {(metrics_smote['f1'] - metrics_baseline['f1']):.4f}")
print(f"   Recall Improvement: {(metrics_smote['recall'] - metrics_baseline['recall']):.4f}")

print("\n3. TUNING IMPACT:")
print(f"   Tuning improved F1-Score by: {(metrics_tuned['f1'] - metrics_smote['f1']):.4f}")

print("\n4. KEY INSIGHTS:")
print("   - SMOTE generally improves recall for minority classes")
print("   - Hyperparameter tuning can further optimize performance")
print("   - Consider business cost of false positives vs false negatives")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
