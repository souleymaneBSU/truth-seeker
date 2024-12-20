import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# 1. Data Loading

# Load the features dataset
print("Loading features dataset...")
features_df = pd.read_csv('Features_For_Traditional_ML_Techniques.csv')

# Load the truth dataset
#print("Loading truth dataset...")
truth_df = pd.read_csv('Truth_Seeker_Model_Dataset.csv')

# Sample a subset of the data before merging
sample_frac = 0.1  # 1% sample
print(f"Sampling {sample_frac * 100}% of the features dataset...")
features_sample = features_df.sample(frac=sample_frac, random_state=42)

print(f"Sampling {sample_frac * 100}% of the truth dataset...")
truth_sample = truth_df.sample(frac=sample_frac, random_state=42)

#Reduce the features in truth_sample to only necessary columns
truth_sample = truth_sample[['statement', '5_label_majority_answer','3_label_majority_answer']]

# Merge datasets on 'statement'
print("Merging sampled datasets...")
data = pd.merge(
    features_sample,
    truth_sample,
    on='statement',
    how='inner'
)

# Ensure 'majority_target' exists in data
if 'majority_target' not in data.columns:
    print("Adding 'majority_target' from features_df to the merge.")
    # Since 'majority_target' is in features_df, include it in the merge
    data = pd.merge(
        features_sample[['statement', 'majority_target'] + list(features_sample.columns)],
        truth_sample,
        on='statement',
        how='inner'
    )

# 2. Data Preprocessing

# Remove rows with missing or 'NO MAJORITY' in '5_label_majority_answer'
print("Preprocessing data...")
data = data.dropna(subset=['5_label_majority_answer'])
data = data[data['5_label_majority_answer'] != 'NO MAJORITY']
data = data[data['5_label_majority_answer'] != 'Unrelated']
data = data.reset_index(drop=True)

# Remove rows with missing or 'NO MAJORITY' in '5_label_majority_answer'
print("Preprocessing data...")
data = data.dropna(subset=['3_label_majority_answer'])
data = data[data['3_label_majority_answer'] != 'NO MAJORITY']
data = data[data['3_label_majority_answer'] != 'Unrelated']
data = data.reset_index(drop=True)

# Map labels to numerical classes
label_mapping = {
    'Agree': 0,
    'Mostly Agree': 1,
    'Mostly Disagree': 2,
    'Disagree': 3
}

label_mapping_2 = {
    'Agree': 0,
    'Disagree': 1
}
data['5_class_target'] = data['5_label_majority_answer'].map(label_mapping)
data['3_class_target'] = data['3_label_majority_answer'].map(label_mapping_2)


# 3. Feature Selection

# Define feature columns (ensure they exist in the merged data)
feature_columns = [
    'followers_count', 'friends_count', 'favourites_count', 'statuses_count',
    'listed_count', 'BotScore', 'cred', 'normalize_influence', 'mentions',
    'quotes', 'replies', 'retweets', 'favourites', 'hashtags', 'URLs',
    'unique_count', 'total_count', 'Word count', 'Average word length',
    # Add more features if desired
]

# Remove columns that cannot be converted to numeric or do not exist
non_numeric_columns = ['tweet', 'embeddings', 'statement']
feature_columns = [col for col in feature_columns if col not in non_numeric_columns]
feature_columns = [col for col in feature_columns if col in data.columns]

# 4. Convert Feature Columns to Numeric
print("Converting feature columns to numeric...")
for col in feature_columns:
    print(f"Converting column: {col}")
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 5. Drop Rows with NaN in Feature Columns
print("Dropping rows with NaN values in feature columns...")
data = data.dropna(subset=feature_columns)

# Check if the dataset is empty after dropping NaNs
if data.empty:
    print("No data available after dropping NaNs. Please adjust the sample fraction or check the data.")
    exit()

# 6. Train/Test Split

print("Ensuring train and test datasets have different statements...")
# Group by statement to ensure rows with the same statement are treated together
grouped = data.groupby('statement')

# Assign each group a unique group ID
grouped_indices = {statement: i for i, statement in enumerate(grouped.groups.keys())}
data['group_id'] = data['statement'].map(grouped_indices)

# Split based on unique group IDs
train_groups, test_groups = train_test_split(
    list(grouped_indices.values()),
    test_size=0.2,
    random_state=42
)

# Create train and test datasets using the group IDs
train_data = data[data['group_id'].isin(train_groups)]
test_data = data[data['group_id'].isin(test_groups)]

# Drop the auxiliary 'group_id' column
train_data = train_data.drop(columns=['group_id'])
test_data = test_data.drop(columns=['group_id'])

# 7. Separate Features and Targets

X_train = train_data[feature_columns]
y_train_binary = train_data['3_class_target']
y_train_multi = train_data['5_class_target']

X_test = test_data[feature_columns]
y_test_binary = test_data['3_class_target']
y_test_multi = test_data['5_class_target']

# 8. Feature Scaling

print("Scaling feature data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9. Model Training and Evaluation

# a. For 2-Class Problem

print("\n=== Training and Evaluating 2-Class Problem ===")
# Initialize the Random Forest classifier
rf_binary = RandomForestClassifier(random_state=42)

# Define a simplified hyperparameter grid to reduce computation
param_grid_binary = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
}

# Initialize Grid Search with reduced CV and n_jobs
grid_search_binary = GridSearchCV(
    estimator=rf_binary,
    param_grid=param_grid_binary,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

# Fit Grid Search on training data
print("Starting Grid Search for 2-Class Problem...")
grid_search_binary.fit(X_train_scaled, y_train_binary)
print("Grid Search completed.")

# Best estimator
best_rf_binary = grid_search_binary.best_estimator_
print("Best Parameters:", grid_search_binary.best_params_)

# Predict on test data
y_pred_binary = best_rf_binary.predict(X_test_scaled)

# Evaluation metrics
accuracy_binary = accuracy_score(y_test_binary, y_pred_binary)
print(f"\nAccuracy: {accuracy_binary:.4f}")
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_binary))

# Confusion Matrix
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
disp_binary = ConfusionMatrixDisplay(confusion_matrix=cm_binary)
disp_binary.plot()
plt.title("2-Class Confusion Matrix")
plt.show()

# b. For 4-Class Problem

print("\n=== Training and Evaluating 4-Class Problem ===")
# Initialize the Random Forest classifier
rf_multi = RandomForestClassifier(random_state=42)

# Define a simplified hyperparameter grid
param_grid_multi = {
    'n_estimators': [100],
    'max_depth': [None],
    'min_samples_split': [2],
}

# Initialize Grid Search with reduced CV and n_jobs
grid_search_multi = GridSearchCV(
    estimator=rf_multi,
    param_grid=param_grid_multi,
    cv=3,
    scoring='accuracy',
    n_jobs=1,
    verbose=2
)

# Fit Grid Search on training data
print("Starting Grid Search for 4-Class Problem...")
grid_search_multi.fit(X_train_scaled, y_train_multi)
print("Grid Search completed.")

# Best estimator
best_rf_multi = grid_search_multi.best_estimator_
print("Best Parameters:", grid_search_multi.best_params_)

# Predict on test data
y_pred_multi = best_rf_multi.predict(X_test_scaled)

# Evaluation metrics
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
print(f"\nAccuracy: {accuracy_multi:.4f}")
print("Classification Report:")
print(
    classification_report(
        y_test_multi,
        y_pred_multi,
        target_names=list(label_mapping.keys()),
    )
)

# Confusion Matrix
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
disp_multi = ConfusionMatrixDisplay(
    confusion_matrix=cm_multi, display_labels=list(label_mapping.keys())
)
disp_multi.plot()
plt.title("4-Class Confusion Matrix")
plt.show()

# 10. Additional Steps to Improve Performance

# a. Feature Importance
print("\nCalculating feature importances for 2-Class Problem...")
importances = best_rf_binary.feature_importances_
feature_importance_df = pd.DataFrame(
    {'Feature': feature_columns, 'Importance': importances}
).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.title('Feature Importances for 2-Class Problem')
plt.show()

# b. Try Other Models (e.g., XGBoost)
print("\nTraining XGBoost classifier for 2-Class Problem...")
from xgboost import XGBClassifier

# Initialize XGBoost classifier
xgb_model = XGBClassifier(
    random_state=42, use_label_encoder=False, eval_metric='logloss'
)

# Fit on training data
xgb_model.fit(X_train_scaled, y_train_binary)

# Predict on test data
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluation metrics
accuracy_xgb = accuracy_score(y_test_binary, y_pred_xgb)
print(f"\nAccuracy: {accuracy_xgb:.4f}")
print("Classification Report:")
print(classification_report(y_test_binary, y_pred_xgb))

# 11. Save Models for Future Use
import joblib

# Save the best models
print("\nSaving trained models...")
joblib.dump(best_rf_binary, 'best_rf_binary.pkl')
joblib.dump(best_rf_multi, 'best_rf_multi.pkl')
