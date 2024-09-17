import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV

# Load the data
df = pd.read_csv('process_data.csv')

# Separate features and target
X = df.drop('cpu_burst_time', axis=1)
y = df['cpu_burst_time']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps
numeric_features = ['avg_previous_bursts', 'memory_usage', 'io_operations', 'system_load', 'time_of_day']
categorical_features = ['process_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    # Fit the pipeline on the entire dataset
    pipeline.fit(X, y)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'MSE': mse,
        'R2': r2,
        'CV R2': cv_scores.mean(),
        'CV R2 std': cv_scores.std(),
        'Pipeline': pipeline
    }

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R2: {metrics['R2']:.4f}")
    print(f"  CV R2: {metrics['CV R2']:.4f} (+/- {metrics['CV R2 std']:.4f})")
    print()

# Select the best model (you can change the criterion if needed)
best_model = max(results, key=lambda x: results[x]['CV R2'])
print(f"Best model based on cross-validated R2 score: {best_model}")

# Feature importance analysis
def plot_feature_importance(model_name, importance, feature_names):
    plt.figure(figsize=(10, 6))
    sorted_idx = importance.argsort()
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(feature_names)[sorted_idx])
    plt.title(f'Feature Importance - {model_name}')
    plt.tight_layout()
    plt.show()

# Analyze feature importance for Linear Regression and Random Forest
for model_name in ['Linear Regression', 'Random Forest']:
    pipeline = results[model_name]['Pipeline']
    feature_names = numeric_features + [f"{categorical_features[0]}_{cat}" for cat in pipeline.named_steps['preprocessor'].named_transformers_['cat'].categories_[0][1:]]
    
    if model_name == 'Linear Regression':
        importance = np.abs(pipeline.named_steps['regressor'].coef_)
    else:  # Random Forest
        importance = pipeline.named_steps['regressor'].feature_importances_
    
    plot_feature_importance(model_name, importance, feature_names)

# Perform permutation importance
perm_importance = permutation_importance(results['Random Forest']['Pipeline'], X_test, y_test, n_repeats=10, random_state=42)
plot_feature_importance('Random Forest (Permutation)', perm_importance.importances_mean, X.columns)

# Hyperparameter tuning for Random Forest
rf_param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}

rf_grid_search = GridSearchCV(results['Random Forest']['Pipeline'], rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
rf_grid_search.fit(X, y)

print("Best Random Forest parameters:", rf_grid_search.best_params_)
print("Best Random Forest R2 score:", rf_grid_search.best_score_)

# Residual analysis for Linear Regression
lr_pipeline = results['Linear Regression']['Pipeline']
y_pred_lr = lr_pipeline.predict(X)
residuals = y - y_pred_lr

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lr, residuals)
plt.xlabel('Predicted CPU Burst Time')
plt.ylabel('Residuals')
plt.title('Residual Plot - Linear Regression')
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
plt.show()

# Function to predict CPU burst time for new processes
def predict_cpu_burst(process_data):
    return rf_grid_search.best_estimator_.predict(process_data)

# Example usage:
# new_process = pd.DataFrame({
#     'process_type': ['CPU-bound'],
#     'avg_previous_bursts': [50],
#     'memory_usage': [500],
#     'io_operations': [10],
#     'system_load': [70],
#     'time_of_day': [14.5]
# })
# predicted_burst_time = predict_cpu_burst(new_process)
# print(f"Predicted CPU burst time: {predicted_burst_time[0]:.2f}")