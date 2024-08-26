import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

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
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
results = {}

for name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    # Fit the pipeline
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {'MSE': mse, 'R2': r2}

# Print results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"  MSE: {metrics['MSE']:.4f}")
    print(f"  R2: {metrics['R2']:.4f}")
    print()

# Select the best model (you can change the criterion if needed)
best_model = max(results, key=lambda x: results[x]['R2'])
print(f"Best model based on R2 score: {best_model}")

# Train the best model on the entire dataset
best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', models[best_model])
])
best_pipeline.fit(X, y)

# Function to predict CPU burst time for new processes
def predict_cpu_burst(process_data):
    return best_pipeline.predict(process_data)

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