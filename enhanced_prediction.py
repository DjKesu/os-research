import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

# Load the data
df = pd.read_csv('enhanced_process_data.csv')

# Prepare the target variable
y = df['process_type']
process_type_mapping = {'CPU-bound': 0, 'I/O-bound': 1, 'Mixed': 2}
y = y.map(process_type_mapping)

# Prepare the input features
numeric_features = ['avg_previous_bursts', 'memory_usage', 'io_operations', 'system_load', 'time_of_day', 
                    'priority', 'ready_queue_size', 'ready_queue_cpu_bound', 'ready_queue_io_bound', 'ready_queue_mixed']
X_numeric = df[numeric_features]

# Prepare the sequence data (last 100 decisions)
last_100_decisions = df['last_100_decisions'].apply(lambda x: [process_type_mapping[t] for t in x.split(',')])
last_100_decisions_padded = pad_sequences(last_100_decisions, maxlen=100, padding='pre')

# Split the data
X_numeric_train, X_numeric_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_numeric, last_100_decisions_padded, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_numeric_train_scaled = scaler.fit_transform(X_numeric_train)
X_numeric_test_scaled = scaler.transform(X_numeric_test)

# Define the model
model = Sequential([
    # Sequence input branch
    LSTM(64, input_shape=(100, 1)),
    
    # Numeric input branch
    Dense(64, activation='relu', input_shape=(len(numeric_features),)),
    
    # Merge branches
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: CPU-bound, I/O-bound, Mixed
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [X_seq_train, X_numeric_train_scaled], y_train,
    validation_data=([X_seq_test, X_numeric_test_scaled], y_test),
    epochs=10, batch_size=32
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X_seq_test, X_numeric_test_scaled], y_test)
print(f"Test accuracy: {test_accuracy:.4f}")

# Save the model and scaler
model.save('trained_model.h5')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved successfully.")