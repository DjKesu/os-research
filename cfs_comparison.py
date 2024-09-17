import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
import joblib

def load_cfs_data(file_path):
    """
    Load and preprocess CFS scheduling data.
    Adjust this function based on the format of your CFS data.
    """
    df = pd.read_csv(file_path)
    
    # Assuming CFS data has similar columns to our simulated data
    numeric_features = ['avg_previous_bursts', 'memory_usage', 'io_operations', 'system_load', 'time_of_day', 
                        'priority', 'ready_queue_size', 'ready_queue_cpu_bound', 'ready_queue_io_bound', 'ready_queue_mixed']
    X_numeric = df[numeric_features]
    
    # Preprocess last 100 decisions
    process_type_mapping = {'CPU-bound': 0, 'I/O-bound': 1, 'Mixed': 2}
    last_100_decisions = df['last_100_decisions'].apply(lambda x: [process_type_mapping[t] for t in x.split(',')])
    last_100_decisions_padded = pad_sequences(last_100_decisions, maxlen=100, padding='pre')
    
    # Prepare actual decisions (target)
    y_actual = df['next_scheduled_process'].map(process_type_mapping)
    
    return X_numeric, last_100_decisions_padded, y_actual

def load_trained_model(model_path, scaler_path):
    """
    Load the trained model and scaler.
    """
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def make_predictions(model, scaler, X_numeric, X_seq):
    """
    Use the trained model to make predictions.
    """
    X_numeric_scaled = scaler.transform(X_numeric)
    predictions = model.predict([X_seq, X_numeric_scaled])
    return np.argmax(predictions, axis=1)

def compare_predictions(y_actual, y_pred):
    """
    Compare predictions with actual CFS decisions and print metrics.
    """
    accuracy = accuracy_score(y_actual, y_pred)
    conf_matrix = confusion_matrix(y_actual, y_pred)
    class_report = classification_report(y_actual, y_pred, target_names=['CPU-bound', 'I/O-bound', 'Mixed'])
    
    # Calculate F1 score
    f1 = f1_score(y_actual, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nDetailed Classification Report:")
    print(class_report)

def main():
    cfs_data_path = 'path_to_cfs_data.csv'  # Replace with actual path
    model_path = 'trained_model.h5'
    scaler_path = 'scaler.joblib'
    
    # Load CFS data
    X_numeric, X_seq, y_actual = load_cfs_data(cfs_data_path)
    
    # Load trained model and scaler
    model, scaler = load_trained_model(model_path, scaler_path)
    
    # Make predictions
    y_pred = make_predictions(model, scaler, X_numeric, X_seq)
    
    # Compare predictions with actual CFS decisions
    compare_predictions(y_actual, y_pred)

if __name__ == "__main__":
    main()