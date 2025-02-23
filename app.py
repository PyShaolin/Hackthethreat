import os
import time
import threading
import queue
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Threat Detection Logic (Deep Learning) ---

def generate_mock_logs(n=1000):
    data = {
        'source_ip': np.random.randint(1, 255, n),
        'destination_ip': np.random.randint(1, 255, n),
        'port': np.random.choice([22, 80, 443, 8080, 5000], n),
        'bytes_sent': np.random.randint(100, 10000, n),
        'bytes_received': np.random.randint(50, 5000, n),
        'anomaly': np.random.choice([0, 1], n, p=[0.95, 0.05])
    }
    return pd.DataFrame(data)

logs = generate_mock_logs()

features = ['source_ip', 'destination_ip', 'port', 'bytes_sent', 'bytes_received']
X = logs[features]
y = logs['anomaly']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# --- Prediction Function ---

def predict_anomaly(log_entry):
    log_scaled = scaler.transform(log_entry[features])
    prediction = model.predict(np.array(log_scaled))[0][0]
    return prediction

# --- Real-Time Alerting ---

alert_queue = queue.Queue()

def send_alert(log_entry, confidence):
    alert_message = f"Threat detected: {log_entry.to_dict()}, Confidence: {confidence:.4f}"
    alert_queue.put(alert_message)

def alert_worker():
    while True:
        alert = alert_queue.get()
        if alert is None:
            break
        print(f"ALERT: {alert}")
        alert_queue.task_done()

alert_thread = threading.Thread(target=alert_worker)
alert_thread.start()

for index, log_entry in logs.iterrows():
    log_df = pd.DataFrame([log_entry[features].values], columns=features)
    confidence = predict_anomaly(log_df)
    if confidence > 0.5:
        send_alert(log_entry, confidence)
    time.sleep(0.01)

alert_queue.put(None)
alert_thread.join()

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_dashboard_data')
def get_dashboard_data():
    total_logs = len(logs)
    threat_count = sum(logs['anomaly'])
    normal_count = total_logs - threat_count

    data = {
        'total_logs': total_logs,
        'threat_count': threat_count,
        'normal_count': normal_count,
        'cpu_usage': 25,
        'memory_usage': 60,
        'latency': 0.1
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)