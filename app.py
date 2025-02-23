import os
import time
import threading
import queue
import socket
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Real-Time Log Collection ---
def get_real_time_logs(n=100):
    logs = []
    for _ in range(n):
        log_entry = {
            'source_ip': socket.gethostbyname(socket.gethostname()),  # Get real system IP
            'destination_ip': np.random.randint(1, 255),  # Simulated external IP
            'port': np.random.choice([22, 80, 443, 8080, 5000]),  # Common ports
            'bytes_sent': psutil.net_io_counters().bytes_sent,  # Actual network usage
            'bytes_received': psutil.net_io_counters().bytes_recv,  # Actual network usage
            'anomaly': np.random.choice([0, 1], p=[0.98, 0.02])  # Assume 2% anomaly rate
        }
        logs.append(log_entry)

    return pd.DataFrame(logs)

logs = get_real_time_logs(n=500)

# --- Data Preprocessing ---
features = ['source_ip', 'destination_ip', 'port', 'bytes_sent', 'bytes_received']
X = logs[features]
y = logs['anomaly']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- AI Model ---
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

# --- Threat Prediction ---
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

# --- Continuous System Monitoring ---
def monitor_system():
    while True:
        live_logs = get_real_time_logs(n=1)
        log_scaled = scaler.transform(live_logs[features])
        confidence = model.predict(np.array(log_scaled))[0][0]

        if confidence > 0.5:
            send_alert(live_logs.iloc[0], confidence)

        time.sleep(2)  # Monitor every 2 seconds

monitoring_thread = threading.Thread(target=monitor_system)
monitoring_thread.start()

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_dashboard_data')
def get_dashboard_data():
    live_logs = get_real_time_logs(n=1)

    data = {
        'total_logs': len(logs),
        'threat_count': sum(logs['anomaly']),
        'normal_count': len(logs) - sum(logs['anomaly']),
        'cpu_usage': psutil.cpu_percent(interval=1),
        'memory_usage': psutil.virtual_memory().percent,
        'latency': np.random.uniform(0.1, 0.5)
    }
    
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
