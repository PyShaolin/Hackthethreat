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

# --- Convert IP to Numeric ---
def ip_to_int(ip):
    try:
        return int.from_bytes(socket.inet_aton(ip), 'big')
    except OSError:
        return 0  # Return 0 if IP conversion fails

# --- Generate Real-Time Logs ---
def get_real_time_logs(n=100):
    logs = []
    for _ in range(n):
        src_ip = socket.gethostbyname(socket.gethostname())  # Get system IP
        dest_ip = f"192.168.1.{np.random.randint(1, 255)}"  # Simulated external IP
        
        log_entry = {
            'source_ip': ip_to_int(src_ip),
            'destination_ip': ip_to_int(dest_ip),
            'port': np.random.choice([22, 80, 443, 8080, 5000]),  # Common ports
            'bytes_sent': psutil.net_io_counters().bytes_sent,
            'bytes_received': psutil.net_io_counters().bytes_recv,
            'anomaly': np.random.choice([0, 1], p=[0.98, 0.02])  # 2% anomaly rate
        }
        logs.append(log_entry)

    return pd.DataFrame(logs)

# --- Initial Dataset ---
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
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# --- Threat Prediction ---
def predict_anomaly(log_entry):
    log_entry_numeric = log_entry.copy()
    log_entry_numeric['source_ip'] = ip_to_int(log_entry['source_ip'])
    log_entry_numeric['destination_ip'] = ip_to_int(log_entry['destination_ip'])

    log_df = pd.DataFrame([log_entry_numeric])
    log_scaled = scaler.transform(log_df[features])
    
    prediction = model.predict(log_scaled)[0][0]
    return prediction

# --- Alert Queue ---
alert_queue = queue.Queue()

def send_alert(log_entry, confidence):
    alert_message = f"Threat detected: {log_entry}, Confidence: {confidence:.4f}"
    alert_queue.put(alert_message)

def alert_worker():
    while True:
        alert = alert_queue.get()
        if alert is None:
            break
        print(f"🚨 ALERT: {alert}")
        alert_queue.task_done()

alert_thread = threading.Thread(target=alert_worker, daemon=True)
alert_thread.start()

# --- System Monitoring ---
def monitor_system():
    while True:
        live_logs = get_real_time_logs(n=1)
        log_entry = live_logs.iloc[0].to_dict()
        confidence = predict_anomaly(log_entry)

        if confidence > 0.5:
            send_alert(log_entry, confidence)

        time.sleep(2)  # Monitor every 2 seconds

monitoring_thread = threading.Thread(target=monitor_system, daemon=True)
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
        'latency': round(np.random.uniform(0.1, 0.5) * 1000, 2)  # Latency in ms
    }
    
    return jsonify(data)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

