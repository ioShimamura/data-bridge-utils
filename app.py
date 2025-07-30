from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
import joblib
import requests
import json
from datetime import datetime, timedelta
import pytz
import os
import logging

app = Flask(__name__)

# Configuración de logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuración de MySQL
db_config = {
    'host': '193.203.166.231',
    'user': 'u988743154_shima',
    'password': 'ticzi9-daqdot-gyFqep',
    'database': 'u988743154_DataManager'
}

# Configuración de OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_URL = 'https://api.openai.com/v1/chat/completions'
OPENAI_MODEL = 'gpt-4o'
OPENAI_PARAMS = {
    'temperature': 1,
    'max_tokens': 1024,
    'top_p': 1,
    'frequency_penalty': 0,
    'presence_penalty': 0
}

# Cargar modelo y scaler
knn = joblib.load('knn_model_fixed.joblib')
scaler = joblib.load('scaler_fixed.joblib')

# Mapeo inverso
reverse_label_mapping = {0: 'low', 1: 'medium', 2: 'high'}

@app.route('/send_data', methods=['POST'])
def send_data():
    data = request.get_json()
    logging.debug(f"Datos recibidos: {data}")

    user = data.get('user')
    gsr = data.get('gsr')
    heart_rate = data.get('heart_rate')
    skin_temp = data.get('skin_temperature')
    timestamp = data.get('timestamp')

    if None in (user, gsr, heart_rate, skin_temp, timestamp):
        return jsonify({'message': 'Missing fields'}), 400

    # Normalizar y predecir
    X = np.array([[gsr, heart_rate, skin_temp]])
    X_scaled = scaler.transform(X)
    prediction = knn.predict(X_scaled)[0]
    logging.debug(f"Predicción bruta: {prediction}")

    stress_label = reverse_label_mapping.get(prediction, "unknown")
    logging.debug(f"Etiqueta de estrés: {stress_label}")

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        insert_query = """
            INSERT INTO StressData (user, gsr, heart_rate, skin_temperature, timestamp, stress_label)
            VALUES (%s, %s, %s, %s, FROM_UNIXTIME(%s), %s)
        """
        cursor.execute(insert_query, (user, gsr, heart_rate, skin_temp, timestamp, stress_label))
        conn.commit()

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        logging.error(f"Error al insertar en la base de datos: {err}")
        return jsonify({'message': 'Database error'}), 500

    return jsonify({
        'message': 'Data processed and stored',
        'stress_label': stress_label,
        'advice': 'No change'
    })

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    user = request.args.get('user')
    logging.debug(f"Consulta de feedback para usuario: {user}")

    if not user:
        return jsonify({'message': 'User not provided'}), 400

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT stress_label
            FROM StressData
            WHERE user = %s AND stress_label IS NOT NULL AND stress_label != ''
            ORDER BY timestamp DESC
            LIMIT 1
        """, (user,))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row and row['stress_label']:
            label = row['stress_label']
            logging.debug(f"Última etiqueta encontrada: {label}")
            return jsonify({
                'stress_label': label,
                'advice': 'Mantener equilibrio emocional' if label == 'medium' else 'No change'
            })
        else:
            logging.warning("No se encontró una etiqueta válida")
            return jsonify({'stress_label': '', 'advice': 'No change'})

    except mysql.connector.Error as err:
        logging.error(f"Error al consultar la base de datos: {err}")
        return jsonify({'message': 'Database error'}), 500

if __name__ == '__main__':
    app.run(debug=True)
