from flask import Flask, request, jsonify
import mysql.connector
import numpy as np
import joblib
import requests
import json
from datetime import datetime, timedelta
import pytz
import os  # Agregado para env vars

app = Flask(__name__)

# Configuración de MySQL
db_config = {
    'host': '193.203.166.231',  # O IP de tu servidor MySQL (actualizar para Render)
    'user': 'u988743154_shima',
    'password': 'ticzi9-daqdot-gyFqep',
    'database': 'u988743154_DataManager'
}

# Configuración de OpenAI (clave desde env var)
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
knn = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Mapeo inverso
reverse_label_mapping = {0: 'low', 1: 'medium', 2: 'high'}

# Función para clasificar momento del día
def get_momento_dia(timestamp_str):
    dt = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
    hour = dt.hour
    if hour < 12:
        return 'inicio de jornada'
    elif 12 <= hour < 15:
        return 'mitad de jornada'
    else:
        return 'fin de jornada'

# Función para obtener resumen de historial
def get_historial_summary(conn, user_id, timestamp_str):
    cursor = conn.cursor()
    one_week_ago = (datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S') - timedelta(days=7)).strftime('%Y-%m-%d %H:%M:%S')
    query = """
        SELECT COUNT(*) as high_count FROM StressData 
        WHERE user_id = %s AND stress_label IN ('medium', 'high') AND timestamp >= %s
    """
    cursor.execute(query, (user_id, one_week_ago))
    result = cursor.fetchone()
    high_count = result[0] if result else 0
    cursor.close()
    return f"{high_count} picos de estrés medio/alto en la última semana"

# Función para generar consejo con OpenAI
def generate_advice(stress_label, previous_label, gsr, heart_rate, skin_temperature, timestamp_str, historial_summary, preferences='prácticas'):
    momento_dia = get_momento_dia(timestamp_str)
    prompt = (
        f"Genera una recomendación breve y empática para un estudiante con estrés {stress_label}, que pasó de {previous_label}. "
        f"Incluye lecturas: GSR {gsr}, frecuencia cardíaca {heart_rate} bpm, temperatura {skin_temperature} °C. "
        f"Momento del día: {momento_dia}. Historial: {historial_summary}. Preferencias: {preferences}. "
        "Sé positivo, actionable, y aclara que no sustituye ayuda profesional. No induzcas culpa o ansiedad; refuerza respeto y cuidado emocional. "
        "Ejemplos: para bajo: 'Sigue así, tu cuerpo está en calma'; para medio: 'Respira profundo y haz una pausa breve'; para alto: 'Haz una pausa, no estás solo, considera hablar con un tutor.'"
    )

    data = {
        'model': OPENAI_MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        **OPENAI_PARAMS
    }

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {OPENAI_API_KEY}'
    }

    try:
        response = requests.post(OPENAI_URL, data=json.dumps(data), headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            print(f"Error en OpenAI: {response.status_code} - {response.text}")
            return "No advice available."
    except Exception as e:
        print(f"Error en generate_advice: {e}")
        return "No advice available."

# Endpoint /send_data
@app.route('/send_data', methods=['POST'])
def send_data():
    try:
        data = request.get_json()
        user_id = data.get('user')
        gsr = float(data.get('gsr'))
        heart_rate = float(data.get('heart_rate'))
        skin_temperature = float(data.get('skin_temperature'))
        timestamp_unix = data.get('timestamp')

        # Convertir a CDT
        tz = pytz.timezone('America/Mexico_City')
        timestamp = datetime.fromtimestamp(timestamp_unix, tz).strftime('%Y-%m-%d %H:%M:%S')

        # Clasificar
        input_data = np.array([[gsr, heart_rate, skin_temperature]])
        input_scaled = scaler.transform(input_data)
        stress_numeric = knn.predict(input_scaled)[0]
        stress_label = reverse_label_mapping.get(stress_numeric, 'unknown')

        # Conectar a BD
        conn = mysql.connector.connect(**db_config)
        
        # Obtener previous_label y historial
        cursor = conn.cursor()
        cursor.execute("SELECT stress_label FROM StressData WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        result = cursor.fetchone()
        previous_label = result[0] if result else None

        historial_summary = get_historial_summary(conn, user_id, timestamp) if previous_label else 'sin historial previo'

        # Generar consejo solo si hay cambio abrupto
        advice = "No change"
        if previous_label and previous_label != stress_label:
            if (previous_label == 'low' and stress_label in ['medium', 'high']) or \
               (previous_label == 'medium' and stress_label == 'high') or \
               (previous_label == 'high' and stress_label in ['medium', 'low']) or \
               (previous_label == 'medium' and stress_label == 'low'):
                advice = generate_advice(stress_label, previous_label, gsr, heart_rate, skin_temperature, timestamp, historial_summary)

        # Almacenar siempre
        query = """
            INSERT INTO StressData (user_id, gsr, heart_rate, skin_temperature, stress_label, advice, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (user_id, gsr, heart_rate, skin_temperature, stress_label, advice, timestamp))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': 'Data processed and stored', 'stress_label': stress_label, 'advice': advice}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# Endpoint /get_feedback
@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    try:
        user_id = request.args.get('user')
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        query = "SELECT stress_label, advice FROM StressData WHERE user_id = %s ORDER BY timestamp DESC LIMIT 1"
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        if result and result['advice'] and result['advice'] != "No change":
            return jsonify({'stress_label': result['stress_label'], 'advice': result['advice']}), 200
        else:
            return jsonify({'stress_label': result['stress_label'] if result else 'N/A', 'advice': 'No change'}), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
