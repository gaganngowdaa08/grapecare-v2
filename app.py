import os
import random
from flask import Flask, render_template, request, session, redirect, url_for
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_very_secret_key_12345' 

# --- LOAD AI MODEL ---
try:
    model = load_model('grapecare_model.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Could not load 'grapecare_model.keras'.")
    model = None

CLASS_NAMES = ['black_rot', 'esca', 'healthy', 'leaf_blight']

def predict_image(image_path):
    if model is None: return {'label_key': 'error', 'confidence': 0.0, 'top_2': []}
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0]
        
        # Get top 2 predictions
        top_2_indices = np.argsort(prediction)[-2:][::-1]
        top_2 = []
        for i in top_2_indices:
            top_2.append({
                'label_key': CLASS_NAMES[i],
                'confidence': float(prediction[i])
            })

        return {
            'label_key': top_2[0]['label_key'],
            'confidence': top_2[0]['confidence'],
            'top_2': top_2 # Send top 2 predictions to template
        }
    except Exception as e:
        print(f"Prediction Error: {e}")
        return {'label_key': 'error', 'confidence': 0.0, 'top_2': []}

@app.route('/')
def index():
    result = session.pop('result_data', None)
    error = session.pop('error_msg', None)
    return render_template('index.html', result=result, error_msg=error)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        session['error_msg'] = 'No file uploaded.'
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        session['error_msg'] = 'No file selected.'
        return redirect(url_for('index'))
    if file:
        if not os.path.exists('static'): os.makedirs('static')
        file_path = os.path.join('static', 'temp_image.jpg')
        file.save(file_path)
        session['result_data'] = predict_image(file_path)
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)