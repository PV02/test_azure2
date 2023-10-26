from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps

application = Flask(__name__, static_folder='static')

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/runscript', methods=['POST'])
def run_script():
    if 'file' not in request.files:
        return jsonify({"result": "No image uploaded", "filename": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"result": "No image selected", "filename": "No file selected"})

    # Process the uploaded file for prediction
    image = Image.open(file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    prediction = model.predict(data)
    index = np.argmax(prediction)
    predicted_label = class_names[index].strip()
    confidence_score = prediction[0][index]

    return jsonify({'result': predicted_label, 'confidence_score': float(confidence_score), 'filename': file.filename})

@application.route('/about-us')
def about_us():
    return render_template('about-us.html')

@application.route('/diseases')
def diseases():
    return render_template('diseases.html')

if __name__ == '__main__':
    application.run(debug=True)
