from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (تأكد من تعديل المسار)
model = load_model(r'D:\Ammar\AMIT Diploma\Machine Leaning\ML Supervised (challenge)\Wonders_of_the_world_classfication.h5')

# Define the path for uploaded images
UPLOAD_FOLDER = r'D:\\Ammar\\AMIT Diploma\\Machine Leaning\\ML Supervised (challenge)\\uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Define the class names (تأكد من تعديل الأسماء حسب تصنيفاتك)
class_names = [
    'burj_khalifa',
    'chichen_itza',
    'christ_the_reedemer',
    'eiffel_tower',
    'great_wall_of_china',
    'machu_pichu',
    'pyramids_of_giza',
    'roman_colosseum',
    'statue_of_liberty',
    'stonehenge',
    'taj_mahal',
    'venezuela_angel_falls',
]

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the upload route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    # Save the file
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Load the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    # Generate the URL for the uploaded image
    image_url = f"/uploads/{file.filename}"  # استخدم المسار الصحيح للصورة

    # Display results in HTML page
    return render_template('result.html', prediction=predicted_class, image_url=image_url)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
