import os
import shutil

import pandas as pd
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

from src.utils import ocr_predict_date, ocr_predict_nominal, ocr_predict_type, convert_pdf_to_png, detect_png, \
    crop_image

app = Flask(__name__)
OCR_Model_Path = "./model/my_model_3.h5"
Object_Detection_Model_Path = "./model/my_model_3.weights.h5"
model = load_model(OCR_Model_Path)
list_label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
UPLOAD_FOLDER = 'uploads'  # Make sure this folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return "Welcome to the Python Web Server!"

@app.route('/predict', methods=['POST'])
def predict():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    convert_pdf_to_png(app, file)

    # If the user does not select a file, the browser submits an empty file without a filename
    # if file.filename == '':
    #     return jsonify({'error': 'No selected file'}), 400
    # if file and file.filename.endswith('.pdf'):
    #     # Save the uploaded PDF file
    #     pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    #     file.save(pdf_path)
    #
    #     pages = convert_from_path(pdf_path, 150)  # 150 dpi for decent quality
    #
    #     # Specify the output directory in /content
    #     output_dir = './content/pdf_2_png'
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir)
    #     os.makedirs(output_dir)
    #
    #     print(subprocess.run('python3 --version', shell=True, capture_output=True, text=True))
    #
    #     for i, page in enumerate(pages):
    #         output_path = os.path.join(output_dir, f'page_{i + 1}.png')
    #         page.save(output_path, 'PNG')
    detect_png()
    crop_image()

    dates = []
    nominals = []
    types = []

    # Get the list of image files in the directory
    files = sorted(os.listdir('./content/cropped_images'))

    # Iterate through the images
    for file in files:
        img_path = os.path.join('./content/cropped_images', file)

        # Check for class in the file name and apply the respective OCR function
        if 'class0' in file:  # Assuming class0 corresponds to Date
            recognized_date = ocr_predict_date(img_path, model, list_label)
            dates.append(recognized_date)

        elif 'class1' in file:  # Assuming class1 corresponds to Nominal
            recognized_nominal = ocr_predict_nominal(img_path, model, list_label)
            nominals.append(recognized_nominal)

        elif 'class2' in file:  # Assuming class2 corresponds to Type
            recognized_type = ocr_predict_type(img_path, model, list_label)
            types.append(recognized_type)

    # Create a DataFrame from the collected data
    df = pd.DataFrame({
        'Date': dates,
        'Nominal': nominals,
        'Type': types
    })
    return jsonify({
        'Date': dates,
        'Nominal': nominals,
        'Type': types
    })
    # return "hello world"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

