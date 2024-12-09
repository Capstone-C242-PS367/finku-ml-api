import os
import shutil
import tempfile
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

from src.utils import ocr_predict_date, ocr_predict_nominal, ocr_predict_type, convert_pdf_to_png, detect_png, \
    crop_image, convert_jpg_to_png, copy_file_png, download_blob

app = Flask(__name__)
BUCKET_NAME = 'finku-model'
MODEL_NAME = 'my_model_3.h5'
WEIGHTS_NAME = 'my_model_3.weights.h5'

with tempfile.TemporaryDirectory() as tmpdirname:
    OCR_Model_Path = os.path.join(tmpdirname, MODEL_NAME)
    Object_Detection_Model_Path = os.path.join(tmpdirname, WEIGHTS_NAME)

    download_blob(BUCKET_NAME, MODEL_NAME, OCR_Model_Path)
    download_blob(BUCKET_NAME, WEIGHTS_NAME, Object_Detection_Model_Path)

    model = load_model(OCR_Model_Path)

list_label = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def home():
    return "Welcome to the Finku ML Server!"


@app.route('/predict', methods=['POST'])
def predict():
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER)

    file = request.files['file']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    output_dir = './content/png_image'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if file.content_type == 'application/pdf':
        convert_pdf_to_png(file_path, output_dir)
    elif file.content_type == 'image/jpeg':
        convert_jpg_to_png(file_path, output_dir)
    elif file.content_type == 'image/png':
        copy_file_png(file_path, output_dir)
    else:
        os.remove(file_path)
        return jsonify({'error': 'Only PDF, PNG, and JPG files are supported'}), 400

    detect_png()
    crop_image()

    dates = []
    nominals = []
    types = []

    files = sorted(os.listdir('./content/cropped_images'))

    for file in files:
        img_path = os.path.join('./content/cropped_images', file)

        if 'class0' in file:
            recognized_date = ocr_predict_date(img_path, model, list_label)
            dates.append(recognized_date)

        elif 'class1' in file:
            recognized_nominal = ocr_predict_nominal(img_path, model, list_label)
            nominals.append(recognized_nominal)

        elif 'class2' in file:
            recognized_type = ocr_predict_type(img_path, model, list_label)
            types.append(recognized_type)

    return jsonify({
        'Date': dates,
        'Nominal': nominals,
        'Type': types
    })


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
