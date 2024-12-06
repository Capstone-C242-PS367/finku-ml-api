import os
import shutil
import subprocess

import cv2
import imutils
from flask import jsonify
import numpy as np
from pdf2image import convert_from_path


def sort_bounding_boxes_with_tolerance(boxes, y_tolerance=5):
    """
    Sort bounding boxes primarily by y-coordinate and secondarily by x-coordinate,
    allowing a small tolerance for misalignments in the vertical position (y).
    """
    # First, sort the bounding boxes by y-coordinate and x-coordinate
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    # List to store the final sorted boxes with adjustments for y tolerance
    sorted_boxes_adjusted = []

    last_y = None
    line = []  # Temporary list to hold boxes in the same y-line
    for (x, y, w, h) in sorted_boxes:
        if last_y is None or abs(last_y - y) <= y_tolerance:
            # Add character to current line group
            line.append((x, y, w, h))
        else:
            # Sort characters in the same line by x position
            line_sorted = sorted(line, key=lambda b: b[0])
            sorted_boxes_adjusted.extend(line_sorted)  # Add sorted line to final list
            line = [(x, y, w, h)]  # Start a new line

        last_y = y  # Update the last y position

    # After the loop, sort and add the last line of characters
    if line:
        line_sorted = sorted(line, key=lambda b: b[0])
        sorted_boxes_adjusted.extend(line_sorted)

    return sorted_boxes_adjusted

def segment_image(image_path, target_size=28, min_width=5, min_height=5, y_tolerance=5):
    # Load the input image and preprocess
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 100)

    # Find contours and get bounding boxes
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    widths = []
    heights = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_width and h >= min_height:
            widths.append(w)
            heights.append(h)
    # Calculate the average width of the bounding boxes
    avg_width = np.mean(widths) if widths else 0
    avg_height = np.mean(heights) if heights else 0

    boxes = []
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= (avg_width - 25) and w >= min_width and h >= (avg_height-25) and h >= min_height:
            # Append bounding box and process the ROI
            boxes.append((x, y, w, h))
            roi = gray[y:y + h, x:x + w]
            (tH, tW) = roi.shape
            scale = target_size / max(tW, tH)
            newW, newH = int(tW * scale), int(tH * scale)
            resized_roi = cv2.resize(roi, (newW, newH), interpolation=cv2.INTER_AREA)
            padW, padH = (target_size - newW) // 2, (target_size - newH) // 2
            padded = cv2.copyMakeBorder(
                resized_roi,
                top=padH, bottom=target_size - newH - padH,
                left=padW, right=target_size - newW - padW,
                borderType=cv2.BORDER_CONSTANT, value=255
            )
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)
            chars.append((padded, (x, y, w, h)))

    # Sort the bounding boxes with tolerance for small y differences
    sorted_boxes = sort_bounding_boxes_with_tolerance(boxes, y_tolerance)

    # Reorder the character list based on sorted bounding boxes
    sorted_chars = [char for box in sorted_boxes for char in chars if char[1] == box]

    return sorted_chars, image

def ocr_predict_nominal(img_path, model, list_label):
    """
    Perform OCR prediction on a cropped image and return the recognized text.

    Args:
    - img_path (str): Path to the cropped image.
    - model: The trained character recognition model.
    - list_label (list): The list of possible characters (labels) used for OCR.

    Returns:
    - recognized_text (str): The final recognized text from the image.
    """
    # Segment the image to extract characters and boxes
    chars, image = segment_image(img_path)

    # Check for detected characters
    if len(chars) == 0:
        print("[ERROR] No characters detected.")
        return ""

    # Extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using the model
    preds = model.predict(chars)
    recognized_text = ""

    # Loop over predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # Find the top 3 predictions with the largest probabilities
        top_3_indices = np.argsort(pred)[-3:][::-1]

        # Get the top 3 predicted labels and their probabilities
        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        # Choose the most likely character that is numeric, otherwise fallback to the next label
        character = label1 if label1.isnumeric() else (label2 if label2.isnumeric() else label3)

        # Add the recognized character to the final text
        recognized_text += character

        # Optionally, draw the bounding box and the predicted text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Remove any non-numeric characters (if needed)
    cleaned_text = ''.join([c for c in recognized_text if c.isdigit()])
    if cleaned_text:
        # Separate the last two digits and format the rest
        integer_part = cleaned_text[:-2]  # Everything except the last two digits
        decimal_part = cleaned_text[-2:]  # The last two digits

        # Format the integer part with commas for thousands
        formatted_integer_part = "{:,}".format(int(integer_part)).replace(",", ".")

        # Combine integer and decimal parts with formatting
        formatted_nominal = f"{formatted_integer_part},{decimal_part}"
        formatted_nominal = "Rp " + formatted_nominal

        return formatted_nominal
    else:
        return "Invalid nominal"

def ocr_predict_type(img_path, model, list_label):
    """
    Perform OCR prediction on a cropped image to recognize the transaction type (DB/CR).

    Args:
    - img_path (str): Path to the cropped image.
    - model: The trained character recognition model.
    - list_label (list): The list of possible characters (labels) used for OCR.

    Returns:
    - recognized_text (str): The recognized transaction type (DB or CR).
    """
    # Segment the image to extract characters and boxes
    chars, image = segment_image(img_path)

    # Check for detected characters
    if len(chars) == 0:
        print("[ERROR] No characters detected.")
        return ""

    # Extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using the model
    preds = model.predict(chars)
    recognized_text = ""

    # Loop over predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # Find the top 3 predictions with the largest probabilities
        top_3_indices = np.argsort(pred)[-3:][::-1]

        # Get the top 3 predicted labels and their probabilities
        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        # Add the most likely character to the recognized text
        recognized_text += label1

        # Optionally, draw the bounding box and the predicted text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label1, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Determine the transaction type based on the recognized characters
    if 'D' in recognized_text or 'B' in recognized_text:
        recognized_text = 'DB'
    elif 'C' in recognized_text or 'R' in recognized_text:
        recognized_text = 'CR'

    return recognized_text

def ocr_predict_date(img_path, model, list_label):
    """
    Perform OCR prediction on a cropped image to recognize a date.

    Args:
    - img_path (str): Path to the cropped image.
    - model: The trained character recognition model.
    - list_label (list): The list of possible characters (labels) used for OCR.

    Returns:
    - recognized_text (str): The recognized date formatted as 'DD-MM-YYYY'.
    """
    # Segment the image to extract characters and boxes
    chars, image = segment_image(img_path)

    # Check for detected characters
    if len(chars) == 0:
        print("[ERROR] No characters detected.")
        return ""

    # Extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using the model
    preds = model.predict(chars)
    recognized_text = ""

    # Loop over predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # Find the top 3 predictions with the largest probabilities
        top_3_indices = np.argsort(pred)[-3:][::-1]

        # Get the top 3 predicted labels and their probabilities
        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        # Add the most likely character to the recognized text
        character = label1
        if not label1.isnumeric():
            character = label2
            if not label2.isnumeric():
                character = label3
        recognized_text += character

        # Optionally, draw the bounding box and the predicted text on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Format the recognized text as 'DD-MM-YYYY' (insert dashes)
    if len(recognized_text) >= 8:
        recognized_text = recognized_text[:2] + "-" + recognized_text[3:5] + "-" + recognized_text[6:]

    return recognized_text

def convert_pdf_to_png(app, file):
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        # Save the uploaded PDF file
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(pdf_path)

        pages = convert_from_path(pdf_path, 150)  # 150 dpi for decent quality

        # Specify the output directory in /content
        output_dir = './content/pdf_2_png'
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for i, page in enumerate(pages):
            output_path = os.path.join(output_dir, f'page_{i + 1}.png')
            page.save(output_path, 'PNG')

def detect_png():
    input_dir = './content/pdf_2_png/'
    output_dir = './content/yolov5/runs/detect'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    page_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    print("Found pages:", page_files)  # Debugging: Check the detected page files

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for page in page_files:
        page_path = os.path.join(input_dir, page)
        print(f"Processing: {page_path}")

        # Define output subdirectory
        output_subdir = os.path.join(output_dir, "exp")
        os.makedirs(output_subdir, exist_ok=True)

        # Run detection command
        command = f"python3 ./content/yolov5/detect.py --weights ./content/Best-Object-Detection-Weight.pt --img 640 --conf 0.4 --source {page_path} --save-txt --project {output_subdir}"
        print(f"Running detection for {page_path}...")

        # Run command and check the return code
        return_code = subprocess.run(command, shell=True, capture_output=True, text=True)

        if return_code.returncode == 0:
            print(f"Detection completed successfully for {page_path}")
        else:
            print(f"Error occurred during detection for {page_path}")
            print(f"stderr: {return_code.stderr}")
            print(f"stdout: {return_code.stdout}")

def crop_image():
    image_folder = "./content/pdf_2_png/"  # Folder containing the images
    output_folder = "./content/cropped_images/"  # Folder to save cropped images
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Base folder containing 'exp', 'exp2', 'exp3', etc.
    label_base_folder = "./content/yolov5/runs/detect/exp/"

    # Get a list of all subdirectories (e.g., exp, exp2, exp3) in the label base folder
    label_subdirectories = [d for d in sorted(os.listdir(label_base_folder),
                                              key=lambda x: int(x.replace('exp', '')) if x != 'exp' else 0)
                            if os.path.isdir(os.path.join(label_base_folder, d))]

    print(f"Label subdirectories found: {label_subdirectories}")

    # Loop through each label subdirectory (exp, exp2, exp3, etc.)
    for label_folder in label_subdirectories:
        # Construct the full path to the current label folder
        current_label_folder = os.path.join(label_base_folder, label_folder, 'labels')
        print(f"Processing labels in: {current_label_folder}")

        # Loop through all label files in the current label folder
        for label_file in os.listdir(current_label_folder):
            if label_file.endswith(".txt"):
                # Extract the base name of the image (e.g., 'page_1' from 'page_1.txt')
                base_name = os.path.splitext(label_file)[0]
                image_path = os.path.join(image_folder, f"{base_name}.png")
                print(f"Looking for image: {image_path}")

                # Check if the corresponding image exists
                if not os.path.exists(image_path):
                    print(f"Image not found for label: {label_file}")
                    continue

                # Read the image
                image = cv2.imread(image_path)
                h, w, _ = image.shape  # Get image dimensions

                # Read the label file
                with open(os.path.join(current_label_folder, label_file), "r") as file:
                    labels = file.readlines()

                # Extract bounding boxes from the label file
                for i, label in enumerate(labels):
                    class_id, x_center, y_center, box_width, box_height = map(float, label.split())

                    # Convert normalized coordinates to pixel values
                    x1 = int((x_center - box_width / 2) * w)
                    y1 = int((y_center - box_height / 2) * h)
                    x2 = int((x_center + box_width / 2) * w)
                    y2 = int((y_center + box_height / 2) * h)

                    # Crop the image based on the bounding box
                    cropped_image = image[y1:y2, x1:x2]

                    # Generate the output file name in the format: page_1_class0_box_1.png
                    output_filename = f"{base_name}_class{int(class_id)}_box_{i + 1}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    # Save the cropped image
                    cv2.imwrite(output_path, cropped_image)

                    print(f"Saved cropped image: {output_path}")