import os
import shutil
import subprocess
import cv2
import imutils
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
from google.cloud import storage


def download_blob(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded storage object {source_blob_name} to local file {destination_file_name}.")


def sort_bounding_boxes_with_tolerance(boxes, y_tolerance=5):
    """
    Sort bounding boxes primarily by y-coordinate and secondarily by x-coordinate,
    allowing a small tolerance for misalignments in the vertical position (y).
    """
    sorted_boxes = sorted(boxes, key=lambda b: (b[1], b[0]))

    sorted_boxes_adjusted = []

    last_y = None
    line = []
    for (x, y, w, h) in sorted_boxes:
        if last_y is None or abs(last_y - y) <= y_tolerance:
            line.append((x, y, w, h))
        else:
            line_sorted = sorted(line, key=lambda b: b[0])
            sorted_boxes_adjusted.extend(line_sorted)
            line = [(x, y, w, h)]

        last_y = y

    if line:
        line_sorted = sorted(line, key=lambda b: b[0])
        sorted_boxes_adjusted.extend(line_sorted)

    return sorted_boxes_adjusted


def segment_image(image_path, target_size=28, min_width=5, min_height=5, y_tolerance=5):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 100)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    widths = []
    heights = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= min_width and h >= min_height:
            widths.append(w)
            heights.append(h)
    avg_width = np.mean(widths) if widths else 0
    avg_height = np.mean(heights) if heights else 0

    boxes = []
    chars = []

    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= (avg_width - 25) and w >= min_width and h >= (avg_height - 25) and h >= min_height:
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

    sorted_boxes = sort_bounding_boxes_with_tolerance(boxes, y_tolerance)

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

    res = process_image_and_predict(img_path, model)
    if not res['success']:
        return res
    preds = res['predictions']
    boxes = res['boxes']
    image = res['image']
    recognized_text = ""
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        top_3_indices = np.argsort(pred)[-3:][::-1]

        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        character = label1 if label1.isnumeric() else (label2 if label2.isnumeric() else label3)

        recognized_text += character

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cleaned_text = ''.join([c for c in recognized_text if c.isdigit()])

    if not cleaned_text or len(cleaned_text) < 3:
        return "Invalid nominal"

    integer_part = cleaned_text[:-2]

    if not integer_part:
        integer_part = '0'

    formatted_integer_part = "{:,}".format(int(integer_part)).replace(",", ".")

    formatted_nominal = f"{formatted_integer_part}"
    return formatted_nominal


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

    res = process_image_and_predict(img_path, model)
    if not res['success']:
        return res
    preds = res['predictions']
    boxes = res['boxes']
    image = res['image']
    recognized_text = ""

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        top_3_indices = np.argsort(pred)[-3:][::-1]

        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        recognized_text += label1

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label1, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

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

    res = process_image_and_predict(img_path, model)
    if not res['success']:
        return res
    preds = res['predictions']
    boxes = res['boxes']
    image = res['image']
    recognized_text = ""

    for (pred, (x, y, w, h)) in zip(preds, boxes):
        top_3_indices = np.argsort(pred)[-3:][::-1]

        prob1, label1 = pred[top_3_indices[0]], list_label[top_3_indices[0]]
        prob2, label2 = pred[top_3_indices[1]], list_label[top_3_indices[1]]
        prob3, label3 = pred[top_3_indices[2]], list_label[top_3_indices[2]]

        character = label1
        if not label1.isnumeric():
            character = label2
            if not label2.isnumeric():
                character = label3
        recognized_text += character

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, character, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    if len(recognized_text) >= 8:
        recognized_text = recognized_text[:2] + "-" + recognized_text[3:5] + "-" + recognized_text[6:]

    return recognized_text


def convert_pdf_to_png(pdf_path, output_dir):
    try:
        pages = convert_from_path(pdf_path, 150)
        for i, page in enumerate(pages):
            output_path = os.path.join(output_dir, f'page_{i + 1}.png')
            page.save(output_path, 'PNG')
    except Exception as e:
        print(f"Error: {e}")
        return None


def convert_jpg_to_png(image_path, output_dir, brightness=1.2, contrast=1.3, sharpness=1.5):
    try:
        img = Image.open(image_path)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness)

        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast)

        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(sharpness)
        output_path = os.path.join(output_dir, 'page_1.png')
        img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error: {e}")
        return None


def copy_file_png(image_path, output_dir):
    try:
        output_path = os.path.join(output_dir, 'page_1.png')
        shutil.copy(str(image_path), output_path)
    except Exception as e:
        print(f"Error: {e}")
        return None


def detect_png():
    input_dir = './content/png_image/'
    output_dir = './content/yolov5/runs/detect'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    page_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    print("Found pages:", page_files)

    os.makedirs(output_dir, exist_ok=True)

    for page in page_files:
        page_path = os.path.join(input_dir, page)
        print(f"Processing: {page_path}")

        output_subdir = os.path.join(output_dir, "exp")
        os.makedirs(output_subdir, exist_ok=True)

        command = f"python3 ./content/yolov5/detect.py --weights ./content/Best-Object-Detection-Weight.pt --img 640 --conf 0.4 --source {page_path} --save-txt --project {output_subdir}"
        print(f"Running detection for {page_path}...")

        return_code = subprocess.run(command, shell=True, capture_output=True, text=True)

        if return_code.returncode == 0:
            print(f"Detection completed successfully for {page_path}")
        else:
            print(f"Error occurred during detection for {page_path}")
            print(f"stderr: {return_code.stderr}")
            print(f"stdout: {return_code.stdout}")


def crop_image():
    image_folder = "./content/png_image/"
    output_folder = "./content/cropped_images/"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    label_base_folder = "./content/yolov5/runs/detect/exp/"

    label_subdirectories = [d for d in sorted(os.listdir(label_base_folder),
                                              key=lambda x: int(x.replace('exp', '')) if x != 'exp' else 0)
                            if os.path.isdir(os.path.join(label_base_folder, d))]

    print(f"Label subdirectories found: {label_subdirectories}")

    for label_folder in label_subdirectories:
        current_label_folder = os.path.join(label_base_folder, label_folder, 'labels')
        print(f"Processing labels in: {current_label_folder}")

        for label_file in os.listdir(current_label_folder):
            if label_file.endswith(".txt"):
                base_name = os.path.splitext(label_file)[0]
                image_path = os.path.join(image_folder, f"{base_name}.png")
                print(f"Looking for image: {image_path}")

                if not os.path.exists(image_path):
                    print(f"Image not found for label: {label_file}")
                    continue

                image = cv2.imread(image_path)
                h, w, _ = image.shape

                with open(os.path.join(current_label_folder, label_file), "r") as file:
                    labels = file.readlines()

                for i, label in enumerate(labels):
                    class_id, x_center, y_center, box_width, box_height = map(float, label.split())

                    x1 = int((x_center - box_width / 2) * w)
                    y1 = int((y_center - box_height / 2) * h)
                    x2 = int((x_center + box_width / 2) * w)
                    y2 = int((y_center + box_height / 2) * h)

                    cropped_image = image[y1:y2, x1:x2]

                    output_filename = f"{base_name}_class{int(class_id)}_box_{i + 1}.png"
                    output_path = os.path.join(output_folder, output_filename)

                    cv2.imwrite(output_path, cropped_image)

                    print(f"Saved cropped image: {output_path}")


def process_image_and_predict(img_path, model):
    try:
        chars, image = segment_image(img_path)

        if len(chars) == 0:
            print("[ERROR] No characters detected.")
            return {
                "success": False,
                "error": "No characters detected in the image.",
                "predictions": None,
                "boxes": None,
                "image": image,
            }

        boxes = [b[1] for b in chars]
        chars = np.array([c[0] for c in chars], dtype="float32")

        preds = model.predict(chars)

        return {
            "success": True,
            "error": None,
            "predictions": preds,
            "boxes": boxes,
            "image": image,
        }

    except Exception as e:
        print('error', e)
        return {
            "success": False,
            "error": str(e),
            "predictions": None,
            "boxes": None,
            "image": None,
        }
