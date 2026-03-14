import cv2
import numpy as np
import os
import glob

def split_dictionary_columns(image_path, output_dir="dataset_images/crops"):
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 200)) 
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > 10000]
    valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[0])

    crop_paths = []
    base_name = os.path.basename(image_path)

    for i, contour in enumerate(valid_contours):
        x, y, w, h = cv2.boundingRect(contour)
        pad = 20
        y1, y2 = max(0, y-pad), min(img.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(img.shape[1], x+w+pad)
        
        crop = img[y1:y2, x1:x2]
        out_path = os.path.join(output_dir, f"column_{i+1}_{base_name}")
        cv2.imwrite(out_path, crop)
        crop_paths.append(out_path)

    return crop_paths

# --- BATCH EXECUTION ---
input_directory = "dataset_images/"
all_raw_images = glob.glob(os.path.join(input_directory, "raw*.jpg")) # Finds all your raw PDF pages

print(f"Found {len(all_raw_images)} raw images to split.")

for img_path in all_raw_images:
    print(f"Splitting: {os.path.basename(img_path)}")
    split_dictionary_columns(img_path)

print("SUCCESS: All pages successfully split into columns inside 'dataset_images/crops/'")