import cv2
import pytesseract
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Use LSTM engine + Spanish + Historical optimization
custom_config = r'--oem 1 --psm 6 -l spa'

image_dir = "dataset_images"
results = []

print("Running Advanced Preprocessing Pipeline...")
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith((".jpg", ".png")):
        img_path = os.path.join(image_dir, filename)
        
        # 1. Load image
        img = cv2.imread(img_path)
        
        # 2. Denoising: Removes small spots and paper grain
        dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        
        # 3. Convert to Grayscale
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        
        # 4. Increase Contrast: Makes the ink darker
        alpha = 1.5 # Contrast control
        beta = 0    # Brightness control
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # 5. Thresholding: Turns the image into pure Black and White
        thresh = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 6. OCR
        text = pytesseract.image_to_string(thresh, config=custom_config)
        
        results.append({"file": filename, "transcription": text.strip()})

# Save
df = pd.DataFrame(results)
df.to_csv("modular_results_v2.csv", index=False)
print("\nAdvanced Pipeline Complete! Check modular_results_v2.csv.")