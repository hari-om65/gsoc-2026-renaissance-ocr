import os
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from tqdm import tqdm

# Load a base model to help us start
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

image_dir = "dataset_images"
data = []

print("Generating draft labels for your crops...")
for filename in tqdm(os.listdir(image_dir)):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        
        # Generate text guess
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        data.append({"file_name": filename, "text": generated_text})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("metadata.csv", index=False)
print("\nDone! Please open 'metadata.csv' and check if the text matches your images.")