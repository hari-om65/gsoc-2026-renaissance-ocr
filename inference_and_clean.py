import torch
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    T5Tokenizer, 
    T5ForConditionalGeneration
)
from PIL import Image
import os
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Stage 1: Transformer OCR...")
processor = TrOCRProcessor.from_pretrained("./gsoc_transformer_model")
model = VisionEncoderDecoderModel.from_pretrained("./gsoc_transformer_model").to(device)

print("Loading Stage 2: LLM Cleaner (Flan-T5)...")
llm_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
llm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

def run_pipeline(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    
    generated_ids = model.generate(
        pixel_values, 
        max_new_tokens=64, 
        num_beams=4, 
        repetition_penalty=3.5
    )
    raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    prompt = f"Fix OCR typos in this 1611 Spanish text: {raw_text}"
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
    outputs = llm_model.generate(**inputs, max_new_tokens=100)
    clean_text = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return raw_text, clean_text

# --- BATCH EXECUTION ---
# Pointing to the newly cropped single-column images
crop_directory = "dataset_images/crops/"
all_crops = glob.glob(os.path.join(crop_directory, "*.jpg"))

print(f"\nFound {len(all_crops)} column crops to process. Starting AI pipeline...")

# Create a master text file to save all results
with open("final_transcriptions.txt", "w", encoding="utf-8") as file:
    for img_path in all_crops:
        print(f"Processing: {os.path.basename(img_path)}")
        try:
            raw, clean = run_pipeline(img_path)
            
            # Save to the text file
            file.write(f"--- {os.path.basename(img_path)} ---\n")
            file.write(f"RAW OCR : {raw}\n")
            file.write(f"CLEANED : {clean}\n\n")
            
        except Exception as e:
            print(f"Failed to process {os.path.basename(img_path)}. Error: {e}")

print("\nSUCCESS: Entire dataset processed! Check 'final_transcriptions.txt' for the results.")