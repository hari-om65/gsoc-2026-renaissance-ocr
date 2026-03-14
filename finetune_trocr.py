import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW 
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load Model & Processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed").to(device)

# --- THE GSoC CONFIGURATION FIX ---
# This stops the loops and fixes the internal library AttributeErrors
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.repetition_penalty = 3.5
model.config.no_repeat_ngram_size = 2
# ----------------------------------

class GSoCDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.df = pd.read_csv(csv_file, sep='|').head(50).dropna()
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = f"dataset_images/{self.df.iloc[idx, 0].strip()}"
        text = str(self.df.iloc[idx, 1]).strip()
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values[0]
        labels = self.processor.tokenizer(text, padding="max_length", max_length=128, truncation=True).input_ids
        labels = [l if l != processor.tokenizer.pad_token_id else -100 for l in labels]
        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}

# 2. Train
train_ds = GSoCDataset("metadata.csv", processor)
loader = DataLoader(train_ds, batch_size=2, shuffle=True)
optimizer = AdamW(model.parameters(), lr=4e-5)

model.train()
print("Starting Deep Learning Fine-tuning...")
for epoch in range(25): # Increased epochs for better recognition
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
        outputs = model(pixel_values=batch["pixel_values"].to(device), labels=batch["labels"].to(device))
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

model.save_pretrained("gsoc_transformer_model")
processor.save_pretrained("gsoc_transformer_model")
print("Model saved to gsoc_transformer_model/")