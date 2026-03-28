import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.optim import AdamW
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None

# Load pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 3 and len(p["gt_text"]) < 200]
print(f"Total pairs: {len(pairs)}")

split = int(len(pairs) * 0.85)
train_pairs = pairs[:split]
val_pairs = pairs[split:]
print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")

print("Loading TrOCR...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = processor.tokenizer.sep_token_id

model.generation_config.max_length = 64
model.generation_config.no_repeat_ngram_size = 3
model.generation_config.length_penalty = 2.0
model.generation_config.num_beams = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

class OCRDataset(Dataset):
    def __init__(self, pairs, processor, max_length=64):
        self.pairs = pairs
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        try:
            img = Image.open(pair["image_path"]).convert("RGB")
        except:
            img = Image.new("RGB", (384, 64), color=255)

        pixel_values = self.processor(
            img, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            pair["gt_text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}

train_dataset = OCRDataset(train_pairs, processor)
val_dataset = OCRDataset(val_pairs, processor)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print("Freezing encoder for first phase...")
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

def evaluate(model, loader, processor, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            total_loss += outputs.loss.item()
            generated = model.generate(pixel_values)
            preds = processor.tokenizer.batch_decode(
                generated, skip_special_tokens=True)
            gt_texts = processor.tokenizer.batch_decode(
                labels.masked_fill(
                    labels == -100,
                    processor.tokenizer.pad_token_id),
                skip_special_tokens=True)
            all_preds.extend(preds)
            all_gts.extend(gt_texts)

    from jiwer import cer, wer
    avg_loss = total_loss / len(loader)
    avg_cer = cer(all_gts, all_preds)
    avg_wer = wer(all_gts, all_preds)
    return avg_loss, avg_cer, avg_wer

EPOCHS = 15
best_cer = float('inf')
out_dir = Path.home() / "ocr_project/models/trocr_finetuned"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nStarting training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    if epoch == 5:
        print("Unfreezing encoder for end-to-end training...")
        for param in model.encoder.parameters():
            param.requires_grad = True
        optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()
    total_loss = 0
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    if (epoch + 1) % 3 == 0 or epoch == 0:
        val_loss, val_cer, val_wer = evaluate(
            model, val_loader, processor, device)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {val_cer:.4f} | WER: {val_wer:.4f}")
        if val_cer < best_cer:
            best_cer = val_cer
            model.save_pretrained(str(out_dir))
            processor.save_pretrained(str(out_dir))
            print(f"  --> Best model saved (CER: {best_cer:.4f})")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f}")

print(f"\nTraining complete! Best CER: {best_cer:.4f}")
print(f"Model saved to: {out_dir}")
