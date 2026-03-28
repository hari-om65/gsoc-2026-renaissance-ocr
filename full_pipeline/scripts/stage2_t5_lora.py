import json
import torch
import random
import numpy as np
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from jiwer import cer

# ── Synthetic noise generator ────────────────────────────────
def inject_noise(text, error_rate=0.15):
    """Inject realistic OCR-like errors into clean text."""
    # Common OCR confusion pairs for historical Spanish
    substitutions = {
        's': ['f', 'ſ', '5'],
        'f': ['s', 'ſ'],
        'u': ['n', 'v'],
        'n': ['u', 'm', 'ri'],
        'v': ['u', 'v'],
        'i': ['l', '1', 'j'],
        'l': ['i', '1'],
        'e': ['c', 'o'],
        'c': ['e', 'o'],
        'o': ['c', 'e', '0'],
        'a': ['o', 'u'],
        'm': ['rn', 'ni'],
        'rn': ['m'],
        'd': ['cl'],
        'cl': ['d'],
        'r': ['t', 'n'],
        'h': ['b', 'li'],
        'p': ['q', 'b'],
    }

    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        ch = chars[i]
        if random.random() < error_rate:
            action = random.choice(['substitute', 'delete', 'insert', 'merge'])

            if action == 'substitute' and ch.lower() in substitutions:
                replacement = random.choice(substitutions[ch.lower()])
                result.append(replacement)
            elif action == 'delete' and len(ch.strip()) > 0:
                pass  # skip character
            elif action == 'insert':
                result.append(ch)
                result.append(random.choice(['i', 'e', 'a', ' ']))
            elif action == 'merge' and ch == ' ':
                pass  # remove space to merge words
            else:
                result.append(ch)
        else:
            result.append(ch)
        i += 1

    return ''.join(result)

# ── Build training data ──────────────────────────────────────
print("Building T5 training data...")

pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 5]

# Real OCR pairs
train_data = []
for p in pairs:
    train_data.append({
        "input": f"correct: {p['ocr_text']}",
        "target": p["gt_text"]
    })

# Synthetic pairs (augment with noise injection)
print(f"Real pairs: {len(train_data)}")
for p in pairs:
    for _ in range(4):  # 4 synthetic per real
        noisy = inject_noise(p["gt_text"], error_rate=0.12)
        train_data.append({
            "input": f"correct: {noisy}",
            "target": p["gt_text"]
        })

print(f"Total pairs after augmentation: {len(train_data)}")

# Split
random.shuffle(train_data)
split = int(len(train_data) * 0.85)
train_split = train_data[:split]
val_split = train_data[split:]
print(f"Train: {len(train_split)}, Val: {len(val_split)}")

# ── Load T5 + LoRA ───────────────────────────────────────────
print("\nLoading T5 with LoRA...")
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
base_model = T5ForConditionalGeneration.from_pretrained(model_name)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ── Dataset ──────────────────────────────────────────────────
class CorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        inputs = self.tokenizer(
            item["input"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        targets = self.tokenizer(
            item["target"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels
        }

train_dataset = CorrectionDataset(train_split, tokenizer)
val_dataset = CorrectionDataset(val_split, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

# ── Training ─────────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=3e-4)
EPOCHS = 10
best_val_loss = float('inf')
out_dir = Path.home() / "ocr_project/models/t5_lora"
out_dir.mkdir(parents=True, exist_ok=True)

print(f"\nTraining T5+LoRA for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        print(f"  --> Best model saved")

print(f"\nT5+LoRA training complete!")

# ── Evaluate correction improvement ─────────────────────────
print("\n=== Evaluating correction improvement ===")
model.eval()

def correct_with_t5(text, model, tokenizer, device):
    input_text = f"correct: {text}"
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=128, truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=128,
            num_beams=4, early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

real_pairs = [p for p in pairs]
split2 = int(len(real_pairs) * 0.85)
test_real = real_pairs[split2:]

cer_before_list = []
cer_after_list = []

print("\nSample corrections:")
for p in test_real[:10]:
    raw = p["ocr_text"]
    gt = p["gt_text"]
    corrected = correct_with_t5(raw, model, tokenizer, device)

    cb = cer(gt, raw)
    ca = cer(gt, corrected)
    cer_before_list.append(cb)
    cer_after_list.append(ca)

    print(f"GT:        {gt[:60]}")
    print(f"Raw OCR:   {raw[:60]}")
    print(f"Corrected: {corrected[:60]}")
    print(f"CER: {cb:.3f} → {ca:.3f}")
    print()

avg_before = np.mean(cer_before_list)
avg_after = np.mean(cer_after_list)
print(f"Avg CER before T5: {avg_before:.4f} ({avg_before*100:.1f}%)")
print(f"Avg CER after T5:  {avg_after:.4f} ({avg_after*100:.1f}%)")
print(f"Improvement: {(avg_before-avg_after)*100:.1f}%")

print("\nStage 2 (T5+LoRA) Complete!")
