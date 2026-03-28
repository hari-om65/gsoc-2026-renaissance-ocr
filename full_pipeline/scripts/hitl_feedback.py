import json
from pathlib import Path

def load_human_corrections(annotation_dir):
    """Load all human-corrected annotations."""
    annotation_dir = Path(annotation_dir)
    new_pairs = []
    stats = {"total": 0, "corrected": 0, "approved": 0, "skipped": 0}

    for ann_file in annotation_dir.glob("*.json"):
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        for ann in data["annotations"]:
            stats["total"] += 1
            ocr_pred = ann["ocr_prediction"]
            human_correction = ann["human_correction"]
            is_correct = ann["is_correct"]

            if is_correct is None:
                stats["skipped"] += 1
                continue

            if is_correct == True:
                # OCR was correct - use as positive training example
                if len(ocr_pred) > 3:
                    new_pairs.append({
                        "image_path": ann["line_image"],
                        "gt_text": ocr_pred,
                        "source": "hitl_approved",
                        "confidence": "high"
                    })
                    stats["approved"] += 1

            elif is_correct == False and human_correction != ocr_pred:
                # Human corrected it - use corrected version
                if len(human_correction) > 3:
                    new_pairs.append({
                        "image_path": ann["line_image"],
                        "gt_text": human_correction,
                        "ocr_text": ocr_pred,
                        "source": "hitl_corrected",
                        "confidence": "high"
                    })
                    stats["corrected"] += 1

    return new_pairs, stats

def merge_with_existing_pairs(new_pairs, existing_pairs_path):
    """Merge HITL pairs with existing training pairs."""
    with open(existing_pairs_path, encoding="utf-8") as f:
        existing = json.load(f)

    print(f"Existing pairs: {len(existing)}")
    print(f"New HITL pairs: {len(new_pairs)}")

    # Avoid duplicates
    existing_images = {p["image_path"] for p in existing}
    truly_new = [p for p in new_pairs
                 if p["image_path"] not in existing_images]

    merged = existing + truly_new
    print(f"Merged total: {len(merged)}")
    return merged

def simulate_human_review(annotation_dir):
    """
    Simulate a human reviewing and correcting annotations.
    In production: human opens JSON, edits corrections manually.
    Here we simulate by auto-approving high-confidence predictions.
    """
    annotation_dir = Path(annotation_dir)
    simulated = 0

    for ann_file in annotation_dir.glob("*.json"):
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        data["annotator"] = "simulated_human_reviewer"
        data["status"] = "reviewed"

        for ann in data["annotations"]:
            pred = ann["ocr_prediction"]
            if not pred or len(pred) < 3:
                ann["is_correct"] = False
                ann["human_correction"] = ""
                continue

            # Simulate: approve if prediction looks reasonable
            # (has mostly alphabetic chars, reasonable length)
            alpha_ratio = sum(c.isalpha() for c in pred) / max(len(pred), 1)
            if alpha_ratio > 0.6 and 5 < len(pred) < 100:
                ann["is_correct"] = True
                ann["confidence"] = "high"
            else:
                ann["is_correct"] = False
                ann["confidence"] = "low"
            simulated += 1

        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Simulated review of {simulated} annotations")

# ── Run the HITL feedback loop ───────────────────────────────
annotation_dir = Path.home() / "ocr_project/data/annotations/hitl"
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
merged_path = Path.home() / "ocr_project/data/annotations/line_pairs_with_hitl.json"

print("=== HITL FEEDBACK LOOP ===\n")

# Step 1: Simulate human review
print("Step 1: Human reviews annotation sheets...")
simulate_human_review(annotation_dir)

# Step 2: Load corrections
print("\nStep 2: Loading human corrections...")
new_pairs, stats = load_human_corrections(annotation_dir)
print(f"  Total annotations: {stats['total']}")
print(f"  Approved (correct): {stats['approved']}")
print(f"  Corrected by human: {stats['corrected']}")
print(f"  Skipped: {stats['skipped']}")
print(f"  New training pairs from HITL: {len(new_pairs)}")

# Step 3: Merge with existing
print("\nStep 3: Merging with existing training data...")
merged = merge_with_existing_pairs(new_pairs, pairs_path)

# Step 4: Save merged dataset
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)
print(f"Saved merged dataset to: {merged_path}")

# Step 5: Show improvement potential
print("\n=== HITL IMPACT SUMMARY ===")
with open(pairs_path) as f:
    original = json.load(f)
print(f"Original training pairs: {len(original)}")
print(f"After HITL feedback:     {len(merged)}")
print(f"Increase:                +{len(merged)-len(original)} pairs ({(len(merged)-len(original))/len(original)*100:.0f}%)")

print("\nHITL Workflow Complete!")
print("\nNext step: retrain TrOCR on merged dataset for improved CER")
print("This is the continuous improvement cycle:")
print("  Train → Deploy → Human Review → More Data → Retrain")
