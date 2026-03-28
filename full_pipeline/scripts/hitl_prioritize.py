import json
import re
from pathlib import Path

def compute_confidence_score(ocr_text):
    if not ocr_text or len(ocr_text) < 2:
        return 0.0
    score = 1.0
    alpha_ratio = sum(c.isalpha() for c in ocr_text) / len(ocr_text)
    score *= alpha_ratio
    if len(ocr_text) < 5:
        score *= 0.5
    artifacts = ['[', ']', '|', '{', '}', '€', '©', '°', '§']
    artifact_count = sum(ocr_text.count(a) for a in artifacts)
    score *= max(0, 1 - artifact_count * 0.2)
    if re.search(r'[A-Z]{4,}', ocr_text):
        score *= 0.8
    if re.search(r'\d{3,}', ocr_text):
        score *= 0.7
    return round(score, 3)

annotation_dir = Path.home() / "ocr_project/data/annotations/hitl"
all_annotations = []

for ann_file in sorted(annotation_dir.glob("*.json")):
    with open(ann_file, encoding="utf-8") as f:
        data = json.load(f)
    for ann in data["annotations"]:
        score = compute_confidence_score(ann["ocr_prediction"])
        all_annotations.append({
            "file": ann_file.name,
            "line_image": ann["line_image"],
            "ocr_prediction": ann["ocr_prediction"],
            "confidence_score": score,
            "needs_review": score < 0.6
        })

all_annotations.sort(key=lambda x: x["confidence_score"])

print("=== HITL PRIORITIZATION REPORT ===")
print(f"Total annotations: {len(all_annotations)}")

needs_review = [a for a in all_annotations if a["needs_review"]]
high_conf = [a for a in all_annotations if not a["needs_review"]]

print(f"Needs human review (low confidence): {len(needs_review)}")
print(f"High confidence (auto-approve):      {len(high_conf)}")
print(f"Human time saved: {len(high_conf)/len(all_annotations)*100:.0f}%")

print("\n=== TOP 10 LINES NEEDING REVIEW ===")
for i, ann in enumerate(needs_review[:10], 1):
    print(f"{i:2d}. Score={ann['confidence_score']:.3f} | {ann['ocr_prediction'][:55]}")

print("\n=== TOP 10 HIGH CONFIDENCE LINES ===")
for i, ann in enumerate(high_conf[:10], 1):
    print(f"{i:2d}. Score={ann['confidence_score']:.3f} | {ann['ocr_prediction'][:55]}")

out_path = Path.home() / "ocr_project/outputs/hitl_prioritized.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "total": len(all_annotations),
        "needs_review": len(needs_review),
        "auto_approved": len(high_conf),
        "human_time_saved_pct": len(high_conf)/len(all_annotations)*100,
        "review_queue": needs_review,
        "approved_queue": high_conf
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {out_path}")
print("Phase 4 Complete!")
