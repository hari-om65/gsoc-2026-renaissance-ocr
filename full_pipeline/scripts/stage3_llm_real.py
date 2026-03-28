import os
import json
import time
from pathlib import Path
from groq import Groq
from jiwer import cer

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def llm_correct(text, max_retries=3):
    """Call Llama 3 via Groq to correct OCR text."""
    prompt = f"""You are an expert in 17th-century Spanish paleography. 
The following text was OCR'd from a historical Spanish document (printed source, circa 1600-1700).
Please correct any OCR errors while:
- Preserving archaic Spanish spellings (ss, vv, fs, etc.)
- Keeping Latin phrases intact
- NOT modernizing the language
- Only fixing clear OCR character errors

Return ONLY the corrected text, nothing else.

OCR text: {text}"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  Retry {attempt+1}: {e}")
            time.sleep(2)
    return text  # return original if all retries fail

# Load test pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 5]
split = int(len(pairs) * 0.85)
test_pairs = pairs[split:]

print(f"Testing real LLM correction on {len(test_pairs)} samples...")
print("Using: Llama 3.1-8b-instant via Groq API\n")

results = []
for i, pair in enumerate(test_pairs[:10]):
    raw = pair["ocr_text"]
    gt = pair["gt_text"]

    print(f"Sample {i+1}:")
    print(f"  GT:      {gt[:60]}")
    print(f"  Raw OCR: {raw[:60]}")

    corrected = llm_correct(raw)
    print(f"  LLM:     {corrected[:60]}")

    cer_before = cer(gt, raw)
    cer_after = cer(gt, corrected)
    print(f"  CER: {cer_before:.3f} → {cer_after:.3f}")
    print()

    results.append({
        "gt": gt,
        "raw_ocr": raw,
        "llm_corrected": corrected,
        "cer_before": cer_before,
        "cer_after": cer_after
    })
    time.sleep(0.5)  # rate limit safety

# Summary
avg_before = sum(r["cer_before"] for r in results) / len(results)
avg_after = sum(r["cer_after"] for r in results) / len(results)

print("=" * 60)
print(f"Avg CER before LLM: {avg_before:.4f} ({avg_before*100:.1f}%)")
print(f"Avg CER after LLM:  {avg_after:.4f} ({avg_after*100:.1f}%)")
print(f"Improvement:        {(avg_before-avg_after)*100:.1f}%")

# Save
out_path = Path.home() / "ocr_project/outputs/stage3_llm_real_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": "llama-3.1-8b-instant",
        "avg_cer_before": avg_before,
        "avg_cer_after": avg_after,
        "improvement": avg_before - avg_after,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to: {out_path}")
print("Stage 3 Real LLM Complete!")
