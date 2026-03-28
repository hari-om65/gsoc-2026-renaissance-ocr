import os
import json
import time
import re
from pathlib import Path
from groq import Groq
from jiwer import cer
from spellchecker import SpellChecker

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

spell = SpellChecker(language="es")
dict_path = Path.home() / "ocr_project/data/custom_dict.txt"
with open(dict_path, encoding="utf-8") as f:
    custom_words = [w.strip() for w in f.readlines()]
spell.word_frequency.load_words(custom_words)

def count_unknown_words(text):
    words = re.findall(r"[a-záéíóúñüçA-ZÁÉÍÓÚÑÜÇ]{3,}", text)
    if not words:
        return 0, 0
    unknown = spell.unknown([w.lower() for w in words])
    return len(unknown), len(words)

def should_invoke_llm(text, threshold=0.4):
    """Only invoke LLM if more than 40% words are unknown."""
    unknown_count, total = count_unknown_words(text)
    if total == 0:
        return False
    ratio = unknown_count / total
    return ratio > threshold

def llm_correct_smart(text):
    """Smarter LLM prompt focused only on OCR artifact removal."""
    prompt = f"""You are correcting OCR errors in 17th-century Spanish text.
Rules:
1. Only fix clear OCR character errors (f→s, rn→m, u→n, etc.)
2. Do NOT change word meanings or modernize language
3. Do NOT add or remove words
4. Keep all Latin as-is
5. Return ONLY the corrected text

Examples of OCR errors to fix:
- "perfona" → "persona" (r misread)
- "defu" → "de su" (space lost)
- "interef" → "interes" (f→s)
- "mútrte" → "muerte" (character swap)

Text to correct: {text}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  LLM error: {e}")
        return text

# Load test pairs
pairs_path = Path.home() / "ocr_project/data/annotations/line_pairs.json"
with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)

pairs = [p for p in pairs if len(p["gt_text"]) > 5]
split = int(len(pairs) * 0.85)
test_pairs = pairs[split:]

print("=== SMART LLM FALLBACK ===")
print(f"Only invoking LLM when >40% words unknown\n")

results = []
llm_invoked = 0

for i, pair in enumerate(test_pairs[:15]):
    raw = pair["ocr_text"]
    gt = pair["gt_text"]

    unknown_count, total = count_unknown_words(raw)
    needs_llm = should_invoke_llm(raw)

    if needs_llm:
        llm_invoked += 1
        corrected = llm_correct_smart(raw)
        stage = "LLM"
        time.sleep(0.3)
    else:
        corrected = raw
        stage = "SKIP"

    cer_before = cer(gt, raw)
    cer_after = cer(gt, corrected)

    print(f"[{stage}] Unknown: {unknown_count}/{total} | CER: {cer_before:.3f}→{cer_after:.3f}")
    print(f"  GT:  {gt[:55]}")
    print(f"  RAW: {raw[:55]}")
    if needs_llm:
        print(f"  LLM: {corrected[:55]}")
    print()

    results.append({
        "gt": gt,
        "raw": raw,
        "corrected": corrected,
        "llm_invoked": needs_llm,
        "cer_before": cer_before,
        "cer_after": cer_after
    })

avg_before = sum(r["cer_before"] for r in results) / len(results)
avg_after = sum(r["cer_after"] for r in results) / len(results)

print("=" * 60)
print(f"LLM invoked: {llm_invoked}/{len(results)} samples")
print(f"Avg CER before: {avg_before*100:.1f}%")
print(f"Avg CER after:  {avg_after*100:.1f}%")
print(f"Improvement:    {(avg_before-avg_after)*100:.1f}%")

out_path = Path.home() / "ocr_project/outputs/stage3_smart_llm_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "model": "llama-3.1-8b-instant",
        "strategy": "smart_fallback_40pct_threshold",
        "llm_invocation_rate": llm_invoked/len(results),
        "avg_cer_before": avg_before,
        "avg_cer_after": avg_after,
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\nSaved to: {out_path}")
print("Smart LLM Stage Complete!")
