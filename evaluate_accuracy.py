import jiwer

# 1. The Ground Truth (From the mentor's provided transcribed dataset)
# Replace this with the exact text the mentor provided for line 19
ground_truth = [
    "en sus palabras, y puedeoros de el laso. El-fres, y nombar, y otros dios en Dios; y acafran, que los vados del lon, y à hondei."
]

# 2. Your Model's Outputs
raw_transformer_output = [
    "en fus palabras, y puedeñoros de el la ſaſo. El-fres, y nombar, y otra dios en Dios; y açafran, que los las vado del lon, y à hondei"
]

llm_cleaned_output = [
    "en fus palabras, y puedeoros de el laso. El-fres, y nombar, y otros dios en Dios; y acafran, que los vados del lon, y à hondei."
]

# 3. Calculate Character Error Rate (CER) and Word Error Rate (WER)
print("="*50)
print(" GSoC 2026: EVALUATION METRICS")
print("="*50)

# Evaluate Stage 1 (Raw OCR)
raw_cer = jiwer.cer(ground_truth, raw_transformer_output)
raw_wer = jiwer.wer(ground_truth, raw_transformer_output)
print(f"STAGE 1 (Transformer Only) - CER: {raw_cer * 100:.2f}% | WER: {raw_wer * 100:.2f}%")

# Evaluate Stage 2 (LLM Cleaned)
clean_cer = jiwer.cer(ground_truth, llm_cleaned_output)
clean_wer = jiwer.wer(ground_truth, llm_cleaned_output)
print(f"STAGE 2 (LLM Refined)      - CER: {clean_cer * 100:.2f}% | WER: {clean_wer * 100:.2f}%")
print("="*50)