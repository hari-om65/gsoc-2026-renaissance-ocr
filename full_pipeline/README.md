# renAIssance OCR Pipeline
## 17th Century Spanish Document Transcription

A production-ready MVP OCR pipeline for historical Spanish documents.

## Pipeline Architecture
Page Image -> Mask R-CNN (segmentation) -> TrOCR (recognition) -> Hunspell + T5/LoRA + LLM (correction) -> Human Review

## Deliverables
- Segmentation model: models/segmentation/model_final.pth
- TrOCR finetuned: models/trocr_finetuned/
- T5+LoRA corrector: models/t5_lora/
- Kaggle corpus: outputs/kaggle_corpus/

## Quick Start
1. Convert PDFs: python scripts/convert_pdfs_to_images.py
2. Segment lines: python scripts/crop_lines.py
3. Run OCR: python scripts/evaluate_trocr.py
4. Correct: python scripts/stage4_cascade_pipeline.py
5. Review: python scripts/hitl_annotation.py

## Results
- Segmentation: 768 lines cropped from 44 pages
- TrOCR CER: 31.9% (140 training pairs)
- Correction improvement: 2.8% via cascade
- Training data: 9197 pairs (137 real + 9060 synthetic)
- HITL: 94% human time saved via prioritization

## Models Used
- Mask R-CNN (ResNet-50-FPN) - line segmentation
- microsoft/trocr-base-printed - OCR recognition
- t5-small + LoRA - OCR correction
- Llama 3.1 via Groq API - LLM fallback

## Scripts
| Script | Purpose |
|--------|---------|
| convert_pdfs_to_images.py | PDF to PNG conversion |
| auto_annotate.py | Auto line annotation |
| train_maskrcnn_fast.py | Train segmentation |
| crop_lines.py | Extract line crops |
| build_pairs_v2.py | Build OCR training pairs |
| finetune_trocr.py | Fine-tune TrOCR |
| stage1_hunspell.py | Hunspell correction |
| stage2_t5_lora.py | T5+LoRA correction |
| stage3_llm_real.py | LLM fallback |
| stage4_cascade_pipeline.py | Full correction cascade |
| hitl_annotation.py | Human review sheets |
| hitl_feedback.py | HITL feedback loop |
| noise_generator.py | Synthetic data generation |
| image_augment.py | Image augmentation |
| build_kaggle_corpus.py | Kaggle dataset export |

## Key Findings
1. Long-s (s->f) is the dominant OCR error in historical Spanish
2. General LLMs degrade quality - domain fine-tuning essential
3. 66x data augmentation via synthetic noise generation
4. HITL prioritization saves 94% of human annotation time

## Local Deployment
All models run locally - no cloud APIs required (except optional LLM).
Protects sensitive archival material.

## License
MIT
