# renAIssance OCR Pipeline - GSoC 2026

**Project:** renAIssance - Optical Character Recognition for Early Modern Printed Sources  

## Overview
This repository contains a robust, dual-stage machine learning pipeline designed to extract and transcribe 17th-century Spanish text (e.g., the 1611 dictionary). It addresses the challenges of early modern typography, complex multi-column layouts, and page embellishments.

## Architecture Pipeline
1. **Layout Analysis & Column Segmentation (OpenCV):** Pre-processes the raw historical scans using morphological dilation and contour detection to automatically slice multi-column layouts, preventing cross-gutter hallucination.
2. **Stage 1: Vision-Encoder-Decoder (TrOCR):** Utilizes a Fine-Tuned Transformer architecture. The Self-Attention mechanism of the Vision Transformer (ViT) explicitly isolates high-entropy textual features while disregarding page embellishments, borders, and marginalia.
3. **Stage 2: Late-Stage LLM Refinement (Flan-T5):** Integrates a generative language model to perform contextual cleaning. This resolves visual ambiguities caused by faded ink and restores proper 1611 Spanish orthography.

## Quantitative Results
Evaluated against the project's transcribed ground truth dataset using the `jiwer` library, the integration of the Late-Stage LLM resulted in an **88.8% reduction in Word Error Rate (WER)**.

| Pipeline Stage | Character Error Rate (CER) | Word Error Rate (WER) |
| :--- | :--- | :--- |
| **Stage 1: Raw TrOCR** | 11.81% | 34.62% |
| **Stage 2: LLM Cleaned** | **0.79%** | **3.85%** |

## Setup & Execution

**Note:** The fine-tuned TrOCR model weights exceed GitHub's file limits. 
Download the model weights here: https://drive.google.com/file/d/1O1AlRN7i_2ARZ_EPngTW5CK7NvWGHWVt/view?usp=sharing
*Instructions: Unzip the `gsoc_transformer_model` folder into the root directory of this repository before running the scripts.*

**Run the Pipeline:**
* **Split Layouts:** `python layout_splitter.py`
* **Train Model:** `python finetune_trocr.py`
* **Run Inference & Clean:** `python inference_and_clean.py`
* **Evaluate Metrics:** `python evaluate_accuracy.py`
* **Visualize Attention:** `python visualize_attention.py`x
