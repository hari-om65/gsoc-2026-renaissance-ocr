# renAIssance OCR Pipeline - GSoC 2026

# 🏛️ HumanAI Foundation | Google Summer of Code 2026

This repository contains a robust, dual-stage machine learning pipeline designed to extract and transcribe 17th-century Spanish text (e.g., the 1611 dictionary). It addresses the severe challenges of early modern typography, complex multi-column layouts, and page embellishments. 

This pipeline was developed as a technical evaluation submission for **Google Summer of Code 2026** under the **HumanAI Foundation**.

---

## 📖 Overview & Architecture

Traditional OCR tools fail on aged 17th-century manuscripts due to faded ink, interchangeable characters (u/v, f/s), and complex marginalia. This pipeline solves these issues using a modern, context-aware approach:

1. **Layout Analysis & Column Segmentation (OpenCV):** Pre-processes raw historical scans using morphological dilation and contour detection to automatically slice multi-column layouts.
2. **Stage 1: Vision-Encoder-Decoder (TrOCR):** Utilizes a Fine-Tuned Vision Transformer (ViT) to isolate textual features while disregarding page noise.
3. **Stage 2: Late-Stage LLM Refinement (Flan-T5):** Integrates a generative language model to perform contextual cleaning and restore proper 1611 Spanish orthography.

---

## 📊 Quantitative Results

The integration of the Late-Stage LLM resulted in an **88.8% reduction in Word Error Rate (WER)**.

| Pipeline Stage | Character Error Rate (CER) | Word Error Rate (WER) |
| :--- | :--- | :--- |
| **Stage 1: Raw TrOCR** | 11.81% | 34.62% |
| **Stage 2: LLM Cleaned** | **0.79%** | **3.85%** |

---

## ⚙️ Installation & Setup

Ensure you have Python 3.9+ installed. Clone this repository and install the required dependencies:

```bash
git clone [https://github.com/hari-om65/gsoc-2026-renaissance-ocr.git](https://github.com/hari-om65/gsoc-2026-renaissance-ocr.git)
cd gsoc-2026-renaissance-ocr
pip install -r requirements.txt
```
### Downloading Model Weights
**Note:** The model weights exceed GitHub limits. Download them from the link below and extract the folder into your root directory:

**Download Link:** [Click here to download weights](https://drive.google.com/file/d/1O1AlRN7i_2ARZ_EPngTW5CK7NvWGHWVt/view?usp=sharing)
---

## 🚀 Pipeline Execution

Run the scripts in this order to replicate the results:

1. **Split Layouts:**
```bash
python layout_splitter.py
```
2. **Train Model (Optional):**
```bash
python finetune_trocr.py
```
3. **Run Inference & Clean:**
```bash
python inference_and_clean.py
```
4. **Evaluate Metrics:**
```bash
python evaluate_accuracy.py
```
5. **Visualize Attention:**
```bash
python visualize_attention.py
```
---

## 🗺️ Future Roadmap (GSoC 2026 Proposal)

While the current backend achieves highly accurate transcription (3.85% WER), the goal for the GSoC 2026 summer coding period is to upgrade this architecture into a fully deployable, user-friendly application for humanities researchers. 

Planned upgrades include:
* **CRAFT Integration:** Upgrading the OpenCV layout splitter to the CRAFT (Character Region Awareness for Text Detection) deep learning model for pixel-perfect line segmentation.
* **ONNX Quantization:** Compressing the TrOCR and Flan-T5 models into ONNX format for highly optimized, CPU-friendly inference.
* **Interactive UI & Dockerization:** Wrapping the pipeline in a containerized Streamlit web interface. This will allow researchers to visually adjust deskewing, padding, and noise thresholds in real-time before running the OCR.
