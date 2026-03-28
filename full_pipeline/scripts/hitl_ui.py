import streamlit as st
import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

st.set_page_config(layout="wide", page_title="OCR HITL Review Tool")

@st.cache_resource
def load_model():
    model_dir = Path.home() / "ocr_project/models/trocr_finetuned"
    processor = TrOCRProcessor.from_pretrained(str(model_dir))
    model = VisionEncoderDecoderModel.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return processor, model, device

def run_ocr(image_path, processor, model, device):
    img = Image.open(str(image_path)).convert("RGB")
    pixel_values = processor(img, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated = model.generate(pixel_values)
    return processor.tokenizer.decode(generated[0], skip_special_tokens=True)

def get_word_confidence(text, custom_words):
    from spellchecker import SpellChecker
    spell = SpellChecker(language="es")
    spell.word_frequency.load_words(custom_words)
    words = text.split()
    result = []
    for word in words:
        import re
        clean = re.sub(r"[^a-zA-ZáéíóúñüçÁÉÍÓÚÑÜÇ]", "", word)
        if len(clean) < 2:
            result.append(("gray", word))
        elif spell.known([clean.lower()]):
            result.append(("green", word))
        else:
            result.append(("red", word))
    return result

# Load custom words
dict_path = Path.home() / "ocr_project/data/custom_dict.txt"
custom_words = []
if dict_path.exists():
    with open(dict_path) as f:
        custom_words = [w.strip() for w in f.readlines()]

# Load annotation files
annotation_dir = Path.home() / "ocr_project/data/annotations/hitl"
ann_files = sorted(annotation_dir.glob("*.json"))

st.title("OCR Human-in-the-Loop Review Tool")
st.markdown("**17th-century Spanish Document OCR Correction**")

if not ann_files:
    st.error("No annotation files found!")
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
selected_file = st.sidebar.selectbox(
    "Select Page",
    [f.name for f in ann_files]
)

# Load selected annotation
ann_path = annotation_dir / selected_file
with open(ann_path, encoding="utf-8") as f:
    data = json.load(f)

annotations = data["annotations"]
total = len(annotations)
approved = sum(1 for a in annotations if a.get("is_correct") == True)
corrected = sum(1 for a in annotations if a.get("is_correct") == False and a.get("human_correction") != a.get("ocr_prediction"))

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Progress:** {approved+corrected}/{total}")
st.sidebar.progress((approved+corrected)/max(total,1))
st.sidebar.markdown(f"✅ Approved: {approved}")
st.sidebar.markdown(f"✏️ Corrected: {corrected}")
st.sidebar.markdown(f"⏳ Pending: {total-approved-corrected}")

# Main area
st.markdown("---")
line_idx = st.slider("Select Line", 0, total-1, 0)
ann = annotations[line_idx]

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Line Image")
    img_path = Path(ann["line_image"])
    if img_path.exists():
        img = Image.open(str(img_path))
        st.image(img, use_column_width=True)
    else:
        st.warning("Image not found")

with col2:
    st.subheader("OCR Prediction")
    ocr_text = ann["ocr_prediction"]

    # Color coded display
    word_conf = get_word_confidence(ocr_text, custom_words)
    colored_html = ""
    for color, word in word_conf:
        if color == "green":
            colored_html += f'<span style="background-color:#90EE90;padding:2px;margin:1px;border-radius:3px">{word}</span> '
        elif color == "red":
            colored_html += f'<span style="background-color:#FFB6C1;padding:2px;margin:1px;border-radius:3px">{word}</span> '
        else:
            colored_html += f'{word} '

    st.markdown(colored_html, unsafe_allow_html=True)
    st.markdown("🟢 Known word  🔴 Unknown/flagged word")
    st.markdown("---")

    # Editable correction field
    st.subheader("Human Correction")
    correction = st.text_area(
        "Edit if needed:",
        value=ann.get("human_correction", ocr_text),
        height=80
    )

    # Action buttons
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("✅ Accept", type="primary"):
            annotations[line_idx]["is_correct"] = True
            annotations[line_idx]["human_correction"] = ocr_text
            annotations[line_idx]["confidence"] = "high"
            data["annotations"] = annotations
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            st.success("Accepted!")

    with col_b:
        if st.button("✏️ Save Correction"):
            annotations[line_idx]["is_correct"] = False
            annotations[line_idx]["human_correction"] = correction
            annotations[line_idx]["confidence"] = "high"
            data["annotations"] = annotations
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            st.success("Correction saved!")

    with col_c:
        if st.button("⏭️ Skip"):
            st.info("Skipped")

# Export button
st.markdown("---")
if st.button("📥 Export Corrected Pairs as JSON"):
    export_pairs = []
    for ann in annotations:
        if ann.get("is_correct") == True:
            export_pairs.append({
                "image": ann["line_image"],
                "text": ann["ocr_prediction"],
                "source": "hitl_approved"
            })
        elif ann.get("human_correction") and ann["human_correction"] != ann["ocr_prediction"]:
            export_pairs.append({
                "image": ann["line_image"],
                "text": ann["human_correction"],
                "source": "hitl_corrected"
            })
    st.download_button(
        "Download JSON",
        data=json.dumps(export_pairs, indent=2, ensure_ascii=False),
        file_name="hitl_export.json",
        mime="application/json"
    )
    st.success(f"Ready to export {len(export_pairs)} pairs!")
