import gradio as gr
import json
import os
from pathlib import Path
from PIL import Image
import numpy as np

# Get all page images
pages_dir = Path.home() / "ocr_project/data/pages"
all_images = []
for split in ["print", "handwriting"]:
    split_dir = pages_dir / split
    if split_dir.exists():
        for source_dir in sorted(split_dir.iterdir()):
            if source_dir.is_dir():
                for img in sorted(source_dir.glob("*.png"))[:3]:
                    all_images.append(str(img))

print(f"Found {len(all_images)} images for annotation")

annotations = {}
current_idx = [0]

ann_file = Path.home() / "ocr_project/data/annotations/annotations.json"
ann_file.parent.mkdir(exist_ok=True)
if ann_file.exists():
    with open(ann_file) as f:
        annotations = json.load(f)

def get_image(idx):
    if 0 <= idx < len(all_images):
        return all_images[idx], f"Image {idx+1} of {len(all_images)}: {Path(all_images[idx]).name}"
    return None, "No image"

def save_annotation(idx, note):
    img_path = all_images[idx]
    annotations[img_path] = {"note": note, "annotated": True}
    with open(ann_file, "w") as f:
        json.dump(annotations, f, indent=2)
    return f"Saved annotation for image {idx+1}"

def next_image(idx, note):
    save_annotation(idx, note)
    new_idx = min(idx + 1, len(all_images) - 1)
    img, label = get_image(new_idx)
    return img, label, new_idx, ""

def prev_image(idx, note):
    save_annotation(idx, note)
    new_idx = max(idx - 1, 0)
    img, label = get_image(new_idx)
    return img, label, new_idx, ""

with gr.Blocks(title="OCR Annotation Tool") as demo:
    gr.Markdown("# OCR Line Annotation Tool")
    gr.Markdown("Review each page and add notes. Use this to understand document structure before Mask R-CNN training.")
    
    with gr.Row():
        with gr.Column(scale=3):
            img_display = gr.Image(label="Page Image", height=700)
        with gr.Column(scale=1):
            img_label = gr.Textbox(label="Current Image", interactive=False)
            idx_state = gr.State(0)
            notes = gr.Textbox(
                label="Annotation Notes",
                placeholder="Describe what you see: number of lines, quality, challenges...",
                lines=8
            )
            with gr.Row():
                prev_btn = gr.Button("Previous", variant="secondary")
                next_btn = gr.Button("Next", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("### Stats")
            stats = gr.Textbox(
                value=f"Total images: {len(all_images)}\nAnnotated: {len(annotations)}",
                label="Progress",
                interactive=False
            )

    demo.load(
        fn=lambda: (*get_image(0), 0, ""),
        outputs=[img_display, img_label, idx_state, notes]
    )
    
    next_btn.click(
        fn=next_image,
        inputs=[idx_state, notes],
        outputs=[img_display, img_label, idx_state, notes]
    )
    
    prev_btn.click(
        fn=prev_image,
        inputs=[idx_state, notes],
        outputs=[img_display, img_label, idx_state, notes]
    )

demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
