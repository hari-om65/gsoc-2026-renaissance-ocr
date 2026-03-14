import torch
import cv2
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import matplotlib.pyplot as plt
import os

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load Model
model_path = "./gsoc_transformer_model"
print("Loading model for attention visualization...")
processor = TrOCRProcessor.from_pretrained(model_path)

# --- THE FIX ---
# Disable 'sdpa' and force 'eager' implementation so attention weights are returned
model = VisionEncoderDecoderModel.from_pretrained(
    model_path, 
    attn_implementation="eager" 
).to(device)

# Explicitly force the internal Vision Encoder to save its attention weights
model.config.output_attentions = True
model.encoder.config.output_attentions = True
# ---------------

def get_attention_map(image_path):
    # Prepare image
    raw_image = Image.open(image_path).convert("RGB")
    pixel_values = processor(raw_image, return_tensors="pt").pixel_values.to(device)

    # Forward pass without calculating gradients
    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values, 
            decoder_input_ids=torch.tensor([[model.config.decoder_start_token_id]]).to(device),
            output_attentions=True,
            return_dict=True
        )
    
    # Get attentions from the last layer of the Vision Encoder
    attentions = outputs.encoder_attentions[-1] 
    
    # Average across all attention heads
    att_map = attentions[0].mean(dim=0).cpu().numpy()
    
    # Calculate grid size (e.g., 576 patches = 24x24 grid)
    num_patches = att_map.shape[0] - 1 
    grid_size = int(np.sqrt(num_patches))
    
    # Extract the attention of the CLS token (index 0) to all other image patches
    cls_attention = att_map[0, 1:]
    
    # Reshape into a 2D grid
    att_grid = cls_attention.reshape(grid_size, grid_size)
    
    # Upscale back to the original image dimensions
    heatmap = cv2.resize(att_grid, (raw_image.width, raw_image.height))
    
    # Normalize to 0-1 for plotting
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return np.array(raw_image), heatmap

# 3. Execution and Plotting
img_path = "dataset_images/raw-2.pdf_page_19.jpg"

if os.path.exists(img_path):
    print(f"Generating heatmap for {img_path}...")
    original, heat = get_attention_map(img_path)

    plt.figure(figsize=(12, 6))
    
    # Plot 1: Original
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original)
    plt.axis('off')

    # Plot 2: Heatmap Overlay
    plt.subplot(1, 2, 2)
    plt.title("Transformer Attention (Focus Area)")
    plt.imshow(original)
    plt.imshow(heat, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    # Save Output
    output_file = "attention_proof.png"
    plt.savefig(output_file, bbox_inches='tight')
    print(f"SUCCESS: Heatmap saved as '{output_file}'.")
else:
    print(f"Error: Could not find image at {img_path}")