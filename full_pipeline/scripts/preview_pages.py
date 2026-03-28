from PIL import Image
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preview_source(pages_dir, split, max_pages=6):
    pages_dir = Path(pages_dir)
    sources = sorted(pages_dir.iterdir())
    
    for source in sources:
        if not source.is_dir():
            continue
        pages = sorted(source.glob("*.png"))[:max_pages]
        if not pages:
            continue
        
        fig, axes = plt.subplots(1, len(pages), figsize=(20, 8))
        if len(pages) == 1:
            axes = [axes]
        
        for ax, page in zip(axes, pages):
            img = Image.open(page)
            ax.imshow(img)
            ax.set_title(page.name, fontsize=7)
            ax.axis('off')
        
        fig.suptitle(f"{split} - {source.name}", fontsize=10)
        out_path = Path.home() / f"ocr_project/outputs/{split}_{source.name[:30]}_preview.png"
        plt.savefig(out_path, dpi=72, bbox_inches='tight')
        plt.close()
        print(f"Saved preview: {out_path}")

preview_source("~/ocr_project/data/pages/handwriting".replace("~", str(Path.home())), "handwriting")
preview_source("~/ocr_project/data/pages/print".replace("~", str(Path.home())), "print")
print("All previews done!")
