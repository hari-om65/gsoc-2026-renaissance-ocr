import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

PDF_ROOTS = {
    "handwriting": Path.home() / "ocr_project/data/pdfs_clean/handwriting",
    "print": Path.home() / "ocr_project/data/pdfs_clean/print",
}

OUT_ROOTS = {
    "handwriting": Path.home() / "ocr_project/data/pages/handwriting",
    "print": Path.home() / "ocr_project/data/pages/print",
}

DPI = 200

for split in ["handwriting", "print"]:
    pdf_root = PDF_ROOTS[split]
    out_root = OUT_ROOTS[split]
    out_root.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_root.glob("*.pdf"))
    print(f"\n=== Processing {split} PDFs ===")
    print(f"Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        safe_name = pdf_path.stem.replace("/", "_").replace(" ", "_")
        pdf_out_dir = out_root / safe_name
        pdf_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {pdf_path.name}")
        try:
            images = convert_from_path(str(pdf_path), dpi=DPI)
            for i, img in enumerate(images, start=1):
                img.save(pdf_out_dir / f"page_{i:04d}.png", "PNG")
            print(f"  Saved {len(images)} pages to {pdf_out_dir}")
        except Exception as e:
            print(f"  Failed on {pdf_path.name}: {e}")

print("\nDone.")
