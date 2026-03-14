import fitz  # Full form: PyMuPDF
import os
import glob

# --- SETTINGS ---
RAW_FOLDER = "raw_images"
DPI = 300 

# This version uses exist_ok=True to prevent the "FileExistsError"
os.makedirs(RAW_FOLDER, exist_ok=True)

def extract_all_pdfs():
    # Finds all Portable Document Format files in your main root folder
    pdf_files = glob.glob("*.pdf")
    
    if not pdf_files:
        print("No Portable Document Format files found!")
        return

    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f"Processing: {filename}...")
        
        try:
            # Open the document
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Loop through pages while the document is OPEN
            for i in range(total_pages):
                page = doc.load_page(i)
                pix = page.get_pixmap(dpi=DPI)
                
                output_name = f"{RAW_FOLDER}/{filename}_page_{i+1}.jpg"
                pix.save(output_name)
            
            # Close it ONLY after the loop is done
            doc.close()
            print(f"Successfully extracted {total_pages} pages from {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    extract_all_pdfs()