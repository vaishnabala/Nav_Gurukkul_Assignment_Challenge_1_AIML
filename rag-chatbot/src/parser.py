import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Windows users: Ensure this path matches your Tesseract installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_page(page):
    """Converts a PDF page into an image and runs OCR."""
    try:
        pix = page.get_pixmap(dpi=100)
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        return f"[OCR Error: {e}]"

def parse_pdf(pdf_path, max_pages=None):
    """Extracts text from a PDF page by page with an optional page limit."""
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    extracted_pages = []
    
    total_pages = min(len(doc), max_pages) if max_pages else len(doc)
    print(f"Parsing {filename} (Processing {total_pages} pages)...")

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        if not text.strip():
            text = ocr_page(page)

        extracted_pages.append({
            "text": text.strip(),
            "metadata": {
                "source": filename,
                "page": page_num + 1
            }
        })
    return extracted_pages