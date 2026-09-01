"""
OCR Text Extraction
=======================

Thin wrapper around pytesseract (which itself wraps the Tesseract OCR
engine). Extracting this into its own module means the rest of the
pipeline (field_extraction.py, document_processor.py) doesn't need to
know or care which OCR engine is used underneath -- if this project
ever swapped Tesseract for a cloud OCR API, only this file would change.

IMPORTANT (Windows setup): pytesseract is a Python wrapper -- it does
NOT include the actual OCR engine. You must separately install the
Tesseract binary:
    https://github.com/UB-Mannheim/tesseract/wiki
After installing, set TESSERACT_CMD below to the install path if it's
not automatically found (commonly
"C:\\Program Files\\Tesseract-OCR\\tesseract.exe").
"""

import pytesseract
from PIL import Image

# Uncomment and edit this line on Windows if pytesseract can't find
# Tesseract automatically:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_text(image: Image.Image) -> str:
    """Runs OCR on a preprocessed PIL Image and returns the raw extracted text."""
    return pytesseract.image_to_string(image)


if __name__ == "__main__":
    import tempfile
    import os
    from preprocessing import load_and_preprocess
    from sample_generator import generate_appointment_slip

    sample_path = generate_appointment_slip(os.path.join(tempfile.gettempdir(), "sample_slip.png"))
    processed = load_and_preprocess(sample_path)
    text = extract_text(processed)
    print("Extracted text:\n")
    print(text)
