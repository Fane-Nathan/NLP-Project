"""OCR Engine for Tesseract"""
import pytesseract
from PIL import Image
import os

class OCREngine:
    """Wrapper for Tesseract OCR (Plan B)."""
    
    def __init__(self, tesseract_cmd: str = None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\felix\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
        
    def extract_text(self, image: Image.Image) -> str:
        """Extract text from an image using OCR."""
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            return "ERROR: Tesseract OCR not found."
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""