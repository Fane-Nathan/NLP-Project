import pytesseract
from PIL import Image
import os
import sys

class OCREngine:
    """
    Wrapper for Tesseract OCR (Plan B).
    """
    def __init__(self, tesseract_cmd: str = None):
        # Attempt to find tesseract if not provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Common default paths on Windows
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
        """
        Extract text from an image using OCR.
        """
import pytesseract
from PIL import Image
import os
import sys

class OCREngine:
    """
    Wrapper for Tesseract OCR (Plan B).
    """
    def __init__(self, tesseract_cmd: str = None):
        # Attempt to find tesseract if not provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        else:
            # Common default paths on Windows
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
        """
        Extract text from an image using OCR.
        """
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            return "ERROR: Tesseract OCR not found. Please install Tesseract or check the path."
        except Exception as e:
            print(f"OCR Error: {e}")
            print("TIP: Make sure Tesseract is installed and added to PATH.")
            print("Download: https://github.com/UB-Mannheim/tesseract/wiki")
            return ""
