import mss
import mss.tools
from PIL import Image
import io
import base64

class ScreenCapturer:
    """
    Handles screen capture operations.
    """
    def __init__(self):
        pass

    def capture_screen(self) -> Image.Image:
        """
        Captures the primary monitor.
        Returns:
            PIL Image object.
        """
        with mss.mss() as sct:
            # Capture the first monitor (usually the primary one)
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img

    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string for API usage.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
