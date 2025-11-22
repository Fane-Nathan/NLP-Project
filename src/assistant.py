import keyboard
import time
import pyperclip
import threading
import re
import pyperclip
import threading
from src.screen_capture import ScreenCapturer
from src.models.llm_groq import GroqSummarizer
from src.ocr_engine import OCREngine
from src.voice_kokoro import JarvisVoice
from src.monitor import WindowMonitor

def main():
    print("Initializing JARVIS System...")
    
    try:
        capturer = ScreenCapturer()
        summarizer = GroqSummarizer()
        ocr = OCREngine()
        voice = JarvisVoice()
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\n=== JARVIS Online ===")
    # voice.speak("Systems online. Ready to assist.")
    
    print("Hotkeys:")
    print("  Ctrl+Alt+S: Summarize Screen")
    print("  Ctrl+Alt+D: Describe Screen")
    print("  Ctrl+Alt+V: Toggle Voice (Friday/Jarvis)")
    print("  Esc: Exit")

    # --- Proactive Monitoring ---
    def on_interesting_window(title):
        msg = f"Sir, I see you are looking at {title}. Shall I summarize it?"
        print(f"\n[Monitor] {msg}")
        voice.speak(msg)

    monitor = WindowMonitor(callback=on_interesting_window)
    monitor.start()

    # --- Helper Functions ---
    def clean_text_for_speech(text):
        """Removes markdown and special characters for better TTS."""
        if not text: return ""
        # Remove bold/italic markers
        text = re.sub(r'[*_#`]', '', text)
        # Remove links [text](url) -> text
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove raw URLs
        text = re.sub(r'http\S+', 'website', text)
        return text.strip()

    # --- Action Handlers ---
    def process_screen(mode="summarize"):
        print(f"\n[Processing: {mode.upper()}]")
        # voice.speak("Processing visual data.")
        
        try:
            # 1. Capture
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            result = None
            
            # 2. Vision Analysis
            if mode == "summarize":
                result = summarizer.summarize_image(image_base64)
                # Fallback to OCR if Vision fails for summarization
                if not result:
                    voice.speak("Vision sensors unclear. Switching to text extraction.")
                    text = ocr.extract_text(image)
                    if text and len(text) > 10:
                        result = summarizer.summarize_text(text)
            
            elif mode == "describe":
                result = summarizer.describe_image(image_base64)

            # 3. Output
            if result:
                print("\n=== RESULT ===\n")
                print(result)
                print("\n==============\n")
                pyperclip.copy(result)
                # voice.speak("Analysis complete. Result copied to clipboard.")
                voice.speak(clean_text_for_speech(result)) # Read clean result aloud
            else:
                voice.speak("I could not analyze the screen content, sir.")
                
        except Exception as e:
            print(f"Error: {e}")
            voice.speak("An error occurred during processing.")

    def toggle_voice():
        new_persona = "jarvis" if voice.persona == "friday" else "friday"
        voice.set_persona(new_persona)
        voice.speak(f"Voice protocol switched to {new_persona.capitalize()}.")

    # Register Hotkeys
    keyboard.add_hotkey('ctrl+alt+s', lambda: process_screen("summarize"))
    keyboard.add_hotkey('ctrl+alt+d', lambda: process_screen("describe"))
    keyboard.add_hotkey('ctrl+alt+v', toggle_voice)
    
    # Keep running
    keyboard.wait('esc')
    
    # Cleanup
    monitor.stop()
    voice.speak("Shutting down.")
    print("Exiting...")

if __name__ == "__main__":
    main()
