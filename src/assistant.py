import keyboard
import time
import pyperclip
import threading
import re
from src.screen_capture import ScreenCapturer
from src.models.llm_groq import GroqSummarizer
from src.ocr_engine import OCREngine
from src.voice_kokoro import JarvisVoice
from src.monitor import WindowMonitor

# Hoax detection imports
try:
    from src.hoax_detection import HoaxClassifier
    HOAX_DETECTION_AVAILABLE = True
except ImportError:
    HOAX_DETECTION_AVAILABLE = False
    print("[Warning] Hoax detection module not available")


def main():
    print("Initializing JARVIS System...")
    
    try:
        capturer = ScreenCapturer()
        summarizer = GroqSummarizer()
        ocr = OCREngine()
        voice = JarvisVoice()
        
        # Initialize hoax classifier
        hoax_classifier = None
        if HOAX_DETECTION_AVAILABLE:
            try:
                hoax_classifier = HoaxClassifier("models/hoax_indobert_lora")
                print("✓ Hoax Detector loaded (99.5% accuracy)")
            except Exception as e:
                print(f"[Warning] Hoax classifier not loaded: {e}")
                
    except Exception as e:
        print(f"Initialization failed: {e}")
        return

    print("\n=== JARVIS Online ===")
    
    print("Hotkeys:")
    print("  Ctrl+Alt+S: Summarize Screen")
    print("  Ctrl+Alt+D: Describe Screen")
    print("  Ctrl+Alt+H: Hoax Check (NEW!)")  # NEW
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
        text = re.sub(r'[*_#`]', '', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'http\S+', 'website', text)
        return text.strip()

    # --- Action Handlers ---
    def process_screen(mode="summarize"):
        print(f"\n[Processing: {mode.upper()}]")
        
        try:
            # 1. Capture
            image = capturer.capture_screen()
            image_base64 = capturer.image_to_base64(image)
            
            result = None
            
            # 2. Vision Analysis
            if mode == "summarize":
                result = summarizer.summarize_image(image_base64)
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
                voice.speak(clean_text_for_speech(result))
            else:
                voice.speak("I could not analyze the screen content, sir.")
                
        except Exception as e:
            print(f"Error: {e}")
            voice.speak("An error occurred during processing.")

    # --- NEW: Hoax Check Handler ---
    def check_hoax():
        """Extract text from screen and check for hoax/misinformation."""
        print("\n[Processing: HOAX CHECK]")
        
        if not hoax_classifier:
            voice.speak("Hoax detection is not available. Please ensure the model is trained.")
            return
        
        try:
            # 1. Capture screen
            voice.speak("Scanning for misinformation.")
            image = capturer.capture_screen()
            
            # 2. Extract text via OCR
            text = ocr.extract_text(image)
            
            if not text or len(text.strip()) < 20:
                voice.speak("I couldn't extract enough text from the screen. Please try again with more visible text.")
                return
            
            print(f"[OCR] Extracted {len(text)} characters")
            print(f"[OCR] Preview: {text[:200]}...")
            
            # 3. Run hoax classification
            result = hoax_classifier.predict(text)
            
            # 4. Generate response
            print("\n=== HOAX ANALYSIS ===")
            print(f"Label: {result.label}")
            print(f"Confidence: {result.confidence:.1%}")
            print(f"Hoax Probability: {result.hoax_probability:.1%}")
            print("=====================\n")
            
            # 5. Speak result
            if result.hoax_probability >= 0.7:
                # High confidence hoax
                response = f"Warning! This content has a {result.hoax_probability:.0%} probability of being misinformation. I recommend fact-checking before sharing."
                print(f"🚨 {response}")
            elif result.hoax_probability >= 0.5:
                # Uncertain
                response = f"This content is uncertain. There's a {result.hoax_probability:.0%} chance it could be misinformation. Please verify with trusted sources."
                print(f"⚠️ {response}")
            else:
                # Likely valid
                response = f"This content appears credible with {result.valid_probability:.0%} confidence. However, always verify important information."
                print(f"✅ {response}")
            
            voice.speak(response)
            
            # Copy result to clipboard
            clipboard_text = f"""
HOAX ANALYSIS RESULT
====================
Label: {result.label}
Hoax Probability: {result.hoax_probability:.1%}
Valid Probability: {result.valid_probability:.1%}
Confidence: {result.confidence:.1%}

Text Analyzed:
{text[:500]}{'...' if len(text) > 500 else ''}
"""
            pyperclip.copy(clipboard_text)
            print("[Clipboard] Analysis copied")
                
        except Exception as e:
            print(f"Error during hoax check: {e}")
            import traceback
            traceback.print_exc()
            voice.speak("An error occurred during the hoax analysis.")

    # --- NEW: Hoax Check with LLM Enhancement ---
    def check_hoax_with_llm():
        """Extract text, check hoax, and get LLM explanation."""
        print("\n[Processing: ENHANCED HOAX CHECK]")
        
        if not hoax_classifier:
            voice.speak("Hoax detection is not available.")
            return
        
        try:
            voice.speak("Performing deep credibility analysis.")
            image = capturer.capture_screen()
            
            # Extract text
            text = ocr.extract_text(image)
            
            if not text or len(text.strip()) < 20:
                voice.speak("Not enough text to analyze.")
                return
            
            # Run hoax classification
            result = hoax_classifier.predict(text)
            
            # Build prompt for LLM
            analysis_prompt = f"""Analyze this content for misinformation. My AI detector classified it as {result.label} with {result.hoax_probability:.0%} hoax probability.

Content:
"{text[:1000]}"

Provide a brief 2-3 sentence analysis:
1. Why it might be hoax/valid
2. What red flags or credibility indicators you see
3. Recommendation for the reader

Be concise and speak naturally."""

            # Get LLM analysis
            llm_analysis = summarizer.summarize_text(analysis_prompt)
            
            if llm_analysis:
                print("\n=== LLM ANALYSIS ===")
                print(llm_analysis)
                print("====================\n")
                
                # Combine results
                full_response = f"Hoax probability: {result.hoax_probability:.0%}. {llm_analysis}"
                voice.speak(clean_text_for_speech(full_response))
                pyperclip.copy(llm_analysis)
            else:
                # Fallback to basic response
                check_hoax()
                
        except Exception as e:
            print(f"Error: {e}")
            voice.speak("Analysis failed.")

    def toggle_voice():
        new_persona = "jarvis" if voice.persona == "friday" else "friday"
        voice.set_persona(new_persona)
        voice.speak(f"Voice protocol switched to {new_persona.capitalize()}.")

    # Register Hotkeys
    keyboard.add_hotkey('ctrl+alt+s', lambda: process_screen("summarize"))
    keyboard.add_hotkey('ctrl+alt+d', lambda: process_screen("describe"))
    keyboard.add_hotkey('ctrl+alt+h', check_hoax)  # NEW: Basic hoax check
    keyboard.add_hotkey('ctrl+alt+j', check_hoax_with_llm)  # NEW: Enhanced with LLM
    keyboard.add_hotkey('ctrl+alt+v', toggle_voice)
    
    # Keep running
    keyboard.wait('esc')
    
    # Cleanup
    monitor.stop()
    voice.speak("Shutting down.")
    print("Exiting...")


if __name__ == "__main__":
    main()