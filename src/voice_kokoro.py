from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import time
import re

class JarvisVoice:
    def __init__(self):
        print("Initializing JARVIS Voice System (Kokoro TTS)...")
        # Available voices in Kokoro
        self.voices = {
            "jarvis": "am_adam",      # American Male - authoritative
            "friday": "af_heart"       # American Female - expressive and warm (DEMO voice!)
        }
        self.persona = "friday"  # Default to female voice
        # Expressiveness settings
        self.speed = 1  # Slightly slower for more natural expression (0.5-2.0)
        self.volume = 2.1  # Volume multiplier (1.0 = normal, 1.5 = 50% louder)
        # Initialize Kokoro pipeline (American English)
        self.pipeline = KPipeline(lang_code='a')
        # Sample rate (Kokoro outputs at 24kHz)
        self.sample_rate = 24000
        # Greeting flag
        self._greeted = False
        print(f"✓ Kokoro TTS initialized (Persona: {self.persona.upper()})")
        print("   Model: Kokoro-82M | #1 in TTS Arena")
        # Initial greeting on first use
        self._initial_greeting()

    def set_persona(self, persona):
        """Switch between different voice personas"""
        if persona in self.voices:
            self.persona = persona
            print(f"[Voice] Switched to {persona.upper()}")
            return True
        else:
            print(f"[Voice] Unknown persona: {persona}")
            return False

    def _initial_greeting(self):
        """Speak a standard greeting the first time the engine is created"""
        if not self._greeted:
            self._greeted = True
            try:
                voice_name = self.voices[self.persona]
                audio_chunks = []
                for i, (gs, ps, audio) in enumerate(self.pipeline("Hello, how may I help you?", voice=voice_name, speed=self.speed)):
                    audio_chunks.append(audio)
                if audio_chunks:
                    audio_samples = np.concatenate(audio_chunks) * self.volume
                    # Clip to prevent distortion
                    audio_samples = np.clip(audio_samples, -1.0, 1.0)
                    sd.play(audio_samples, self.sample_rate)
                    sd.wait()
            except Exception as e:
                print(f"[Voice] Greeting error: {e}")

    def speak(self, text):
        """Generate and play speech from text with naturalness tweaks (no intro phrases)"""
        if not text or len(text.strip()) == 0:
            print("[Voice] No text to speak")
            return

        # ---- Preprocess text for naturalness ----
        filler_patterns = [
            r'^\s*so\s+it\s+looks\s+like\s*[:,]?\s*',
            r'^\s*it\s+looks\s+like\s*[:,]?\s*',
            r'^\s*looks\s+like\s*[:,]?\s*'
        ]
        cleaned = text
        for pat in filler_patterns:
            cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

        # No intro phrases – speak directly
        print(f"[Voice] Generating: {cleaned[:50]}{'...' if len(cleaned) > 50 else ''}")
        start_time = time.time()
        try:
            voice_name = self.voices[self.persona]
            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(self.pipeline(cleaned, voice=voice_name, speed=self.speed)):
                audio_chunks.append(audio)
            if not audio_chunks:
                print("[Voice] No audio generated")
                return
            audio_samples = np.concatenate(audio_chunks) * self.volume
            # Clip to prevent distortion
            audio_samples = np.clip(audio_samples, -1.0, 1.0)
            
            generation_time = time.time() - start_time
            duration = len(audio_samples) / self.sample_rate
            rtf = generation_time / duration if duration > 0 else float('inf')
            print(f"[Voice] Generated in {generation_time:.2f}s | Duration: {duration:.2f}s | RTF: {rtf:.2f}x")
            sd.play(audio_samples, self.sample_rate)
            sd.wait()
            total_time = time.time() - start_time
            print(f"[Voice] Total time: {total_time:.2f}s")
        except Exception as e:
            print(f"[Voice] Error during synthesis: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    voice = JarvisVoice()
    voice.speak("Hello! I am Friday, using the Heart voice. How may I assist you today?")
