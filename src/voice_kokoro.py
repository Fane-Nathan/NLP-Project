from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import time
import re
import torch
import threading
import queue

class JarvisVoice:
    def __init__(self):
        print("Initializing JARVIS Voice System (Kokoro TTS)...")
        
        # Check for GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[Voice] Device: {self.device.upper()}")

        # Available voices in Kokoro
        self.voices = {
            "jarvis": "am_adam",      # American Male - authoritative
            "friday": "af_heart"       # American Female - expressive and warm (DEMO voice!)
        }
        self.persona = "friday"  # Default to female voice
        
        # Expressiveness settings
        self.speed = 1.1
        self.volume = 1.0 
        
        # Initialize Kokoro pipeline
        try:
            self.pipeline = KPipeline(lang_code='a', device=self.device)
        except Exception as e:
            print(f"[Voice] Failed to init on {self.device}, falling back to CPU. Error: {e}")
            self.device = 'cpu'
            self.pipeline = KPipeline(lang_code='a', device='cpu')

        # Sample rate (Kokoro outputs at 24kHz)
        self.sample_rate = 24000
        
        # Queues for Producer-Consumer
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Flags
        self.running = True
        self._greeted = False
        
        # Start Threads
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        
        self.generation_thread.start()
        self.playback_thread.start()

        print(f"✓ Kokoro TTS initialized (Persona: {self.persona.upper()})")
        print("   Model: Kokoro-82M | #1 in TTS Arena")
        
        # Initial greeting
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
        if not self._greeted:
            self._greeted = True
            self.speak("Systems online. Voice module parallelized.")

    def speak(self, text):
        """Add text to the generation queue (Non-blocking)"""
        if not text or len(text.strip()) == 0:
            return
        self.text_queue.put(text)

    def _generation_loop(self):
        """Producer: Consumes text, generates audio, puts to audio_queue"""
        while self.running:
            try:
                text = self.text_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Preprocess
            filler_patterns = [
                r'^\s*so\s+it\s+looks\s+like\s*[:,]?\s*',
                r'^\s*it\s+looks\s+like\s*[:,]?\s*',
                r'^\s*looks\s+like\s*[:,]?\s*'
            ]
            cleaned = text
            for pat in filler_patterns:
                cleaned = re.sub(pat, '', cleaned, flags=re.IGNORECASE)

            print(f"[Voice] Generating: {cleaned[:50]}...")
            
            try:
                voice_name = self.voices[self.persona]
                # Generate chunks
                for i, (gs, ps, audio) in enumerate(self.pipeline(cleaned, voice=voice_name, speed=self.speed)):
                    # Convert Tensor to Numpy if needed
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().numpy()
                    
                    # Apply volume
                    audio = audio * self.volume
                    
                    # Push to audio queue
                    self.audio_queue.put(audio.astype(np.float32))
            except Exception as e:
                print(f"[Voice] Generation Error: {e}")
            
            self.text_queue.task_done()

    def _playback_loop(self):
        """Consumer: Consumes audio chunks, plays them"""
        # Create a persistent stream
        try:
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            stream.start()
            
            while self.running:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                try:
                    stream.write(audio_chunk)
                except Exception as e:
                    print(f"[Voice] Playback Error: {e}")
                
                self.audio_queue.task_done()
            
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"[Voice] Stream Error: {e}")

    def stop(self):
        self.running = False
        self.generation_thread.join()
        self.playback_thread.join()

if __name__ == "__main__":
    voice = JarvisVoice()
    voice.speak("This is the first sentence.")
    voice.speak("This is the second sentence, queued immediately.")
    voice.speak("And this is the third one. All should play smoothly without blocking.")
    
    # Keep main thread alive to let threads work
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        voice.stop()
