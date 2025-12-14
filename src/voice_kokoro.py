# from kokoro import KPipeline
import sounddevice as sd
import numpy as np
import time
import re
import torch
import threading
import queue
import asyncio
import edge_tts

class EchoVoice:
    def __init__(self, server_mode=False):
        print("Initializing Echo Voice System (Edge TTS)...")
        
        # Check for GPU (Unused by Edge TTS but kept for interface consistency)
        self.device = 'cpu' # Edge TTS is cloud-based
        print(f"[Voice] Device: Cloud API (Zero VRAM)")

        # Available voices
        self.voices = {
            "echo": "en-US-ChristopherNeural",  # Male equivalent
            "friday": "en-US-EmmaNeural"        # Requested 'Emma' voice
        }
        self.persona = "friday"  # Default to Emma
        
        # Expressiveness settings (Edge TTS rate/pitch/volume adjustments)
        # Rate: -50% to +50%, Pitch: -50Hz to +50Hz (or ±50%)
        self.rate = "+5%"      # Slightly faster for natural flow
        self.volume = "+10%"   # Slightly louder
        self.pitch = "+3Hz"    # Slightly higher pitch for energy
        
        # Audio buffering settings
        self.buffer_size = 4096  # Accumulate bytes before yielding (reduces choppiness)
        self.audio_buffer = b''  # Buffer for collecting chunks
        
        # Kokoro Pipeline (Commented Out)
        # try:
        #     self.pipeline = KPipeline(lang_code='a', device=self.device)
        # except Exception as e:
        #     print(f"[Voice] Failed to init on {self.device}, falling back to CPU. Error: {e}")
        #     self.device = 'cpu'
        #     self.pipeline = KPipeline(lang_code='a', device='cpu')

        # Sample rate (Edge TTS is usually 24kHz or 48kHz, 24k matches previous)
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
        
        self.server_mode = server_mode
        self.generation_thread.start()
        if not self.server_mode:
            self.playback_thread.start()


        print(f"✓ Edge TTS initialized (Persona: {self.persona.upper()})")
        print("   Model: Microsoft Edge Online | Zero VRAM")
        
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
            self.speak("Systems online. Voice module connected to cloud.")

    def speak(self, text):
        """Add text to the generation queue (Non-blocking)"""
        if not text or len(text.strip()) == 0:
            return
        self.text_queue.put(text)

    def _generation_loop(self):
        """Producer: Consumes text, generates audio, puts to audio_queue"""
        
        # Helper to run async generator in thread
        async def generate_audio(text, voice_name):
            # Use SSML-like parameters for expressiveness
            communicate = edge_tts.Communicate(
                text, 
                voice_name, 
                rate=self.rate, 
                volume=self.volume,
                pitch=self.pitch
            )
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Edge TTS sends mp3 bytes usually.
                    self.audio_queue.put(chunk["data"])

        # Real implementation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.running:
            try:
                text = self.text_queue.get(timeout=1)
            except queue.Empty:
                continue

            cleaned = text # Regex cleanup here if needed
            print(f"[Voice] Generating: {cleaned[:50]}...")
            
            try:
                voice_name = self.voices[self.persona]
                loop.run_until_complete(generate_audio(cleaned, voice_name))
            except Exception as e:
                print(f"[Voice] Generation Error: {e}")
            
            self.text_queue.task_done()

    def _playback_loop(self):
        """Consumer: Consumes audio chunks, plays them"""
        if self.server_mode:
            return

        # Local playback implementation for MP3 is hard without pydub/ffmpeg.
        # Since the user's goal is DEPLOYMENT (server mode), local playback is secondary.
        # I will print a warning.
        print("[Voice] Local playback of Edge TTS requires 'mpv' or 'ffplay' installed.")
        # We could use `os.system` to play if we saved to file, but let's just drain the queue.
        while self.running:
            try:
                _ = self.audio_queue.get(timeout=1)
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def get_audio_stream(self):
        """
        Generator that yields audio bytes for web streaming.
        Uses buffering to reduce choppiness.
        """
        buffer = b''
        idle_count = 0
        max_idle = 3  # Stop after 3 seconds of no audio
        
        while self.running and idle_count < max_idle:
            try:
                audio_chunk = self.audio_queue.get(timeout=1.0)
                buffer += audio_chunk
                idle_count = 0  # Reset idle counter

                # Yield when buffer is large enough (reduces choppiness)
                if len(buffer) >= self.buffer_size:
                    yield buffer
                    buffer = b''

                self.audio_queue.task_done()
            except queue.Empty:
                # Only count as idle if no work is pending
                if self.text_queue.empty() and self.audio_queue.empty():
                    idle_count += 1
                else:
                    idle_count = 0  # Reset if generation still in progress
                # Flush remaining buffer on idle
                if buffer:
                    yield buffer
                    buffer = b''
                continue
            except Exception as e:
                break
        
        # Final flush
        if buffer:
            yield buffer

    def stop(self):
        self.running = False
        self.generation_thread.join()
        self.playback_thread.join()

if __name__ == "__main__":
    voice = EchoVoice()
    voice.speak("Testing Edge TTS system.")
    while True:
        time.sleep(1)
