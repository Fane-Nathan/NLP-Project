"""
Kokoro TTS Voice Configuration Guide

EXPRESSIVENESS SETTINGS
========================

1. SPEED (Current: 0.95)
   - Range: 0.5 - 2.0
   - Default: 1.0
   - Recommended for expressiveness: 0.85 - 0.95
   
   Effects:
   - 0.85 - 0.90: Very expressive, dramatic pauses, emotional
   - 0.90 - 0.95: Natural, conversational, friendly (CURRENT)
   - 1.0: Standard, clear, professional
   - 1.1 - 1.3: Fast, energetic, urgent
   - 1.5+: Very fast, robotic

2. VOICE OPTIONS
   Current: af_sarah (warm, expressive female)
   
   Other female voices to try:
   - af_bella: More energetic, youthful
   - af_nicole: Professional, clear
   - af_sky: Soft, gentle
   
   Male voices:
   - am_adam: Deep, authoritative (default Jarvis)
   - am_michael: Friendly, warm

3. TEXT FORMATTING FOR EXPRESSIVENESS
   
   Add emphasis with:
   - CAPS for emphasis: "This is VERY important"
   - Ellipsis for pauses: "Well... I suppose..."
   - Exclamation marks: "That's amazing!"
   - Question marks: "Really?"
   - Commas for pacing: "First, second, and third"

4. EXAMPLES

   More Expressive (speed=0.85):
   ---
   voice.speed = 0.85
   voice.speak("Well... that is VERY interesting! I wonder what this means for us?")
   
   Professional (speed=1.0):
   ---
   voice.speed = 1.0
   voice.speak("The system is operational and ready for deployment.")
   
   Energetic (speed=1.15):
   ---
   voice.speed = 1.15
   voice.speak("Quick update! The process is complete and everything looks great!")

5. ADVANCED: Custom Voice Tensors
   
   You can load custom voice .pt files:
   ---
   import torch
   voice_tensor = torch.load('path/to/custom_voice.pt', weights_only=True)
   generator = pipeline(text, voice=voice_tensor, speed=0.95)

USAGE IN JARVIS
================

To adjust expressiveness, edit src/voice_kokoro.py:

Line 18-19:
    self.persona = "friday"  # or "jarvis"
    self.speed = 0.95        # Adjust 0.5-2.0

For different moods:
- Calm assistant: speed=0.90, voice="af_sarah"
- Energetic helper: speed=1.15, voice="af_bella"  
- Professional: speed=1.0, voice="af_nicole"
- Authoritative: speed=0.95, voice="am_adam"
"""
