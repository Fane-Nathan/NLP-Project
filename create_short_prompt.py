import torchaudio
import torch

# Load original prompt
audio, sr = torchaudio.load('index-tts/examples/voice_05.wav')
print(f"Original: {audio.shape}, SR: {sr}, Duration: {audio.shape[-1]/sr:.2f}s")

# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)
    sr = 16000
    print(f"Resampled to 16kHz: {audio.shape}, Duration: {audio.shape[-1]/sr:.2f}s")

# Extract first 3 seconds (optimal for CosyVoice)
max_samples = 3 * sr  # 3 seconds
if audio.shape[-1] > max_samples:
    audio_short = audio[:, :max_samples]
    print(f"Trimmed to 3s: {audio_short.shape}, Duration: {audio_short.shape[-1]/sr:.2f}s")
else:
    audio_short = audio
    print("Audio already shorter than 3s, using full audio")

# Save optimized prompt
torchaudio.save('index-tts/examples/voice_05_short.wav', audio_short, sr)
print("✓ Saved to: index-tts/examples/voice_05_short.wav")
