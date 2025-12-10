# Echo Voice Assistant 🤖🔊

A powerful, local AI assistant capable of screen analysis and ultra-fast, high-quality voice interaction using **Kokoro TTS**.

## 🚀 Features

- **Ultra-Fast Local TTS:** Uses **Kokoro-82M** (#1 ranked model) for high-quality speech with ~0.27x Real-Time Factor (4x faster than real-time).
- **Screen Intelligence:** Captures and summarizes screen content using Vision LLMs (Groq) and OCR.
- **Voice Personas:**
  - **Friday:** Warm, expressive female voice (`af_heart` - US Heart ❤️)
  - **Echo:** Authoritative male voice (`am_adam`)
- **Hotkeys:** Global keyboard shortcuts for instant interaction.
- **Privacy Focused:** Runs TTS locally on your machine.

## 🛠️ Requirements

- **Python 3.11** (Required for Kokoro compatibility)
- **GPU:** NVIDIA GPU recommended (runs on GTX 1650 with ~500MB VRAM usage)
- **OS:** Windows (tested), Linux, MacOS

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd echo-assistant
   ```

2. **Create a Python 3.11 environment:**
   ```bash
   py -3.11 -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   *Note: This installs PyTorch (CUDA 12.1), Kokoro, and other required packages.*

4. **Download Models:**
   - Kokoro models will automatically download on the first run (~350MB).

## 🎮 Usage

Run the assistant:
```bash
python src/assistant.py
```

### Hotkeys
| Shortcut | Action |
|----------|--------|
| **Ctrl+Alt+S** | **Summarize Screen** - Analyzes current window content |
| **Ctrl+Alt+D** | **Describe Screen** - Detailed visual description |
| **Ctrl+Alt+V** | **Toggle Voice** - Switch between Friday and Echo |
| **Esc** | **Exit** - Shutdown the assistant |

## 📂 Project Structure

```
├── src/
│   ├── assistant.py       # Main entry point & logic
│   ├── voice_kokoro.py    # Kokoro TTS wrapper (The Voice)
│   ├── screen_capture.py  # MSS screen capture logic
│   ├── ocr_engine.py      # Tesseract OCR wrapper
│   ├── monitor.py         # Active window monitoring
│   └── models/            # LLM handlers
├── requirements.txt       # Project dependencies
└── README.md              # This file
```

## ⚙️ Configuration

### Voice Settings
Edit `src/voice_kokoro.py` to tweak:
- **Volume:** `self.volume = 1.5` (Default: 1.5x boost)
- **Speed:** `self.speed = 1.0` (0.5 - 2.0)
- **Voices:** Change default voices in `self.voices` dict.

## 🏆 Credits

- **Kokoro TTS:** [hexgrad/kokoro](https://huggingface.co/hexgrad/Kokoro-82M) - The amazing TTS model powering this project.
