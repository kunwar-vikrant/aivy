# 🎙️ Aivy: Your Voice Assistant Brought to Life! 🤖🚀

Welcome to **Aivy**, an open-source voice assistant that channels the charm and intelligence of Tony Stark’s iconic AI companion! Built with Python, Aivy combines **real-time speech-to-text (STT)**, **text-to-speech (TTS)**, and a **local large language model (LLM)** to deliver seamless, witty conversations. Whether you’re asking about the weather or craving a clever quip, Aivy is here to assist with style! 😎

## ✨ Features

- 🎙️ **Real-Time Speech Recognition**: Transcribes speech after 2 seconds of talking followed by 1.5 seconds of silence using `faster_whisper`.
- 🗣️ **Human-Like Speech Synthesis**: Generates natural-sounding audio with the `mimi` TTS model (CSM-1B).
- 🧠 **Witty Conversations**: Powered by LLaMA-3.2-1B via LM Studio, delivering Iron Man-style responses.
- 🔄 **Overlap Prevention**: Blocks new speech during responses for clear, uninterrupted interactions.
- 🌐 **Cross-Platform**: Runs on Linux, macOS, and Windows with GPU or CPU support.

## 🛠️ Getting Started

Follow these steps to bring Aivy to life on your system. Whether you’re on **Linux**, **macOS**, or **Windows**, Aivy is designed to be easy to set up and fun to use!

### Prerequisites

Before you begin, ensure you have the following:

- **Python 3.8+**: Check with `python --version`.
- **CUDA (Optional)**: For GPU acceleration (requires an NVIDIA GPU and CUDA toolkit).
- **FFmpeg**: For audio processing.
  - **Ubuntu**: `sudo apt install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Windows**: Download from FFmpeg and add to your PATH.
- **LM Studio**: To serve LLaMA-3.2-1B locally (download).
- **Models**:
  - **LLaMA-3.2-1B**: Available via Hugging Face or LM Studio (requires a Hugging Face account).
  - **CSM-1B**: `mimi` weights from Hugging Face.
- **Microphone and Speakers**: Any standard USB, Bluetooth, or built-in mic/speaker.
- **Git**: To clone the repository (`git --version`).

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/aivy.git
   cd aivy
   ```

2. **Set Up a Virtual Environment**:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   **requirements.txt** (included in the repo):

   ```
   faster-whisper
   openai
   sounddevice
   numpy
   torch
   torchaudio
   transformers
   huggingface_hub
   safetensors
   psutil
   ```

4. **Download Models**:

   - **LLaMA-3.2-1B**:
     - Download via LM Studio or Hugging Face (e.g., `meta-llama/Llama-3.2-1B`).
     - Set the model path environment variable:

       ```bash
       export LLAMA_MODEL_PATH=/path/to/Llama-3.2-1B
       ```
   - **CSM-1B**:
     - Download `mimi` weights from Hugging Face (e.g., `moshi/mimi`).
     - Set the model path:

       ```bash
       export CSM_MODEL_PATH=/path/to/csm-1b
       ```

5. **Configure Audio Devices (Optional)**:

   - Set environment variables for your microphone and speaker:

     ```bash
     export INPUT_DEVICE=0  # Microphone index
     export OUTPUT_DEVICE=0  # Speaker index
     ```
   - Run `python -c "import sounddevice as sd; print(sd.query_devices())"` to list available devices.
   - If not set, Aivy will prompt you to select devices interactively.

6. **Set Environment (Linux Only)**: To ensure compatibility with audio libraries:

   ```bash
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

### Running Aivy

1. **Start LM Studio**:

   - Open LM Studio and load the LLaMA-3.2-1B model.
   - Start the server at `http://localhost:1234/v1` (default settings).

2. **Launch Aivy**:

   ```bash
   python aivy.py
   ```

   - Select your microphone and speaker when prompted (or use `INPUT_DEVICE`/`OUTPUT_DEVICE`).
   - Speak: "Hey Aivy, what’s the weather?" and enjoy a witty response!

3. **Test the Interaction**:

   - Say: "Hey Aivy, tell me a joke."
   - Expected response: "Why did the AI go to therapy? It had an identity crisis, sir!"
   - Aivy will wait for 2 seconds of speech followed by 1.5 seconds of silence before transcribing.

### Example Interaction

```
JARVIS initializing...
Initialization complete in 2.35 seconds
Entering main loop...
Audio stream started...
Buffer size: 32000 samples, amplitude: 0.0450
Detected speech, updating last_speech_time
Waiting for silence (1500/1500ms)
Transcribing audio buffer...
User: Hey Aivy, what's the weather like today?
Sending to LLM...
JARVIS: Sunny with a chance of brilliance, sir! (LLM took 0.20 seconds)
Generating speech for: Sunny with a chance...
Audio generation completed in 8.00 seconds
```

## 🐛 Known Issues

- **Audio Device Detection**: If your mic or speaker isn’t detected, check devices with `python -c "import sounddevice as sd; print(sd.query_devices())"` and set `INPUT_DEVICE`/`OUTPUT_DEVICE` accordingly.
- **Model Paths**: Ensure `LLAMA_MODEL_PATH` and `CSM_MODEL_PATH` are set correctly to avoid loading errors.
- **Latency on CPU**: GPU is recommended for real-time performance; CPU may introduce delays.
- **LM Studio Connection**: If Aivy fails to connect to `http://localhost:1234/v1`, ensure LM Studio is running and the server is active.

File an issue on GitHub if you encounter problems, and include logs for faster resolution!

## 🚀 Future Work & Improvements

Aivy is just the beginning! Here’s a roadmap to make it even more powerful:

### Short-Term 

- 🗣️ **Interruption Handling**: Reintroduce real-time interruption, allowing users to stop playback by speaking (currently disabled for stability).
- 🎛️ **GUI for Device Selection**: Add a simple interface using `tkinter` or `PyQt` to choose audio devices.
- ⚡ **STT Optimization**: Use smaller `faster_whisper` models (e.g., `base`) for faster transcription.

### Medium-Term 

- 🎤 **Wake-Word Detection**: Implement "Hey Aivy" to activate listening using `webrtcvad` or `snowboy`.
- 🌐 **Multilingual Support**: Extend STT and TTS to non-English languages with `whisper-large` and multilingual TTS models.Integrate realtime web search and function calling.
- 🔇 **Noise Cancellation**: Integrate `rnnoise` for robust performance in noisy environments.

### Long-Term 

- ☁️ **Cloud LLM Integration**: Support cloud-based LLMs (e.g., via xAI’s API) for users without powerful hardware.
- 📱 **Mobile App**: Develop a mobile interface for Aivy using Kivy or Flutter.
- 🔌 **Plugin System**: Enable custom commands (e.g., control smart home devices) via a plugin architecture.

## 🤝 Contributing

We’re thrilled to welcome contributions to Aivy! 🙌 Whether you’re fixing bugs, adding features, or improving docs, your help will make Aivy the ultimate voice assistant.

### How to Contribute

1. **Fork the Repo**: Click the "Fork" button on GitHub.
2. **Create a Branch**: `git checkout -b my-feature`.
3. **Make Changes**: Implement your feature or bug fix.
4. **Commit**: `git commit -m "Add cool feature"`.
5. **Push**: `git push origin my-feature`.
6. **Open a Pull Request**: Describe your changes and submit!

### Contribution Ideas

- Debug and stabilize interruption handling (see Issues).
- Add support for alternative TTS models (e.g., VITS, Tacotron).
- Improve documentation with video tutorials or setup guides.
- Optimize GPU memory usage for larger models.

Check the Issues tab for open tasks, or share your ideas in Discussions.

## 📜 License

Aivy is licensed under the **MIT License**. See LICENSE for details.

## 🙏 Acknowledgments

Aivy wouldn’t be possible without these amazing projects:

- **faster_whisper**: Lightning-fast speech recognition.
- **mimi (CSM-1B)**: High-quality text-to-speech synthesis.
- **LLaMA-3.2-1B**: Powering Aivy’s clever responses.
- **LM Studio**: Seamless local LLM serving.
- **sounddevice**: Reliable audio input and output.

Special thanks to the open-source community for inspiring and supporting Aivy’s development! 🌟

## 📞 Contact

Have questions or ideas? Reach out via GitHub Discussions or open an issue. Let’s make Aivy legendary together!

---

⭐ **Star this repo if you love Aivy!** Join us in building the ultimate voice assistant! 🚀🤖
