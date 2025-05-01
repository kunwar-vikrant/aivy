from generator import load_csm_1b, generate_streaming_audio
from faster_whisper import WhisperModel
from openai import OpenAI
import sounddevice as sd
import numpy as np
import torch
import torchaudio
import queue
import time
import threading
import os
import sys

# Fast Whisper STT
class FastWhisperSTT:
    def __init__(self, model_size="tiny", device="cuda", input_device=None):
        print("Initializing Fast Whisper...")
        self.device = device if torch.cuda.is_available() else "cpu"
        try:
            self.model = WhisperModel(model_size, device=self.device, compute_type="float16")
            print(f"Whisper loaded on {self.device}")
        except Exception as e:
            print(f"Whisper init failed: {e}, falling back to CPU")
            self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.queue = queue.Queue(maxsize=100)
        self.samplerate = 16000
        self.blocksize = 1024
        self.input_device = input_device or self.select_input_device()
        self.running = True
        self.min_speech_duration_ms = 2000
        self.silence_duration_ms = 1500
        print("Available audio devices:")
        print(sd.query_devices())
        print(f"Using input device: {self.input_device}")

    def select_input_device(self):
        devices = sd.query_devices()
        print("Available input devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                print(f"{i}: {dev['name']}")
        try:
            choice = int(os.getenv("INPUT_DEVICE", input("Select input device index (default 0): ") or 0))
            return choice if 0 <= choice < len(devices) else 0
        except ValueError:
            return 0

    def audio_callback(self, indata, frames, stream_time_info, status):
        if status:
            print(f"Audio status: {status}")
        audio = np.frombuffer(indata, dtype=np.float32)
        amplitude = np.mean(np.abs(audio))
        if amplitude > 0.001:
            try:
                self.queue.put_nowait((audio, time.time() * 1000))
                print(f"Queued audio amplitude: {amplitude:.4f}")
            except queue.Full:
                print("Queue full, dropping audio chunk")

    def transcribe_stream(self):
        audio_buffer = np.array([], dtype=np.float32)
        timestamps = []
        stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            callback=self.audio_callback,
            device=self.input_device
        )
        last_speech_time = None
        processing = False

        def process():
            nonlocal audio_buffer, timestamps, last_speech_time, processing
            with stream:
                print("Audio stream started...")
                while self.running:
                    if processing:
                        time.sleep(0.1)
                        continue
                    try:
                        audio_chunk, timestamp = self.queue.get(timeout=1.0)
                        audio_buffer = np.concatenate([audio_buffer, audio_chunk])
                        timestamps.append(timestamp)
                        amplitude = np.mean(np.abs(audio_chunk))
                        speech_threshold = 0.07

                        print(f"Buffer size: {len(audio_buffer)} samples, amplitude: {amplitude:.4f}")
                        if amplitude > speech_threshold:
                            last_speech_time = timestamp
                            print("Detected speech, updating last_speech_time")
                        else:
                            print("Silence or low noise detected, not updating last_speech_time")

                        if last_speech_time and len(audio_buffer) >= self.samplerate * (self.min_speech_duration_ms / 1000):
                            time_since_speech = (time.time() * 1000) - last_speech_time
                            if time_since_speech >= self.silence_duration_ms:
                                print("Transcribing audio buffer...")
                                processing = True
                                try:
                                    segments, _ = self.model.transcribe(
                                        audio_buffer,
                                        language="en",
                                        beam_size=10,
                                        vad_filter=True,
                                        vad_parameters=dict(min_silence_duration_ms=self.silence_duration_ms)
                                    )
                                    text = ""
                                    for segment in segments:
                                        if segment.text.strip():
                                            text += segment.text + " "
                                    if text.strip():
                                        print(f"User: {text.strip()}")
                                        yield text.strip()
                                except Exception as e:
                                    print(f"Transcription failed: {e}")
                                audio_buffer = np.array([], dtype=np.float32)
                                timestamps = []
                                last_speech_time = None
                                processing = False
                            else:
                                print(f"Waiting for silence ({time_since_speech:.0f}/{self.silence_duration_ms}ms)")
                    except queue.Empty:
                        print("No audio input detected, waiting...")
                        audio_buffer = audio_buffer[-self.samplerate // 2:]
                        timestamps = timestamps[-len(audio_buffer) // self.blocksize:] if timestamps else []
                        if last_speech_time and (time.time() * 1000) - last_speech_time >= self.silence_duration_ms:
                            audio_buffer = np.array([], dtype=np.float32)
                            timestamps = []
                            last_speech_time = None

        return process()

    def stop(self):
        self.running = False

generator = load_csm_1b(os.getenv("CSM_MODEL_PATH", "cuda"))

def speak(text, output_device=None):
    output_device = output_device or select_output_device()
    print(f"Generating speech for: {text}")
    start_time = time.time()
    try:
        generate_streaming_audio(
            generator=generator,
            text=text,
            speaker=0,
            context=[],
            output_file="streaming_output.wav",
            play_audio=True,
            output_device=output_device
        )
    except Exception as e:
        print(f"Generation error: {e}")
        raise e
    print(f"Speech generated in {time.time() - start_time:.2f} seconds")

def select_output_device():
    devices = sd.query_devices()
    print("Available output devices:")
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:
            print(f"{i}: {dev['name']}")
    try:
        choice = int(os.getenv("OUTPUT_DEVICE", input("Select output device index (default 0): ") or 0))
        return choice if 0 <= choice < len(devices) else 0
    except ValueError:
        return 0

def main():
    print("JARVIS initializing...")
    print("Loading STT...")
    stt = FastWhisperSTT(input_device=None)
    print("Loading LLM client...")
    try:
        llm_client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
    except Exception as e:
        print(f"Failed to connect to LM Studio: {e}. Ensure LM Studio is running at http://localhost:1234/v1.")
        sys.exit(1)
    print("Loading TTS...")
    start_time = time.time()
    print(f"Initialization complete in {time.time() - start_time:.2f} seconds")
    print("Entering main loop...")
    processing_lock = threading.Lock()
    try:
        for text in stt.transcribe_stream():
            if text.strip() and processing_lock.acquire(blocking=False):
                try:
                    print("Sending to LLM...")
                    start_time = time.time()
                    response = llm_client.chat.completions.create(
                        model="llama-3.2-1b-instruct",
                        messages=[
                            {"role": "system", "content": "You are JARVIS from Iron Man. Be witty and concise."},
                            {"role": "user", "content": text}
                        ]
                    ).choices[0].message.content
                    print(f"JARVIS: {response} (LLM took {time.time() - start_time:.2f} seconds)")
                    speak(response)
                finally:
                    processing_lock.release()
            else:
                print("Skipping transcription, response still processing...")
    except KeyboardInterrupt:
        print("Shutting down...")
        stt.stop()

if __name__ == "__main__":
    try:
        import sounddevice
    except ImportError:
        print("sounddevice not found. Install with: pip install sounddevice")
        sys.exit(1)
    main()