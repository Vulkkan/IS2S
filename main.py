# Volume-based interruption - speak louder than bot to interrupt it.
import re
import time
import wave
import queue
import threading
from queue import Queue
from time import sleep
from enum import Enum
import numpy as np
import pyaudio
import requests
import speech_recognition as sr
from piper import PiperVoice
from faster_whisper import WhisperModel
from collections import deque
import webrtcvad


# ================== CONFIG ==================
MODEL: str = 'smollm2:135m' # LLM
DEFAULT_MIC: str = "pipewire"  # LINUX, change to the default from Mac (Built-in microphone)
ENERGY_THRESHOLD: int = 1000 # When voice is raised above this threshold, interruption occurs. Keep the volume lower and speak louder.

# ==== Whisper (TTS) ==== #
WHISPER_THREADS: int = 4
LENGTH_IN_SEC: int = 5
MAX_SENTENCE_CHARACTERS = 80
RECORD_TIMEOUT: int = 2

# === Interruption / VAD ===
VAD_FRAME_MS: int = 20           # 10/20/30 ms supported
VAD_SENSITIVITY: int = 2         # 0=least, 3=most aggressive
BARGE_IN_MS: int = 250           # how long user must speak to interrupt


class Mode(Enum):
    LISTENING: int = 1
    SPEAKING: int = 2

class AudioTranscriber:
    def __init__(self):
        self.audio_model = WhisperModel(
            model_size_or_path='tiny.en',
            device="cpu",
            compute_type="int8",
            cpu_threads=WHISPER_THREADS,
            download_root="./models",
        )
        self.transcribe = threading.Event()
        self.thread_terminate = threading.Event()
        self.audio_queue = Queue()
        self.length_queue = Queue(maxsize=LENGTH_IN_SEC)
        self.transcription_queue = Queue()
        self.vad = webrtcvad.Vad(VAD_SENSITIVITY)
        self.mode_state = Mode.LISTENING
        self.voice_window = deque(maxlen=BARGE_IN_MS // VAD_FRAME_MS)

    def process_audio_segment(self, audio_data):
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = self.audio_model.transcribe(audio_np, language='en', beam_size=5)
        return "".join(segment.text for segment in segments)

    def clear_queues(self):
        with self.length_queue.mutex:
            self.length_queue.queue.clear()
        while not self.audio_queue.empty():
            self.audio_queue.get()

    def consumer_loop(self):
        transcription = ""
        pause_count = 0
        stored_transcription = []

        while not self.thread_terminate.is_set():
            if self.length_queue.qsize() >= LENGTH_IN_SEC:
                self.clear_queues()
                stored_transcription.append(transcription)
                transcription = ""
                print()

            try:
                audio_data = self.audio_queue.get(timeout=2)
                self.length_queue.put(audio_data)
                self.audio_queue.task_done()

                if not self.transcribe.is_set():
                    self.transcribe.set()

                audio_data_to_process = b"".join(list(self.length_queue.queue))
                transcription = self.process_audio_segment(audio_data_to_process)
                transcription = re.sub(r"\s\s+", "", transcription)

                if transcription:
                    print(transcription, end="\r", flush=True)
                    pause_count = 0
                else:
                    pause_count += 1

            except queue.Empty:
                pause_count += 1

            if pause_count >= 1:
                if self.transcribe.is_set():
                    self.handle_pause(transcription, stored_transcription)
                    transcription = ""

    def handle_pause(self, transcription, stored_transcription):
        self.transcribe.clear()
        if transcription:
            stored_transcription.append(transcription)

        audio_data_to_process = b""
        while not self.audio_queue.empty():
            audio_data_to_process += self.audio_queue.get(timeout=1)

        if audio_data_to_process:
            final_transcription = self.process_audio_segment(audio_data_to_process)
            final_transcription = re.sub(r"\s\s+", "", final_transcription)
            if final_transcription:
                stored_transcription.append(final_transcription)

        self.transcription_queue.put("".join(stored_transcription))
        self.clear_queues()

class ChatAssistant:
    def __init__(self):
        self.voice = PiperVoice.load("models/piper/en_GB-alba-medium.onnx")
        self.output_file = "output.wav"
        self.messages = [
            {"role": "system", "content": "You are a conversational chatbot."},
        ]
        self.pyaudio_instance = pyaudio.PyAudio()

    def synthesize_speech(self, text):
        with wave.open(self.output_file, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

    def play_audio(self, transcribe_event, terminate_event):
        wf = wave.open(self.output_file, "rb")
        
        def callback(in_data, frame_count, time_info, status):
            if transcribe_event.is_set() or terminate_event.is_set():
                return (None, pyaudio.paComplete)
            data = wf.readframes(frame_count)
            return (data, pyaudio.paContinue if data else pyaudio.paComplete)
        
        stream = self.pyaudio_instance.open(
            format=self.pyaudio_instance.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
            stream_callback=callback,
        )
        stream.start_stream()
        while stream.is_active():
            sleep(0.1)
            if terminate_event.is_set():
                break
        stream.stop_stream()
        stream.close()
        wf.close()

    def get_llm_response(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        try:
            response = requests.post(
                "http://localhost:11434/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": self.messages,
                    "stream": False,
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as e:
            print(f"Ollama error: {e}")
            return f"[Ollama error: {e}]"

    def chat_loop(self, transcription_queue, transcribe_event, terminate_event):
        while not terminate_event.is_set():
            sleep(1)
            if transcribe_event.is_set():
                continue

            try:
                if not transcription_queue.empty():
                    user_input = transcription_queue.get(timeout=1)
                    bot_reply = self.get_llm_response(user_input)
                    
                    self.synthesize_speech(bot_reply)
                    print("========== Ollama is answering ==========")
                    
                    self.play_audio(transcribe_event, terminate_event)
                    self.messages.append({"role": "assistant", "content": bot_reply})
                    transcription_queue.task_done()

            except Exception as e:
                print("Exception occurred:", e)
                continue

        self.pyaudio_instance.terminate()

class MicrophoneHandler:
    def __init__(self, audio_queue, energy_threshold=ENERGY_THRESHOLD, default_mic=DEFAULT_MIC):
        self.audio_queue = audio_queue
        self.energy_threshold = energy_threshold
        self.default_mic = default_mic
        self.recorder = sr.Recognizer()
        
        self.recorder.energy_threshold = energy_threshold
        self.recorder.dynamic_energy_threshold = False

    def find_microphone(self):
        mic_names = sr.Microphone.list_microphone_names()
        for index, name in enumerate(mic_names):
            if str(self.default_mic) in str(index) or self.default_mic in name:
                return index
        print(f"Microphone '{self.default_mic}' not found.")
        return None

    def record_callback(self, _, audio: sr.AudioData):
        data = audio.get_raw_data()
        self.audio_queue.put(data)

    def start_listening(self, transcribe_event):
        mic_index = self.find_microphone()
        if mic_index is None:
            return None

        source = sr.Microphone(device_index=mic_index, sample_rate=16000)
        with source:
            self.recorder.adjust_for_ambient_noise(source, duration=1.5)

        try:
            return self.recorder.listen_in_background(
                source, self.record_callback, phrase_time_limit=RECORD_TIMEOUT
            )
        except Exception as e:
            print(f"Failed to start microphone stream: {e}")
            return None

def main():
    transcriber = AudioTranscriber()
    assistant = ChatAssistant()
    mic_handler = MicrophoneHandler(transcriber.audio_queue)

    print("=======Wait for model speech and start talk after hearing it.==========")
    
    stop_listening = mic_handler.start_listening(transcriber.transcribe)
    if not stop_listening:
        return

    consumer_thread = threading.Thread(
        target=transcriber.consumer_loop
    )
    consumer_thread.start()

    chat_thread = threading.Thread(
        target=assistant.chat_loop,
        args=(transcriber.transcription_queue, transcriber.transcribe, transcriber.thread_terminate)
    )
    chat_thread.start()

    try:
        while consumer_thread.is_alive():
            sleep(3)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt occurs, stopping threads...")
        transcriber.thread_terminate.set()
        stop_listening(wait_for_stop=False)
        consumer_thread.join()
        chat_thread.join()

    print("The program stops.")

if __name__ == "__main__":
    main()
