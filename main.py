# System doesn't interrupt LLM inference, it only interrupts TTS playback.

import queue
import re
import threading
import time
import wave
from queue import Queue
from time import sleep

import requests
import numpy as np
import speech_recognition as sr # Read MIC
from faster_whisper import WhisperModel # STT
from piper import PiperVoice # TTS
import pyaudio # Read aloud from SPEAKERS


WHISPER_LANGUAGE = "en"
WHISPER_THREADS: int = 4
LENGHT_IN_SEC: int = 5
MAX_SENTENCE_CHARACTERS = 80
RECORD_TIMEOUT: int = 2
ENERGY_THRESHOLD: int = 1000
DEFAULT_MIC = "pipewire" # LINUX


def read_mic(
    audio_queue: Queue,
    default_microphone=None,
    energy_threshold=None,
    record_timeout=None,
):
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    mic_index = None

    mic_names = sr.Microphone.list_microphone_names()
    
    # Match by index/partial name and Assign mic (LINUX)
    for index, name in enumerate(mic_names):
        if str(default_microphone) in str(index) or default_microphone in name:
            mic_index = index
            break

    if mic_index is None:
        print(f"Microphone '{default_microphone}' not found.")
        return
    else:
        source = sr.Microphone(device_index=mic_index, sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source, duration=1.5)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        audio_queue.put(data)
        transcribe.set()

    try:
        stop_listening = recorder.listen_in_background(
            source, record_callback, phrase_time_limit=record_timeout,
        )
        print("=======Wait for model speech and start talk after hearing it.==========")
        return stop_listening

    except Exception as e:
        print(f"Failed to start microphone stream: {e}")
        return


def main():
    global transcribe

    audio_model = WhisperModel(
        model_size_or_path= 'tiny.en',
        device="cpu",
        compute_type="int8",
        cpu_threads=WHISPER_THREADS,
        download_root="./models",
    )

    record_timeout = RECORD_TIMEOUT

    audio_queue = Queue()
    length_queue = Queue(maxsize=LENGHT_IN_SEC)
    transcription_queue = Queue()

    def consumer_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        transcription = ""
        pause_count = 0 
        current_openaithread_id = None
        openaithread = None
        stored_transcription = []
        transcribe.set()

        while True:
            if length_queue.qsize() >= LENGHT_IN_SEC:
                with length_queue.mutex:
                    length_queue.queue.clear()
                    stored_transcription.append(transcription)
                    transcription = ""
                    print()

            if thread_terminate.is_set():
                print("Stopping the main thread and signal to stop subthread...")
                break

            if current_openaithread_id is None:
                openaithread = threading.Thread(
                    target=chat_thread, args=(transcribe, thread_terminate)
                )
                openaithread.start()
                current_openaithread_id = openaithread.native_id
                prepare_start_time = time.time()

            try:
                audio_data = audio_queue.get(timeout=2)
                length_queue.put(audio_data)
                audio_queue.task_done()

                if not transcribe.is_set():
                    transcribe.set()

                audio_data_to_process = b""
                for i in range(length_queue.qsize()):
                    audio_data_to_process += length_queue.queue[i]

                audio_np = (
                    np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                )

                segments, _ = audio_model.transcribe(
                    audio_np, language=WHISPER_LANGUAGE, beam_size=5
                )

                transcription = ""
                for s in segments:
                    transcription += s.text

                transcription = re.sub(r"\s\s+", "", transcription)

                if transcription == "":
                    pause_count += 1
                else:
                    print(transcription, end="\r", flush=True)
                    pause_count = 0

            except queue.Empty:
                pause_count += 1

            if pause_count >= 1:
                if transcribe.is_set():
                    transcribe.clear()
                    if transcription != "":
                        stored_transcription.append(transcription)
                    audio_data_to_process = b""
                    while not audio_queue.empty():
                        audio_data_to_process += audio_queue.get(timeout=1)
                    audio_np = (
                        np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    segments, _ = audio_model.transcribe(
                        audio_np, language=WHISPER_LANGUAGE, beam_size=5
                    )
                    transcription = ""
                    for s in segments:
                        transcription += s.text
                    transcription = re.sub(r"\s\s+", "", transcription)
                    if transcription != "":
                        stored_transcription.append(transcription)
                    transcription_queue.put("".join(stored_transcription))
                    with length_queue.mutex:
                        length_queue.queue.clear()
                    stored_transcription = []

            else:
                if not transcribe.is_set():
                    print("Transcribing...")
                    transcribe.set()

                continue

        if thread_terminate.is_set():
            print("===In consumer terminating process===")
            if transcription != "":
                stored_transcription.append(transcription)
            audio_data_to_process = b""
            while not audio_queue.empty():
                audio_data_to_process += audio_queue.get(timeout=1)
            audio_np = (
                np.frombuffer(audio_data_to_process, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
            segments, _ = audio_model.transcribe(
                audio_np, language=WHISPER_LANGUAGE, beam_size=5
            )
            transcription = ""
            for s in segments:
                transcription += s.text
            transcription = re.sub(r"\s\s+", "", transcription)
            if transcription != "":
                stored_transcription.append(transcription)
            transcription_queue.put("".join(stored_transcription))
            if current_openaithread_id is not None:
                print("Waiting for openai thread to be closed...")
                openaithread.join()
                
        print("consumer thread is closed")

    def chat_thread(transcribe: threading.Event, thread_terminate: threading.Event):
        voice = PiperVoice.load("models/piper/en_GB-alba-medium.onnx")

        output_file = "output.wav"
        response_queue = Queue()
        tts_queue = Queue()

        messages = [
            {
                "role": "system",
                "content": "You are a conversational chatbot.",
            },
        ]

        # Initialize PyAudio
        p = pyaudio.PyAudio()

        def form_message_from_transcription(transcription: str) -> None:
            messages.append({"role": "user", "content": transcription})

        def form_message_from_response(response: str) -> None:
            messages.append({"role": "assistant", "content": response})

        def synthesize_with_piper(text):
            with wave.open(output_file, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)

        def response_handler(history):
            wf = wave.open(output_file, "rb")

            def callback(in_data, frame_count, time_info, status):
                if transcribe.is_set() or thread_terminate.is_set():
                    return (None, pyaudio.paComplete)
                data = wf.readframes(frame_count)
                return (data, pyaudio.paContinue if data else pyaudio.paComplete)

            while not response_queue.empty() and not transcribe.is_set():
                if not tts_queue.empty():
                    response = response_queue.get()
                    tts_queue.get()
                    wf.rewind()

                    stream = p.open(
                        format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        stream_callback=callback,
                    )
                    stream.start_stream()

                    while stream.is_active():
                        sleep(0.5)
                        if thread_terminate.is_set():
                            stream.stop_stream()
                            break
                    stream.stop_stream()
                    stream.close()

                    wf.close() # *after* playback is fully done

                    history += response
                    form_message_from_response(response)

        while True:
            sleep(1)
            if thread_terminate.is_set():
                p.terminate()
                break
            if transcribe.is_set():
                continue

            try:
                if not transcription_queue.empty():
                    user_input = transcription_queue.get(timeout=1)
                    form_message_from_transcription(user_input)

                    try:
                        response = requests.post(
                            "http://localhost:11434/api/chat",
                            headers={"Content-Type": "application/json"}, 
                            json={
                                "model": "gemma2:2b",
                            "messages": messages,
                            "stream": False,
                            },
                            timeout=60,
                        )
                        response.raise_for_status()
                        bot_reply = response.json()["message"]["content"]
                    except Exception as e:
                        print(f"Ollama error: {e}")
                        bot_reply = f"[Ollama error: {e}]"

                    # TTS with Piper
                    synthesize_with_piper(bot_reply)
                    tts_queue.put(1)
                    response_queue.put(bot_reply)

                    print("========== Ollama is answering ==========")
                    response_handler("")
                    transcription_queue.task_done()
                    response_queue.task_done()

            except Exception as e:
                print("Exception occurred:", e)
                continue

        print("Ollama thread terminated.")


    transcribe = threading.Event()
    thread_terminate = threading.Event()
    stop_listening = None
    audio_thread = None
    stop_listening = None

    stop_listening = read_mic(
        audio_queue, DEFAULT_MIC, ENERGY_THRESHOLD, record_timeout
    )

    consumer = threading.Thread(
        target=consumer_thread, args=(transcribe, thread_terminate)
    )
    consumer.start()

    try:
        while consumer.is_alive():
            sleep(3)
            continue

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt occurs, threads are running?", consumer.is_alive())
        
        stop_listening(wait_for_stop=False)
        thread_terminate.set()
        consumer.join()
        print()

    finally:
        print()
        while consumer.is_alive():
            print("Consumer thread is still running...")
            sleep(3)
        print("Exiting...")
    print("The program stops.")
    
    
if __name__ == "__main__":
    main()
