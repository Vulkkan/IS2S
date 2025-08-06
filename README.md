# IS2S Architecture (Interruptible Speech to Speech Architecture)

┌────────┐     ┌────────────┐     ┌──────────────┐     ┌───────────┐
│ MIC IN │ ──▶ │ Whisper STT│ ──▶ │ Ollama Chat  │ ──▶ │ Piper TTS │
└────────┘     └────────────┘     └──────────────┘     └───────────┘
     ▲                                                |
     └─────── interruption detection loop ◀───────────┘
     
The IS2S (Interruptible Speech-to-Speech) architecture is a fully local, multithreaded speech interface pipeline designed for real-time, human-like conversational AI.
     
It orchestrates four independent but tightly coupled components — live microphone capture, ASR (Automatic Speech Recognition) via Whisper/FasterWhisper, LLM-based response generation via Ollama, and neural TTS (Text-to-Speech) via Piper. The architecture is driven by a non-blocking event loop, where voice input is streamed in background threads, queued into a rolling buffer, and dynamically transcribed once speech activity is detected or a pause threshold is crossed. This transcription triggers a task-specific LLM prompt, streamed to a local model, and fed into the TTS engine for synthesis. A key innovation in IS2S is interruptible synthesis — where input from the microphone continues during TTS playback, and if valid new speech is detected (based on voice activity and transcription confidence), the current output is gracefully interrupted, context is updated, and the assistant re-generates a new response. This allows for natural turn-taking, topic shifts, and mid-response redirection, closely mimicking human conversation. Thread-safe queues, timing controls, and stream callbacks ensure synchronized audio I/O and responsiveness, while avoiding concurrency pitfalls.

Fully offline, speech-to-speech conversational AI designed for real-time interaction. It listens, understands, and speaks — and can be interrupted mid-response just like a human. Perfect for building natural, dynamic AI experiences.


✨ Inspired By
Gemini live.


💪 Powered by: 
Ollama, Whisper, and Piper

    
🎯 Features
    🎤 Live microphone input
    🧠 Offline local LLM chat (via Ollama)
    🗣️ Realistic voice responses (via Piper TTS)
    🔄 Speech-to-speech loop
    🛑 Real-time interruption support
    ⚡ Fast, responsive, low-resource
    🧪 Built for experimentation and extensions
    

1. Piper Setup
Download piper models at:
https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/alba/medium

Download the files:
en_GB-alba-medium.onnx
en_GB-alba-medium.onnx.json

Place them in /models/piper in the project folder.

2. Install libraries:
pip install -r requirements.txt

3. Run the script.
The faster_whisper model would be downloaded automatically.

📦 Requirements

    Python 3.10+

    Ollama with a local model (e.g. smollm2:135m)

    Piper voice model (en_GB-alba-medium.onnx)

    Whisper C++ or Faster-Whisper

🧠 Notes

    Uses event-driven threading for non-blocking audio handling.

    Compatible with Linux (tested on Arch). Windows support pending.

    Future work: emotion detection, hotword activation, token-by-token TTS streaming.


✅ Next Steps

Add hotword activation

Add emotion/mood detection

Switch to streaming token-wise TTS

Multi-speaker recognition

Web-based interface
