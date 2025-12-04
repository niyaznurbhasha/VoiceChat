# Voice-to-Voice Real-Time AI Pipeline

This repository implements a **local, real-time, low-latency voice conversational agent** with full **barge-in**, **streaming ASR**, **streaming LLM**, and **sentence-level streaming TTS**.
All components run **on-device** with **no network hops**, resulting in extremely fast response times and natural interaction flow.

The system is designed as a reusable engine that can be plugged into larger agent frameworks (fitness coach, mental health assistant, productivity coach, etc).

---

## 1. Features at a Glance

### ✔ Real-time microphone → response loop

Pipeline processes audio in 20–25 ms frames, enabling fast ASR and instant barge-in detection.

### ✔ Streaming Whisper-based ASR

* VAD + Whisper small model
* partial speech detection
* turn segmentation
* millisecond-level transcription
* no online APIs

### ✔ Token-streaming LLM

* Local Mistral/Llama models via `llama.cpp`
* streamed token output
* sentence extraction on-the-fly
* barge-in cancel for mid-generation interruption
* extremely short prompt to minimize latency

### ✔ Streaming Piper TTS

* Low-latency Piper model
* Sentence-level synthesis
* Overlaps TTS synthesis with LLM generation to reduce gaps
* Full barge-in cancellation mid-audio
* Configurable voice settings, speaker ID, length scale, volume

### ✔ Full barge-in

User speech immediately interrupts the bot while it is:

* Speaking (TTS)
* Thinking (LLM generating)

Barge-in is triggered **directly from VAD**, not after ASR, so interruption is immediate.

### ✔ True concurrent architecture

All components run independently:

* Audio input thread
* VAD worker
* ASR worker
* LLM worker
* TTS worker
* Playback thread
* Main routing/event loop

This allows overlapping ASR, LLM, TTS, and playback to achieve lowest possible latency.

---

## 2. Setup and Installation

### Install dependencies

```bash
pip install -r requirements.txt
```
Windows Audio Note

If you're running this on Windows, run the pipeline from an Anaconda Prompt (or any terminal that exposes the correct audio devices).
The sounddevice library occasionally fails to open the microphone when launched from PowerShell, VSCode terminal, or other shells on Windows.

If you still get audio errors, verify:

Your input/output audio devices are enabled

The default input device is selected in Windows Sound Settings

### Configure environment

Edit `.env` or rely on defaults:

* `LLM_PROVIDER` (`ollama` or local llama.cpp)
* `OLLAMA_MODEL`
* audio sample rate
* block size
* Whisper ASR config
* VAD config
* TTS config
* LLM config

### Download models

#### Whisper ASR

Download your chosen Whisper `small`/`medium` model into `models/asr/`.

#### LLM

Place your `gguf` quantized model in:

```
models/llm/your_model.gguf
```

#### Piper TTS

Place your Piper `.onnx` or `.pt` voice model in:

```
models/tts/en_US-x.onnx
```

### Run the pipeline

```bash
python main.py
```

You should see:

* LLM loaded
* TTS worker online
* “Listening...”

Speak normally and the agent will respond.

---

## 3. Architecture Overview

```
Microphone → AudioInput → [VAD, ASR]
                     → VAD events
                     → ASR transcripts
ASR final text → LLM worker (streaming tokens)
LLM sentences → TTS worker (Piper synth)
TTS audio → playback → speaker
```

### Thread responsibilities

**AudioInput**
Pushes raw audio frames into two queues: VAD & ASR.

**VADWorker**
Detects start/end of speech + barge-in (voice_start while bot is speaking).

**ASRWorker (Whisper)**
Buffers frames between VAD start/end, transcribes chunk, emits text.

**LLMWorker**
Converts each complete user utterance into streamed tokens.
Stops immediately when `stop_llm_event` is set.

**Main loop**
Routes all events, handles state machine, sentence segmentation, and barge-in logic.

**PiperTTSWorker**
Synthesizes sentences asynchronously, plays them in a dedicated playback thread.
Stops immediately when `stop_tts_event` is set.

---

## 4. Key Design Decisions & Why

### 4.1 No network hops

All components run locally to avoid ~80–300 ms round-trip latency found in cloud ASR/LLM/TTS.

### 4.2 Streaming everything

* ASR streaming
* LLM token streaming
* TTS sentence streaming

This produces:

* instant first response
* overlapping generation/synthesis
* natural conversational pacing

### 4.3 True VAD-based barge-in

We use **voice-start** from VAD (not ASR transcripts) to interrupt:

* LLM
* TTS
  immediately.

This makes the system feel human-like and responsive.

### 4.4 Sentence-level slicing

We accumulate tokens until `. ? !` and send each completed sentence to TTS.
This ensures:

* low delay
* smoother follow-ups
* continuous flow

### 4.5 Configurable through `config.py`

ASR/VAD/LLM/TTS parameters driven from one file.

### 4.6 Low-latency tuning

* Short system prompts
* Low-temperature and short max-token outputs
* Whisper small or smaller
* Piper fast models
* Q4_K_M or Q4_K_S quantized LLM
* Simple state machine

### 4.7 Multi-threaded pipeline

Each component runs fully independently, allowing overlap and minimal blocking.

---

## 5. Current Latency Profile (expected)

| Stage                              | Typical Latency                  |
| ---------------------------------- | -------------------------------- |
| VAD start detection                | 20–40 ms                         |
| Whisper ASR final chunk            | 5–15 ms                          |
| LLM time-to-first-token            | 800–2500 ms depending on CPU/GPU |
| Piper TTS synthesis (per sentence) | 300–1500 ms                      |
| Audio playback start               | ~10 ms                           |

Combined:
**~2.5–4.5 seconds** from end of user speech → first spoken audio.

With small models + GPU offload, this can get much lower.

---

## 6. Future Work

### ✔ Streaming TTS (frame-level)

Instead of synthesizing full sentences:

* stream audio frames to playback in real time
* reduce sentence 1 → 2 gaps
* cut time-to-first-sound in half

### ✔ GPU Whisper or faster ASR

* Whisper tiny/int8
* Faster VAD
* Reduce ASR chunk size

### ✔ Even smaller LLM

Use:

* Mistral-7B-Instruct-Q4_K_M (current)
  → switch to
* Phi-2 (2.7B)
* Llama-3.1-Instruct-8B with GPU
* Qwen2.5-3B-Chat

### ✔ Real-time semantic VAD

Prevent cutting:

* mid-clause
* mid-thought
* too-short utterances

### ✔ Unified memory layer

Add short-term memory and summarization in background threads.
Feed summary + last N turns to LLM for long conversations.

### ✔ Multi-provider LLM wrapper

Plug in:

* local llama.cpp
* vLLM
* OpenAI
* Mistral API
* Groq for fast cold-start token generation

### ✔ Mobile deployment

Use Pipecat or a reduced pipeline:

* tiny ASR
* tiny LLM
* tiny TTS
* C++ backend
* iOS CoreML conversion for all models
* Android NNAPI for acceleration

