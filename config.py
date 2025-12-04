import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_0")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Audio config
SAMPLE_RATE = 16000
BLOCK_SIZE = 320  # 20 ms at 16kHz
# Debug logging
DEBUG = True

# VAD settings
VAD_CFG = dict(
    sample_rate=SAMPLE_RATE,
    block_size=BLOCK_SIZE,
    min_speech_ms=200,
    silence_ms=800,
    calibration_ms=800,
    threshold_factor=8.0,
)

# ASR (faster whisper) settings
ASR_CFG = dict(
    model_size="tiny",     # "tiny", "base", "small", "medium"
    device="cpu",          # "cpu" or "cuda"
    compute_type="int8",   # "int8", "float16", etc
    sample_rate=SAMPLE_RATE,
    block_size=BLOCK_SIZE,
    language="en",
    beam_size=1,
    pre_roll_ms=200,
    # optional extras you can use later
    # chunk_ms=1500,
    # word_timestamps=False,
)

# Local LLM (llama cpp) settings
LLM_CFG = dict(
    model_path="models/llm/mistral-7b-instruct-v0.3-q4_k_m.gguf",
    max_tokens=256,        # cap reply length
    temperature=0.7,       # 0.3 to 0.9 is a good range
    top_p=0.9,             # nucleus sampling
    repeat_penalty=1.1,    # >1 helps avoid loops
    n_threads=8,           # tune based on your CPU
    n_gpu_layers=0,        # set >0 if you offload to GPU
)

# Piper TTS settings
TTS_CFG = dict(
    piper_exe="models/tts/piper.exe",
    piper_model="models/tts/en_US-norman-medium.onnx",  # your chosen voice

    # voice control
    speaker_id=0,        # for multi speaker models
    length_scale=1.0,    # <1.0 faster, >1.0 slower speech
    noise_scale=0.667,   # more or less expressiveness
    volume=1.0,          # you can apply this in playback
)
