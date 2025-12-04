import queue
import threading
from dataclasses import dataclass
from typing import Optional
from collections import deque

import numpy as np
from faster_whisper import WhisperModel

from state import SharedState
import time  # CHANGED: for latency measurement


@dataclass
class ASRConfig:
    model_size: str = "tiny"       # "tiny", "base", etc
    device: str = "cpu"            # "cpu" or "cuda"
    compute_type: str = "int8"     # "int8", "int8_float16", etc
    sample_rate: int = 16000
    block_size: int = 320
    language: Optional[str] = "en"
    beam_size: int = 1             # keep small for latency
    pre_roll_ms: int = 200         # how much audio before voice_started to include


class WhisperASRWorker:
    """
    Reads audio frames from audio_q and VAD events from vad_events_q.
    Keeps a rolling pre-roll buffer of recent frames even when not listening.
    Buffers audio between voice_started and voice_ended (plus pre-roll).
    On voice_ended, runs Whisper and pushes a final transcript to asr_text_q.
    """

    def __init__(
        self,
        shared: SharedState,
        audio_q: queue.Queue,
        vad_events_q: queue.Queue,
        asr_text_q: queue.Queue,
        cfg: ASRConfig,
    ):
        self.shared = shared
        self.audio_q = audio_q
        self.vad_events_q = vad_events_q
        self.asr_text_q = asr_text_q
        self.cfg = cfg
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def join(self, timeout: float | None = None):
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _run(self):
        print("[ASR] Loading Whisper model...")
        model = WhisperModel(
            self.cfg.model_size,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
        )
        print("[ASR] Whisper model loaded")

        listening = False
        buffer_frames: list[np.ndarray] = []

        # Pre-roll buffer: last N frames before voice_started
        frame_duration_s = self.cfg.block_size / float(self.cfg.sample_rate)
        n_pre_roll_frames = max(
            1,
            int((self.cfg.pre_roll_ms / 1000.0) / frame_duration_s),
        )
        pre_roll = deque(maxlen=n_pre_roll_frames)

        while not self.shared.stop_event.is_set():
            # 1) Handle VAD events first (non-blocking)
            try:
                while True:
                    event = self.vad_events_q.get_nowait()
                    etype = event.get("type")
                    if etype == "voice_started":
                        listening = True
                        # start with pre-roll so we don't cut the beginning
                        buffer_frames = list(pre_roll)
                        print("[ASR] voice_started - begin buffering (with pre-roll)")
                    elif etype == "voice_ended":
                        print("[ASR] voice_ended - stop buffering and transcribe")
                        listening = False
                        if buffer_frames:
                            self._transcribe_and_emit(model, buffer_frames)
                        buffer_frames = []
            except queue.Empty:
                pass

            # 2) Consume audio frames
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # Always update pre-roll with latest frame
            pre_roll.append(frame)

            # If we are in an active utterance, also add to main buffer
            if listening:
                buffer_frames.append(frame)

        print("[ASR] Worker stopped")

    def _transcribe_and_emit(self, model: WhisperModel, frames: list[np.ndarray]):
        # Concatenate frames into single 1D float32 array
        audio = np.concatenate(frames, axis=0).astype("float32").flatten()

        if audio.size == 0:
            return

        duration = audio.shape[0] / self.cfg.sample_rate
        print(f"[ASR] Transcribing {duration:.2f} seconds of audio...")

        # CHANGED: measure ASR latency
        t0 = time.time()
        segments, info = model.transcribe(
            audio,
            beam_size=self.cfg.beam_size,
            language=self.cfg.language,
            vad_filter=False,  # we already do VAD outside
        )
        t1 = time.time()
        print(f"[Latency] ASR transcribe time: {(t1 - t0)*1000:.1f} ms")

        texts = [seg.text.strip() for seg in segments if seg.text]
        transcript = " ".join(texts).strip()

        if transcript:
            print(f"[ASR] Final transcript: {transcript}")
            # CHANGED: mark ASR-final time on shared state
            self.shared.mark_asr_final()
            self.asr_text_q.put(transcript)
        else:
            print("[ASR] No text recognized")
