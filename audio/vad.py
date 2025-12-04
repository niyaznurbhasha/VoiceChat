import time
import queue
import threading
from dataclasses import dataclass

import numpy as np

from state import SharedState


@dataclass
class VADConfig:
    # We no longer require you to hand-tune energy_threshold;
    # we will auto-calibrate from initial noise.
    min_speech_ms: int = 200
    silence_ms: int = 300
    sample_rate: int = 16000
    block_size: int = 320          # samples per frame
    calibration_ms: int = 800      # how long to listen to noise for baseline
    threshold_factor: float = 2.0  # how many times above noise to treat as speech


class VADWorker:
    """
    Acoustic VAD that auto-calibrates a baseline energy from initial audio
    and then emits voice_started / voice_ended events.

    This should "just work" in most environments without hand-tuning.
    """

    def __init__(self, shared: SharedState, audio_q: queue.Queue, vad_events_q: queue.Queue, cfg: VADConfig):
        self.shared = shared
        self.audio_q = audio_q
        self.vad_events_q = vad_events_q
        self.cfg = cfg
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def join(self, timeout: float | None = None):
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _run(self):
        print("[VAD] Worker started")

        frame_duration_s = self.cfg.block_size / float(self.cfg.sample_rate)
        min_speech_s = self.cfg.min_speech_ms / 1000.0
        silence_s = self.cfg.silence_ms / 1000.0

        # --- Phase 1: auto-calibrate noise baseline ---
        calibration_frames = int(self.cfg.calibration_ms / 1000.0 / frame_duration_s)
        baseline_energies = []

        print(f"[VAD] Calibrating noise for ~{self.cfg.calibration_ms} ms "
              f"({calibration_frames} frames)...")

        while not self.shared.stop_event.is_set() and len(baseline_energies) < calibration_frames:
            try:
                frame = self.audio_q.get(timeout=1.0)
            except queue.Empty:
                continue

            energy = float(np.mean(frame ** 2))

            baseline_energies.append(energy)

        if not baseline_energies:
            print("[VAD] No audio received during calibration, using fallback threshold.")
            baseline = 1e-7
        else:
            baseline = float(np.median(baseline_energies))

        energy_threshold = baseline * self.cfg.threshold_factor
        print(f"[VAD] Baseline energy ~{baseline:.8f}, "
              f"threshold ~{energy_threshold:.8f}")

        # --- Phase 2: actual VAD loop ---
        is_speaking = False
        speech_start_time = None
        last_voice_time = None

        while not self.shared.stop_event.is_set():
            try:
                frame = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                # handle trailing silence
                if is_speaking and last_voice_time is not None:
                    if time.time() - last_voice_time > silence_s:
                        self.vad_events_q.put({"type": "voice_ended", "ts": time.time()})
                        is_speaking = False
                        speech_start_time = None
                        last_voice_time = None
                continue

            energy = float(np.mean(frame ** 2))

            if energy > energy_threshold:
                # speech-ish frame
                if not is_speaking:
                    if speech_start_time is None:
                        speech_start_time = time.time()
                    else:
                        if time.time() - speech_start_time >= min_speech_s:
                            is_speaking = True
                            last_voice_time = time.time()
                            self.vad_events_q.put({"type": "voice_started", "ts": time.time()})
                            # CHANGED: signal potential barge-in for main loop
                            self.shared.trigger_barge_in()
                else:
                    last_voice_time = time.time()
            else:
                # silence frame
                if is_speaking and last_voice_time is not None:
                    if time.time() - last_voice_time > silence_s:
                        self.vad_events_q.put({"type": "voice_ended", "ts": time.time()})
                        is_speaking = False
                        speech_start_time = None
                        last_voice_time = None
                else:
                    # not speaking yet: reset potential start timer
                    speech_start_time = None

        print("[VAD] Worker stopped")
