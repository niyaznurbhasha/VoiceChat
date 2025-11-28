import time
import queue
import threading

import sounddevice as sd

from state import SharedState
from config import SAMPLE_RATE, BLOCK_SIZE


class AudioInput:
    """Long-lived microphone input that pushes frames into a queue."""

    def __init__(self, shared: SharedState, audio_q: queue.Queue):
        self.shared = shared
        self.audio_q = audio_q
        self.stream = None
        self.thread = None

    def _callback(self, indata, frames, time_info, status):
        if self.shared.stop_event.is_set():
            raise sd.CallbackAbort
        # indata is (frames, channels)
        self.audio_q.put(indata.copy())

    def start(self):
        def run():
            print("[Audio] Starting input stream")
            with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                dtype="float32",
                callback=self._callback,
            ):
                while not self.shared.stop_event.is_set():
                    time.sleep(0.05)
            print("[Audio] Input stream stopped")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self, timeout: float | None = None):
        if self.thread is not None:
            self.thread.join(timeout=timeout)
