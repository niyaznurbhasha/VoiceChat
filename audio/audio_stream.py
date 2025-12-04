import time
import queue
import threading

import sounddevice as sd

from state import SharedState
from config import SAMPLE_RATE, BLOCK_SIZE


class AudioInput:
    """Long-lived microphone input that pushes frames into one or more queues."""

    def __init__(self, shared: SharedState, audio_queues: list[queue.Queue]):
        self.shared = shared
        self.audio_queues = audio_queues
        self.thread = None

    def _callback(self, indata, frames, time_info, status):
        if self.shared.stop_event.is_set():
            raise sd.CallbackAbort
        frame = indata.copy()
        for q in self.audio_queues:
            try:
                q.put_nowait(frame)
            except queue.Full:
                # Drop frame for this consumer if its queue is full
                pass

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
