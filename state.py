from enum import Enum, auto
import threading
import time  # CHANGED: for latency timestamps


class BotState(Enum):
    IDLE = auto()
    LISTENING = auto()
    THINKING = auto()
    SPEAKING = auto()


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.state = BotState.IDLE
        self.barge_in = False
        self.stop_event = threading.Event()

        # CHANGED: latency timestamps and per turn flag
        self._last_asr_final_ts: float | None = None
        self._last_tts_start_ts: float | None = None
        self._tts_started_for_turn: bool = False

    def set_state(self, new_state: BotState, reason: str | None = None):
        with self._lock:
            old_state = self.state
            self.state = new_state

        if old_state is not new_state:
            if reason:
                print(f"[State] {old_state.name} -> {new_state.name}: {reason}")
            else:
                print(f"[State] {old_state.name} -> {new_state.name}")

    def get_state(self) -> BotState:
        with self._lock:
            return self.state

    def trigger_barge_in(self):
        with self._lock:
            self.barge_in = True

    def consume_barge_in(self) -> bool:
        with self._lock:
            val = self.barge_in
            self.barge_in = False
            return val

    # CHANGED: mark ASR final and reset per turn TTS flag
    def mark_asr_final(self):
        with self._lock:
            self._last_asr_final_ts = time.time()
            self._tts_started_for_turn = False

    # CHANGED: mark TTS start, but only log ASR -> sound once per turn
    def mark_tts_start(self):
        with self._lock:
            now = time.time()
            asr_ts = self._last_asr_final_ts
            already_started = self._tts_started_for_turn
            if not already_started:
                self._tts_started_for_turn = True
            self._last_tts_start_ts = now

        if (not already_started) and (asr_ts is not None):
            delta = now - asr_ts
            print(f"[Latency] ASR final -> first sound: {delta*1000:.1f} ms")
