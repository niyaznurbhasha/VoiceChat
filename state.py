from enum import Enum, auto
import threading


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

    def set_state(self, new_state: BotState):
        with self._lock:
            self.state = new_state

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
