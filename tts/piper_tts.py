import threading
import queue
import subprocess
import tempfile
import time
import wave
import traceback

import simpleaudio as sa

from state import SharedState, BotState
from config import DEBUG  # CHANGED: use debug flag


class PiperTTSWorker(threading.Thread):
    """
    TTS worker using Piper.

    CHANGED:
      - This thread now only synthesizes audio (Piper calls) and pushes
        WAV paths into an internal playback queue.
      - A separate playback thread pulls WAVs and plays them.
      - While one sentence is playing, the main TTS thread can synthesize
        the next one, which reduces gaps between sentences.
    """

    def __init__(
        self,
        shared: SharedState,
        tts_text_q: queue.Queue,
        stop_tts_event: threading.Event,
        piper_exe: str,
        piper_model: str,
        speaker_id: int = 0,
        length_scale: float = 1.0,
        noise_scale: float = 0.667,
        volume: float = 1.0,  # not applied yet, simpleaudio plays raw
    ):
        super().__init__(daemon=True)
        self.shared = shared
        self.tts_text_q = tts_text_q
        self.stop_tts_event = stop_tts_event
        self.piper_exe = piper_exe
        self.piper_model = piper_model
        self.speaker_id = speaker_id
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.volume = volume

        # CHANGED: internal queue of ready WAV files for playback
        self.play_q: queue.Queue[str] = queue.Queue(maxsize=16)
        self._play_thread: threading.Thread | None = None

    def _log(self, msg: str) -> None:
        if DEBUG:
            print(msg)

    def run(self) -> None:
        self._log("[TTS] Piper worker started")

        # CHANGED: start dedicated playback thread
        self._play_thread = threading.Thread(
            target=self._play_loop, daemon=True
        )
        self._play_thread.start()

        # Synth loop: text -> wav_path -> play_q
        while not self.shared.stop_event.is_set():
            try:
                text = self.tts_text_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if not text:
                continue

            self._log(f"\n[TTS] Synth enqueue (len={len(text)}): {text[:80]!r}\n")

            # New utterance chunk, allow playback to run
            # Note: stop_tts_event will be set by barge in
            # and checked in playback loop for early stop.
            try:
                wav_path = self._synthesize_with_piper(text)
                if wav_path is None:
                    continue

                # If barge in happened during synth, drop this audio
                if self.stop_tts_event.is_set() or self.shared.stop_event.is_set():
                    self._log("[TTS] Dropping synthesized audio due to barge-in or shutdown")
                    continue

                # Put synthesized audio into playback queue
                try:
                    self.play_q.put_nowait(wav_path)
                except queue.Full:
                    self._log("[TTS] play_q full, dropping audio")
            except Exception as e:
                print(f"[TTS] Error during synth: {e}")
                traceback.print_exc()

        self._log("[TTS] Piper worker exiting (synth loop)")

    def _synthesize_with_piper(self, text: str) -> str | None:
        """Call Piper CLI to synthesize a WAV file for the given text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        cmd = [
            self.piper_exe,
            "--model",
            self.piper_model,
            "--output_file",
            out_path,
        ]

        if self.speaker_id is not None:
            cmd.extend(["--speaker", str(self.speaker_id)])

        if self.length_scale != 1.0:
            cmd.extend(["--length_scale", str(self.length_scale)])
        if self.noise_scale != 0.667:
            cmd.extend(["--noise_scale", str(self.noise_scale)])

        t0 = time.time()
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        t1 = time.time()
        print(f"[Latency] TTS synth time (Piper): {(t1 - t0)*1000:.1f} ms")

        if proc.returncode != 0:
            print("[TTS] Piper failed:")
            print(proc.stderr.decode("utf-8", errors="ignore"))
            return None

        return out_path

    # CHANGED: new playback loop
    def _play_loop(self) -> None:
        """
        Continuously play WAV files from play_q.
        Stops early if stop_tts_event is set (barge in) and drains the queue.
        """
        self._log("[TTS] Playback thread started")
        while not self.shared.stop_event.is_set():
            try:
                wav_path = self.play_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # If we were asked to stop, drain and skip playback
            if self.stop_tts_event.is_set():
                self._log("[TTS] Barge-in before playback, draining play_q")
                self._drain_play_queue()
                continue

            try:
                with wave.open(wav_path, "rb") as wf:
                    wave_obj = sa.WaveObject.from_wave_read(wf)
            except Exception as e:
                print(f"[TTS] Error opening WAV: {e}")
                continue

            # Mark first-sentence playback start for this turn
            self.shared.mark_tts_start()

            # Change state only when we actually start making sound
            if self.shared.get_state() != BotState.SPEAKING:
                self.shared.set_state(BotState.SPEAKING, reason="TTS started speaking")

            play_obj = wave_obj.play()

            # Tight loop to watch for barge-in
            while play_obj.is_playing():
                if self.stop_tts_event.is_set() or self.shared.stop_event.is_set():
                    self._log("[TTS] Stopping playback due to barge-in or shutdown")
                    play_obj.stop()
                    break
                time.sleep(0.005)

            # If barge-in happened, drain pending audio and let main loop
            # drive state back to LISTENING via VAD barge-in.
            if self.stop_tts_event.is_set():
                self._drain_play_queue()
                # Important: do not override state here, main already set LISTENING
                continue

            # Natural finish: if both queues are empty, return to LISTENING
            if self.play_q.empty() and self.tts_text_q.empty():
                self.shared.set_state(
                    BotState.LISTENING,
                    reason="TTS finished or stopped",
                )

        self._log("[TTS] Playback thread exiting")

    def _drain_play_queue(self) -> None:
        """Drain any remaining WAV paths after barge-in."""
        try:
            while True:
                self.play_q.get_nowait()
        except queue.Empty:
            pass

    def stop(self) -> None:
        # CHANGED: signal both TTS and playback to stop
        self.stop_tts_event.set()
