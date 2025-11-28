import queue

from state import SharedState, BotState
from audio.audio_stream import AudioInput
from audio.vad import VADWorker, VADConfig


def main():
    shared = SharedState()

    audio_q: queue.Queue = queue.Queue(maxsize=100)
    vad_events_q: queue.Queue = queue.Queue(maxsize=100)

    # Audio input
    audio_input = AudioInput(shared=shared, audio_q=audio_q)
    audio_input.start()

    # VAD worker

    vad_cfg = VADConfig(
        sample_rate=16000,
        block_size=320,
        min_speech_ms=200,
        silence_ms=400,
        calibration_ms=800,
        threshold_factor=2.0,
    )
    vad_worker = VADWorker(shared=shared, audio_q=audio_q, vad_events_q=vad_events_q, cfg=vad_cfg)

    vad_worker.start()

    print("Listening... Press Ctrl+C to stop.")

    try:
        shared.set_state(BotState.LISTENING)
        while True:
            event = vad_events_q.get()
            etype = event["type"]
            if etype == "voice_started":
                print("[Main] VAD: voice_started")
                # later: if state in {THINKING, SPEAKING}, trigger barge-in
            elif etype == "voice_ended":
                print("[Main] VAD: voice_ended")
                # later: use this plus ASR partials to finalize utterance
    except KeyboardInterrupt:
        print("\n[Main] Shutting down...")
        shared.stop_event.set()
        audio_input.join(timeout=2.0)
        vad_worker.join(timeout=2.0)
        print("[Main] Done.")


if __name__ == "__main__":
    main()
