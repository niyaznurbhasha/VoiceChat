import queue
import time
import threading

from state import SharedState, BotState
from audio.audio_stream import AudioInput
from audio.vad import VADWorker, VADConfig
from asr.whisper_asr import WhisperASRWorker, ASRConfig
from llm.llm_client import LLMWorker
from tts.piper_tts import PiperTTSWorker  # CHANGED: TTS worker
from config import VAD_CFG, ASR_CFG, LLM_CFG, TTS_CFG, DEBUG  # CHANGED: config-driven


def log(msg: str) -> None:
    if DEBUG:
        print(msg)


def main():
    shared = SharedState()

    # -------------------------
    # Queues for components
    # -------------------------
    audio_q_vad = queue.Queue(maxsize=200)
    audio_q_asr = queue.Queue(maxsize=200)
    vad_events_q = queue.Queue(maxsize=200)
    asr_text_q = queue.Queue(maxsize=50)

    llm_in_q = queue.Queue(maxsize=10)
    llm_out_q = queue.Queue(maxsize=200)

    tts_text_q = queue.Queue(maxsize=50)  # CHANGED: TTS text queue

    # Events: stop LLM + stop TTS (for barge-in)
    stop_llm_event = threading.Event()
    stop_tts_event = threading.Event()

    # -------------------------
    # Audio input (pushes to 2 queues)
    # -------------------------
    audio_input = AudioInput(
        shared=shared,
        audio_queues=[audio_q_vad, audio_q_asr],
    )
    audio_input.start()

    # -------------------------
    # VAD
    # -------------------------
    vad_cfg = VADConfig(**VAD_CFG)  # CHANGED

    vad_worker = VADWorker(
        shared=shared,
        audio_q=audio_q_vad,
        vad_events_q=vad_events_q,
        cfg=vad_cfg,
    )
    vad_worker.start()

    # -------------------------
    # ASR
    # -------------------------
    asr_cfg = ASRConfig(**ASR_CFG)  # CHANGED

    asr_worker = WhisperASRWorker(
        shared=shared,
        audio_q=audio_q_asr,
        vad_events_q=vad_events_q,
        asr_text_q=asr_text_q,
        cfg=asr_cfg,
    )
    asr_worker.start()

    # -------------------------
    # LLM worker
    # -------------------------
    llm_worker = LLMWorker(
        shared=shared,
        llm_in_q=llm_in_q,
        llm_out_q=llm_out_q,
        stop_llm_event=stop_llm_event,
        model_path=LLM_CFG["model_path"],
        # You can later pipe in max_tokens, temperature, etc from LLM_CFG
    )
    llm_worker.start()

    # -------------------------
    # TTS worker (Piper)
    # -------------------------
    tts_worker = PiperTTSWorker(
        shared=shared,
        tts_text_q=tts_text_q,
        stop_tts_event=stop_tts_event,
        piper_exe=TTS_CFG["piper_exe"],
        piper_model=TTS_CFG["piper_model"],
        speaker_id=TTS_CFG.get("speaker_id", 0),
        length_scale=TTS_CFG.get("length_scale", 1.0),
        noise_scale=TTS_CFG.get("noise_scale", 0.667),
        volume=TTS_CFG.get("volume", 1.0),
    )
    tts_worker.start()

    shared.set_state(BotState.LISTENING, reason="Voice loop started")
    print("Listening... Press Ctrl+C to stop.")

    # buffers for sentence-level LLM → TTS streaming
    full_response = ""      # entire reply
    sentence_buffer = ""    # current sentence being built

    try:
        while not shared.stop_event.is_set():

            # -------------------------
            # TRUE BARGE-IN from VAD (instant)
            # -------------------------
            if shared.consume_barge_in():
                state = shared.get_state()
                if state in (BotState.THINKING, BotState.SPEAKING):
                    log("[Main] VAD barge-in: user started speaking, canceling LLM/TTS")

                    stop_llm_event.set()
                    stop_tts_event.set()

                    # reset LLM streaming buffers
                    full_response = ""
                    sentence_buffer = ""

                    # drain leftover LLM tokens and TTS text
                    try:
                        while True:
                            llm_out_q.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        while True:
                            tts_text_q.get_nowait()
                    except queue.Empty:
                        pass

                    shared.set_state(BotState.LISTENING, reason="VAD barge-in")

            # -------------------------
            # Drain ASR transcripts (turn-level)
            # -------------------------
            try:
                while True:
                    user_text = asr_text_q.get_nowait()
                    log(f"[Main] User utterance: {user_text}")

                    # new turn: reset per-turn buffers and allow LLM/TTS again
                    full_response = ""          # CHANGED: clear previous reply
                    sentence_buffer = ""        # CHANGED: clear current sentence
                    stop_llm_event.clear()
                    stop_tts_event.clear()      # CHANGED: re-enable TTS after barge-in

                    # Start LLM THINKING for this final transcript
                    shared.set_state(BotState.THINKING, reason="Final ASR transcript")
                    llm_in_q.put(user_text)

            except queue.Empty:
                pass

            # -------------------------
            # Drain LLM tokens (sentence-level streaming to TTS)
            # -------------------------
            try:
                while True:
                    tok = llm_out_q.get_nowait()

                    # sentinel from LLM worker
                    if tok == "[LLM_END]":
                        log("\n[Main] LLM finished.\n")

                        # Flush any leftover partial sentence
                        leftover = sentence_buffer.strip()
                        if leftover:
                            log(
                                f"[Main] Send FINAL chunk to TTS "
                                f"(len={len(leftover)}): {leftover[:80]!r}"
                            )
                            tts_text_q.put(leftover)

                        log(f"[Main] Full response len={len(full_response)}")

                        full_response = ""
                        sentence_buffer = ""
                        continue

                    # accumulate full reply and current sentence
                    full_response += tok
                    sentence_buffer += tok

                    # stream tokens to console
                    print(tok, end="", flush=True)

                    # sentence-level streaming: send each finished sentence to TTS
                    if any(c in ".?!" for c in tok):
                        sentence = sentence_buffer.strip()
                        if sentence:
                            log(
                                f"[Main] Send SENTENCE to TTS "
                                f"(len={len(sentence)}): {sentence[:80]!r}"
                            )
                            tts_text_q.put(sentence)

                            # CHANGED: we NO LONGER set SPEAKING here.
                            # TTS playback thread is responsible for:
                            # - setting SPEAKING when audio actually starts
                            # - setting LISTENING when playback is done.

                        sentence_buffer = ""

            except queue.Empty:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down...")
        shared.stop_event.set()

    finally:
        audio_input.join(timeout=1.0)
        vad_worker.join(timeout=1.0)
        asr_worker.join(timeout=1.0)
        llm_worker.join(timeout=1.0)
        tts_worker.join(timeout=1.0)
        print("[Main] Exit.")


if __name__ == "__main__":
    main()
