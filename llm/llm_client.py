# llm/llm_client.py

import queue
import threading
from typing import Optional

from llama_cpp import Llama

from state import SharedState
import time  # CHANGED: for latency logging


class LLMWorker:
    """
    Streams tokens from a local GGUF model via llama-cpp-python.
    Reads user text from llm_in_q, outputs tokens to llm_out_q.
    Can be interrupted via stop_llm_event.
    """

    def __init__(
        self,
        shared: SharedState,
        llm_in_q: queue.Queue,
        llm_out_q: queue.Queue,
        stop_llm_event: threading.Event,
        model_path: str = "models/mistral-7b-instruct.Q4_K_M.gguf",
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 0,  # set >0 later if you enable GPU offload
    ):
        self.shared = shared
        self.llm_in_q = llm_in_q
        self.llm_out_q = llm_out_q
        self.stop_llm_event = stop_llm_event

        print("[LLM] Loading GGUF model...")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            embedding=False,
        )
        print("[LLM] Model loaded")

        self.thread: Optional[threading.Thread] = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def join(self, timeout: Optional[float] = None):
        if self.thread is not None:
            self.thread.join(timeout=timeout)

    def _run(self):
        while not self.shared.stop_event.is_set():
            try:
                user_text = self.llm_in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # allow fresh generation for this request
            self.stop_llm_event.clear()

            # CHANGED: shorter, less verbose system prompt for lower latency
            prompt = (
                "You are a kind, concise assistant in a voice conversation.\n"
                "Respond naturally and briefly.\n\n"
                f"User: {user_text}\n"
                "Assistant:"
            )

            try:
                # CHANGED: measure time-to-first-token and use smaller max_tokens
                t_start = time.time()
                first_token_emitted = False

                stream = self.llm(
                    prompt,
                    max_tokens=160,          # CHANGED: lower cap for faster replies
                    temperature=0.3,
                    stop=["User:"],
                    stream=True,
                )

                for chunk in stream:
                    if self.stop_llm_event.is_set() or self.shared.stop_event.is_set():
                        break

                    token = chunk["choices"][0]["text"]
                    if not token:
                        continue

                    if not first_token_emitted:
                        first_token_emitted = True
                        t_first = time.time()
                        print(
                            f"[Latency] LLM time to first token: {(t_first - t_start)*1000:.1f} ms"
                        )

                    self.llm_out_q.put(token)

                # mark end of response
                self.llm_out_q.put("[LLM_END]")

            except Exception as e:
                print("[LLM] Error:", e)
                continue
