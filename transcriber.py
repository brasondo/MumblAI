# transcriber.py
from faster_whisper import WhisperModel

class AudioTranscriber:
    def __init__(self, model_size="distil-large-v3"):
        device = "cuda"
        compute_type = "float16"
        print(f"Loading faster-whisper: {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def _classify_word(self, word: str, prob: float | None) -> str:
        if prob is not None:
            if prob < 0.50:  return "MUMBLE"
            if prob < 0.80:  return "UNCERTAIN"
            return "CLEAR"
        # Fallback: tiny words are less reliable
        w = (word or "").strip()
        if len(w) < 3: return "MUMBLE"
        if len(w) < 5: return "UNCERTAIN"
        return "CLEAR"

    def transcribe(self, audio_path: str):
        segments, info = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,      # you already have this
            beam_size=1,
            temperature=0.2,
            condition_on_previous_text=False,
            initial_prompt="This is a rap song. Preserve slang, flow, and lyric structure."
        )

        tokens = []
        full_text_parts = []

        for seg in segments:
            # Prefer true word-level timings
            if getattr(seg, "words", None):
                for w in seg.words:
                    word = (w.word or "").strip()
                    if not word:
                        continue
                    label = self._classify_word(word, getattr(w, "probability", None))
                    tokens.append({
                        "start": float(getattr(w, "start", seg.start) or 0.0),
                        "end":   float(getattr(w, "end", seg.end) or 0.0),
                        "text":  word,
                        "label": label,
                    })
                    full_text_parts.append(word)
            else:
                # Fallback: segment becomes a single token (rare)
                text = (seg.text or "").strip()
                if not text:
                    continue
                label = self._classify_word(text, None)
                tokens.append({
                    "start": float(seg.start or 0.0),
                    "end":   float(seg.end or 0.0),
                    "text":  text,
                    "label": label,
                })
                full_text_parts.append(text)

        return " ".join(full_text_parts).strip(), tokens
