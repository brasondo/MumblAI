from faster_whisper import WhisperModel
import os
class AudioTranscriber:
    def __init__(self, model_size="distil-large-v3"):
        device = "cuda"
        compute_type = "float16"
        print(f"Loading faster-whisper: {model_size} on {device} ({compute_type})")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str):
        segments, info = self.model.transcribe(
            audio_path,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
            word_timestamps=True,
            beam_size=1,
            temperature=0.2,
            condition_on_previous_text=False,
            initial_prompt="This is a rap song. Preserve slang, flow, and lyric structure."
        )
        labeled_segments, full = [], []
        for seg in segments:
            text = (seg.text or "").strip()
            if not text: continue
            full.append(text)
            d = {"start": float(seg.start or 0.0), "end": float(seg.end or 0.0), "text": text}
            d["label"] = "MUMBLE" if len(text)<3 else ("UNCERTAIN" if len(text)<5 else "CLEAR")
            labeled_segments.append(d)
        return " ".join(full).strip(), labeled_segments


    def _classify_segment(self, seg):
        text = seg.get("text","").strip()
        # 3-way tag: CLEAR / UNCERTAIN / MUMBLE
        if len(text) < 3:
            return "MUMBLE"
        if len(text) < 5:
            return "UNCERTAIN"
        return "CLEAR"