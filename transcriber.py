import whisper

class AudioTranscriber:
    def __init__(self, model_size="large-v3"):
        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> tuple[str, list[dict]]:
        print(f"Transcribing: {audio_path}")
        result = self.model.transcribe(
        audio_path,
        word_timestamps=True,
        condition_on_previous_text=False,
        temperature=0.4,
        initial_prompt="This is a rap song. Preserve slang, flow, and lyric structure."
    )

        full_text = result.get("text", "").strip()
        segments = result.get("segments", [])

        labeled_segments = []
        for seg in segments:
            label = self._classify_segment(seg)
            labeled_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
                "label": label
            })

        return full_text, labeled_segments

    def _classify_segment(self, seg):
        # Heuristic: low confidence or short text = mumble
        if seg.get("no_speech_prob", 0) > 0.5:
            return "MUMBLE"
        if len(seg.get("text", "").strip()) < 5:
            return "MUMBLE"
        return "CLEAR"
