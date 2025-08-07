from transcriber import AudioTranscriber

if __name__ == "__main__":
    transcriber = AudioTranscriber(model_size="large-v3")
    full_text, segments = transcriber.transcribe("sbt.mp3")  # Replace with your actual file

    # ✅ Save output for evaluation
    with open("whisper_output.txt", "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(seg["text"].strip() + "\n")

    print("\n--- FULL TRANSCRIPTION ---\n")
    print(full_text)

    print("\n--- LABELED SEGMENTS ---\n")
    for seg in segments:
        print(f"[{seg['label']}] [{seg['start']:.2f}–{seg['end']:.2f}]: {seg['text']}")

    print("\n--- MUMBLE SEGMENTS (for GPT input) ---\n")
    mumbles = [seg['text'] for seg in segments if seg['label'] == "MUMBLE"]
    for i, m in enumerate(mumbles, 1):
        print(f"{i}. \"{m}\"")




