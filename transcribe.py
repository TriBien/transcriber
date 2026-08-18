from faster_whisper import WhisperModel
import sys
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python transcribe.py <audio-file>")
    sys.exit(1)

audio_file = Path(sys.argv[1])

if not audio_file.exists():
    print(f"File not found: {audio_file}")
    sys.exit(1)

print("Loading Whisper model...")

model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8",
)

print(f"Transcribing: {audio_file}")

segments, info = model.transcribe(
    str(audio_file),
    language="en",
    beam_size=1,
    vad_filter=True,
)

output_file = audio_file.with_suffix(".txt")

with output_file.open("w", encoding="utf-8") as f:
    for segment in segments:
        text = segment.text.strip()

        if text:
            # Capitalize first character
            text = text[0].upper() + text[1:]

            # Write REAL newline, not literal "\n"
            print(text)
            f.write(text + "\n")

print()
print(f"Transcript saved to: {output_file}")
