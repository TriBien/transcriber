from faster_whisper import WhisperModel
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac", ".ogg", ".oga", ".m4a", ".opus", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v", ".ts", ".mts", ".m2ts"}


def extract_audio(video_file: Path) -> Path:
    audio_file = video_file.with_suffix(".wav")

    print(f"Extracting audio from: {video_file}")

    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(video_file),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_file),
        ],
        check=True,
    )

    return audio_file


if len(sys.argv) != 2:
    print("Usage: python transcribe.py <audio-or-video-file>")
    sys.exit(1)

input_file = Path(sys.argv[1])

if not input_file.exists():
    print(f"File not found: {input_file}")
    sys.exit(1)

file_ext = input_file.suffix.lower()

if file_ext in AUDIO_EXTENSIONS:
    transcribe_file = input_file
elif file_ext in VIDEO_EXTENSIONS:
    transcribe_file = extract_audio(input_file)
else:
    print(f"Unsupported file type: {input_file} (supported: {', '.join(sorted(AUDIO_EXTENSIONS | VIDEO_EXTENSIONS))})")
    sys.exit(1)

print("Loading Whisper model...")

model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8",
)

print(f"Transcribing: {transcribe_file}")

segments, info = model.transcribe(
    str(transcribe_file),
    language="en",
    beam_size=1,
    vad_filter=True,
)

output_file = input_file.with_suffix(".txt")

with output_file.open("w", encoding="utf-8") as f:
    for segment in segments:
        text = segment.text.strip()

        if text:
            # Capitalize first character
            text = text[0].upper() + text[1:]

            # Write REAL newline, not literal "\n"
            print(text)
            f.write(text + "\n")

# Clean up extracted audio
if transcribe_file != input_file:
    transcribe_file.unlink(missing_ok=True)

print()
print(f"Transcript saved to: {output_file}")