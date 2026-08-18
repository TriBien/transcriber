import tempfile
from pathlib import Path

import streamlit as st
from faster_whisper import WhisperModel

MODELS = ["tiny.en", "base.en", "tiny", "base"]
LANGUAGES = {
    "English": "en",
    "Vietnamese": "vi"
}

st.set_page_config(page_title="Whisper Transcriber", page_icon="🎙️")
st.title("🎙️ Whisper Transcriber")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Whisper Model", MODELS, index=0)
    language_name = st.selectbox("Language", list(LANGUAGES.keys()), index=0)
    language_code = LANGUAGES[language_name]

st.divider()

audio_file = st.file_uploader(
    "Upload audio file",
    type=["mp3", "wav", "m4a", "ogg", "flac", "webm"],
)

if audio_file and st.button("Transcribe", type="primary"):
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(audio_file.name).suffix
    ) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        with st.status("Transcribing...", expanded=True) as status:
            st.write(f"Loading model: **{model_name}**")
            model = WhisperModel(model_name, device="cpu", compute_type="int8")

            st.write(f"Transcribing **{audio_file.name}** ({language_name})")
            segments, info = model.transcribe(
                tmp_path,
                language=language_code,
                beam_size=1,
                vad_filter=True,
            )

            transcript_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    text = text[0].upper() + text[1:]
                    transcript_parts.append(text)

            status.update(label="Transcription complete!", state="complete")

        transcript = "\n".join(transcript_parts)
        st.subheader("Transcript")
        st.text_area("Result", transcript, height=300)

        st.download_button(
            label="Download Transcript",
            data=transcript,
            file_name=Path(audio_file.name).with_suffix(".txt").name,
            mime="text/plain",
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
