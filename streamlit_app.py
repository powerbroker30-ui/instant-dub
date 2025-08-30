import streamlit as st
import tempfile, os, uuid, io
from faster_whisper import WhisperModel
from transformers import pipeline
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip

st.set_page_config(page_title="Instant Dub (MVP)", page_icon="🎬", layout="centered")

SUPPORTED_TARGETS = {
    "hi": {"label": "Hindi", "translator": "Helsinki-NLP/opus-mt-en-hi"},
    "es": {"label": "Spanish", "translator": "Helsinki-NLP/opus-mt-en-es"},
    "fr": {"label": "French", "translator": "Helsinki-NLP/opus-mt-en-fr"},
    "de": {"label": "German", "translator": "Helsinki-NLP/opus-mt-en-de"},
    "ta": {"label": "Tamil", "translator": "Helsinki-NLP/opus-mt-en-ta"},
}

@st.cache_resource(show_spinner=False)
def load_asr():
    # 'tiny' keeps it light for free CPU tiers; int8 quantization for speed
    return WhisperModel("tiny", device="cpu", compute_type="int8")

@st.cache_resource(show_spinner=False)
def get_translator(lang_code: str):
    model_name = SUPPORTED_TARGETS[lang_code]["translator"]
    return pipeline("translation", model=model_name)

def transcribe_english(wav_path: str) -> str:
    model = load_asr()
    segments, info = model.transcribe(wav_path, language="en")
    text = " ".join(seg.text.strip() for seg in segments).strip()
    if not text:
        raise RuntimeError("Transcription produced empty text.")
    return text

def translate_text(text: str, target_lang: str) -> str:
    tr = get_translator(target_lang)
    # simple chunking for long text
    max_len = 4000
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)] or [""]
    out = []
    for c in chunks:
        out.append(tr(c)[0]["translation_text"])
    return " ".join(out)

def synthesize_tts(text: str, lang_code: str, out_path: str):
    tts = gTTS(text=text, lang=lang_code)
    tts.save(out_path)

def mux_audio(video_path: str, audio_path: str, out_path: str):
    with VideoFileClip(video_path) as v:
        with AudioFileClip(audio_path) as a:
            a = a.set_duration(v.duration)
            v = v.set_audio(a)
            v.write_videofile(out_path, codec="libx264", audio_codec="aac", fps=v.fps or 24, verbose=False, logger=None)

def main():
    st.title("🎬 Instant Dub (MVP)")
    st.write("Upload a short **English** video (30–60s recommended). Pick a target language and get a dubbed MP4.")

    lang = st.selectbox(
        "Target language",
        options=list(SUPPORTED_TARGETS.keys()),
        format_func=lambda k: f"{SUPPORTED_TARGETS[k]['label']} ({k})",
        index=0
    )

    video_file = st.file_uploader(
        "Video file (mp4/mov/mkv/webm/m4v)",
        type=["mp4", "mov", "mkv", "webm", "m4v"]
    )

    if st.button("Convert to dubbed video", disabled=not video_file):
        if not video_file:
            st.warning("Please upload a video first.")
            st.stop()

        with st.status("Processing...", expanded=True) as status:
            try:
                status.update(label="Saving upload...", state="running")
                with tempfile.TemporaryDirectory() as tmp:
                    in_path = os.path.join(tmp, "input.mp4")
                    with open(in_path, "wb") as f:
                        f.write(video_file.read())

                    status.write("Extracting audio…")
                    wav_path = os.path.join(tmp, "audio.wav")
                    with VideoFileClip(in_path) as v:
                        v.audio.write_audiofile(wav_path, verbose=False, logger=None)

                    status.write("Transcribing (English → text)…")
                    text_en = transcribe_english(wav_path)

                    status.write(f"Translating (text → {SUPPORTED_TARGETS[lang]['label']})…")
                    text_tr = translate_text(text_en, lang)

                    status.write(f"Synthesizing voice ({lang})…")
                    mp3_path = os.path.join(tmp, "dub.mp3")
                    synthesize_tts(text_tr, lang, mp3_path)

                    status.write("Merging dubbed audio back into video…")
                    out_path = os.path.join(tmp, f"dubbed_{uuid.uuid4().hex}.mp4")
                    mux_audio(in_path, mp3_path, out_path)

                    with open(out_path, "rb") as f:
                        data = f.read()

                status.update(label="Done!", state="complete")

                st.success(f"Dubbed to {SUPPORTED_TARGETS[lang]['label']}")
                st.video(data)

                st.download_button(
                    "Download dubbed MP4",
                    data=data,
                    file_name=f"dubbed_{lang}.mp4",
                    mime="video/mp4"
                )
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Runtime error: {e}")

    st.caption("Note: MVP assumes source audio is **English** and generates a single continuous dubbed track (no lip sync). Keep the first tests short.")

if __name__ == "__main__":
    main()
