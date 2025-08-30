import streamlit as st
from transformers import pipeline
from gtts import gTTS
import moviepy.editor as mp
import tempfile
import os
import time

st.set_page_config(page_title="Instant Dub App", page_icon="🎬", layout="centered")
st.title("🎬 Instant Dub (3.13-compatible)")
st.write("Upload a video (any language), pick a target language, and get a dubbed version!")

# Load a Speech-to-Text model using transformers (works on Py 3.13)
@st.cache_resource
def load_stt():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small")

stt_pipeline = load_stt()

# Translation mapping
translation_models = {
    ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
    ("en", "ja"): "Helsinki-NLP/opus-mt-en-ja",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
}

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
target_lang = st.selectbox("Select target language", ["en", "hi", "fr", "es", "de", "it", "ja"],
                           format_func=lambda x: {"en": "English", "hi": "Hindi", "fr": "French",
                                                  "es": "Spanish", "de": "German", "it": "Italian",
                                                  "ja": "Japanese"}[x])

if uploaded_file:
    st.video(uploaded_file)
    if st.button("Generate Dubbed Video"):
        progress = st.progress(0, text="Starting...")

        tmp = tempfile.TemporaryDirectory()
        video_path = os.path.join(tmp.name, "input.mp4")
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())
        progress.progress(10, text="Saved video")
        time.sleep(0.5)

        # Extract audio
        video_clip = mp.VideoFileClip(video_path)
        audio_path = os.path.join(tmp.name, "audio.wav")
        video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
        progress.progress(30, text="Extracted audio")
        time.sleep(0.5)

        # Transcribe
        stt_res = stt_pipeline(audio_path)
        input_text = stt_res["text"]
        detected_lang = stt_res.get("language", "en")
        st.info(f"Detected Language: {detected_lang.upper()} • Transcription: {input_text}")
        progress.progress(50, text="Transcribed text")

        # Translate if needed
        if detected_lang == target_lang or (detected_lang, target_lang) not in translation_models:
            translated = input_text
            if detected_lang != target_lang:
                st.warning("No translation model for this pair, using original text.")
        else:
            trans = pipeline("translation", model=translation_models[(detected_lang, target_lang)])
            translated = trans(input_text, max_length=400)[0]["translation_text"]
            st.success(f"Translated Text: {translated}")
        progress.progress(70, text="Translated")

        # TTS
        tts = gTTS(translated, lang=target_lang)
        tts_path = os.path.join(tmp.name, "dubbed.mp3")
        tts.save(tts_path)
        progress.progress(80, text="Generated speech")
        time.sleep(0.5)

        # Merge
        dubbed = mp.AudioFileClip(tts_path)
        final = video_clip.set_audio(dubbed)
        output_path = os.path.join(tmp.name, "output.mp4")
        final.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
        progress.progress(100, text="Done! Dubbed video ready")

        st.video(output_path)
        with open(output_path, "rb") as f:
            st.download_button("Download Dubbed Video", f, file_name="dubbed.mp4", mime="video/mp4")

        tmp.cleanup()
