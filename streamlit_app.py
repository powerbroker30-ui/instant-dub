import streamlit as st
from transformers import pipeline
from gtts import gTTS
import moviepy.editor as mp
import tempfile
import os
import whisper
import time

st.set_page_config(page_title="Instant Dub App", page_icon="🎬", layout="centered")

st.title("🎬 Instant Dub App")
st.write("Upload a video, auto-detect language, pick a target language, and get a dubbed version!")

# -------------------
# Upload video
# -------------------
uploaded_file = st.file_uploader("📂 Upload a video file", type=["mp4", "mov", "avi", "mkv"])

# -------------------
# Language selection
# -------------------
target_lang = st.selectbox(
    "🌍 Select target language",
    ["en", "hi", "fr", "es", "de", "it", "ja"],  # English, Hindi, French, Spanish, German, Italian, Japanese
    format_func=lambda x: {
        "en": "English",
        "hi": "Hindi",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese"
    }[x]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("🚀 Generate Dubbed Video"):
        progress = st.progress(0, text="⏳ Starting process...")

        try:
            # -------------------
            # Save uploaded file
            # -------------------
            progress.progress(10, text="📂 Saving uploaded video...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            time.sleep(1)

            # -------------------
            # Extract audio
            # -------------------
            progress.progress(25, text="🎵 Extracting audio...")
            video_clip = mp.VideoFileClip(video_path)
            audio_path = tempfile.mktemp(suffix=".wav")
            video_clip.audio.write_audiofile(audio_path, codec="pcm_s16le")
            time.sleep(1)

            # -------------------
            # Transcription + Auto-detect language
            # -------------------
            progress.progress(45, text="📝 Transcribing & detecting language...")
            model = whisper.load_model("base")
            transcription = model.transcribe(audio_path)
            input_text = transcription["text"]
            detected_lang = transcription["language"]

            st.info(f"📝 Transcribed text ({detected_lang.upper()}): {input_text}")
            time.sleep(1)

            # -------------------
            # Translation
            # -------------------
            progress.progress(60, text="🌐 Translating text...")
            translation_models = {
                ("en", "hi"): "Helsinki-NLP/opus-mt-en-hi",
                ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
                ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
                ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
                ("en", "it"): "Helsinki-NLP/opus-mt-en-it",
                ("en", "ja"): "Helsinki-NLP/opus-mt-en-jap",
                ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
                ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
                ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
                ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
                ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
                ("ja", "en"): "Helsinki-NLP/opus-mt-jap-en",
            }

            if detected_lang == target_lang:
                st.warning("⚠️ Source and target languages are the same. Skipping translation.")
                translated = input_text
            else:
                model_name = translation_models.get((detected_lang, target_lang))
                if not model_name:
                    st.error("❌ Translation model not available for this language pair.")
                    translated = input_text
                else:
                    translator = pipeline("translation", model=model_name)
                    translated = translator(input_text, max_length=400)[0]['translation_text']

            st.success(f"✅ Final Text in {target_lang.upper()}: {translated}")
            time.sleep(1)

            # -------------------
            # Text-to-Speech
            # -------------------
            progress.progress(75, text="🔊 Generating speech...")
            tts = gTTS(translated, lang=target_lang if target_lang != "en" else "en")
            dub_audio_path = tempfile.mktemp(suffix=".mp3")
            tts.save(dub_audio_path)
            time.sleep(1)

            # -------------------
            # Merge dubbed audio + video
            # -------------------
            progress.progress(90, text="🎬 Merging audio & video...")
            dubbed_audio = mp.AudioFileClip(dub_audio_path)
            final_clip = video_clip.set_audio(dubbed_audio)

            output_path = tempfile.mktemp(suffix=".mp4")
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            time.sleep(1)

            # -------------------
            # Show & Download
            # -------------------
            progress.progress(100, text="✅ Done! Dubbed video ready 🎉")

            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Dubbed Video",
                    f,
                    file_name="dubbed_video.mp4",
                    mime="video/mp4"
                )

            # Cleanup
            try:
                os.remove(video_path)
                os.remove(audio_path)
                os.remove(dub_audio_path)
                os.remove(output_path)
            except:
                pass

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
