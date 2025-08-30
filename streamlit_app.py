import streamlit as st
import torch
from transformers import pipeline
from gtts import gTTS
import os
import ffmpeg

# --- Title ---
st.title("🎬 Instant Dubbing App")
st.write("Upload a video, translate/dub it with AI, and download the new version!")

# --- Upload video ---
uploaded_file = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"])

# --- Choose target language ---
target_lang = st.selectbox(
    "Choose dubbing language",
    ["en", "hi", "fr", "es", "de", "it", "ja"]
)

if uploaded_file is not None:
    with open("input_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video("input_video.mp4")

    # --- Subtitle/translation ---
    st.info("Extracting text and generating dub... Please wait ⏳")

    # (Dummy placeholder – replace with real transcription if needed)
    input_text = "Hello! Welcome to this dubbing demo."

    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-en-{target_lang}")
    translated = translator(input_text, max_length=400)[0]['translation_text']

    # --- Generate speech ---
    tts = gTTS(translated, lang=target_lang)
    tts.save("dubbed_audio.mp3")

    st.audio("dubbed_audio.mp3")

    # --- Merge video & audio with ffmpeg ---
    def merge_audio_video(video_path, audio_path, output_path):
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, output_path, vcodec='copy', acodec='aac', strict='experimental')
            .run(overwrite_output=True)
        )

    merge_audio_video("input_video.mp4", "dubbed_audio.mp3", "final_output.mp4")

    st.success("✅ Dubbing complete!")
    st.video("final_output.mp4")

    with open("final_output.mp4", "rb") as f:
        st.download_button("⬇️ Download Dubbed Video", f, file_name="dubbed_video.mp4")
