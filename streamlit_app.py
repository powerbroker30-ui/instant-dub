import streamlit as st
from transformers import pipeline
from gtts import gTTS
import tempfile
import os
import moviepy.editor as mp

# Title
st.title("🎬 Instant Dub Demo")
st.write("Upload a short video and get it dubbed into another language instantly!")

# Step 1: Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_file.read())

    st.video(temp_video.name)

    # Step 2: Extract audio from video
    st.write("Extracting audio...")
    video = mp.VideoFileClip(temp_video.name)
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    video.audio.write_audiofile(audio_path)

    # Step 3: Speech-to-Text (Transcription using Whisper from transformers)
    st.write("Transcribing audio with Whisper...")
    stt_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    result = stt_pipeline(audio_path)
    text = result["text"]

    st.subheader("Original Transcript")
    st.write(text)

    # Step 4: Choose target language
    target_lang = st.selectbox("Choose target language", ["en", "hi", "fr", "es"])

    # Step 5: Text-to-Speech
    st.write("Generating dubbed audio...")
    tts = gTTS(text=text, lang=target_lang)
    dubbed_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    tts.save(dubbed_audio_path)

    # Step 6: Merge dubbed audio with video
    st.write("Merging new audio with video...")
    dubbed_audio_clip = mp.AudioFileClip(dubbed_audio_path)
    final_video = video.set_audio(dubbed_audio_clip)

    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Step 7: Show final result
    st.subheader("Dubbed Video")
    st.video(output_path)

    with open(output_path, "rb") as f:
        st.download_button("Download Dubbed Video", f, file_name="dubbed_output.mp4")
