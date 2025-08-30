import streamlit as st
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import whisper

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Load universal translation model
model_name = "facebook/m2m100_418M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
translator = pipeline("translation", model=model, tokenizer=tokenizer)

# Streamlit UI
st.title("Instant Dub App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Generate Dubbed Video"):
        # Save uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        # Step 1: Transcribe using Whisper
        result = whisper_model.transcribe(video_path)
        detected_lang = result["language"]
        transcription = result["text"]

        st.subheader("Transcribed text")
        st.info(f"Detected Language: {detected_lang.upper()} - Transcription: {transcription}")

        # Step 2: Translate text
        try:
            if detected_lang != "en":  # If not English, translate to English first
                translated = translator(transcription, src_lang=detected_lang, tgt_lang="en")[0]['translation_text']
            else:
                translated = transcription
            st.success(f"Translated: {translated}")
        except Exception as e:
            st.error(f"Translation error: {str(e)}")
            st.stop()

        # Step 3: Convert translated text to speech
        tts = gTTS(translated, lang="en")
        audio_path = tempfile.mktemp(suffix=".mp3")
        tts.save(audio_path)

        # Step 4: Merge audio with video
        final_output = tempfile.mktemp(suffix=".mp4")
        videoclip = VideoFileClip(video_path)
        audioclip = AudioFileClip(audio_path)
        videoclip = videoclip.set_audio(audioclip)
        videoclip.write_videofile(final_output, codec="libx264", audio_codec="aac")

        # Step 5: Show dubbed video
        st.video(final_output)
