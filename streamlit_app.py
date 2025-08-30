import streamlit as st
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from moviepy.editor import VideoFileClip, AudioFileClip
import tempfile
import whisper

# Load Whisper model (for transcription)
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# Load Translation Model (M2M100)
@st.cache_resource
def load_translator():
    model_name = "facebook/m2m100_418M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_translator()

# Streamlit UI
st.title("🎬 Instant Dub App")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Generate Dubbed Video"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            video_path = temp_video.name

        # Step 1: Transcribe using Whisper
        result = whisper_model.transcribe(video_path)
        detected_lang = result["language"]
        transcription = result["text"]

        st.subheader("📝 Transcribed text")
        st.info(f"Detected Language: {detected_lang.upper()} - {transcription}")

        # Step 2: Translate text to English using M2M100
        try:
            if detected_lang != "en":
                tokenizer.src_lang = detected_lang
                encoded = tokenizer(transcription, return_tensors="pt")
                generated_tokens = model.generate(
                    **encoded,
                    forced_bos_token_id=tokenizer.get_lang_id("en")
                )
                translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            else:
                translated = transcription

            st.success(f"🌍 Translated: {translated}")
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
        st.subheader("✅ Dubbed Video")
        st.video(final_output)
