# Instant Dub (Streamlit, GitHub → Streamlit Cloud)

This is a minimal web app that dubs a short **English** video into another language using free components:
- **faster-whisper (tiny, CPU)** → speech-to-text
- **MarianMT** (Helsinki-NLP) via `transformers` → translation
- **gTTS** → text-to-speech
- **moviepy + ffmpeg** → replace audio with dubbed track

> First deployments may take a few minutes because models & dependencies download.

## Run locally (optional)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy on Streamlit Cloud
1. Push this folder to a GitHub repo (public).
2. Go to https://streamlit.io/cloud and click **New app**.
3. Connect your GitHub, pick the repo, branch, and **Main file path** = `streamlit_app.py`.
4. Click **Deploy**. (The app URL will look like `https://your-app-name.streamlit.app`).

### Notes
- Keep test videos short (30–60s).
- Source language assumed English for this MVP.
- TTS needs internet (gTTS).

